"""General LLINBO runner: makes any agent that proposes one design per iteration safe to use in BO.

An agent is any object with a method

    propose(history, t, bounds) -> list of D floats

where `history` is the list of past (x, y) pairs in the original units of the problem, `t` is the
iteration (starting at 1) and `bounds` is the (2, D) tensor of lower and upper bounds. The agent may
be an LLM, an LLM-based BO method, a human, another optimizer, or a dummy (e.g. `RandomAgent`).
It must not evaluate the objective itself, and it is only called when LLINBO needs its suggestion
(LLINBO-Transient calls it only when z_t = 0), so it should rebuild any state it needs from `history`.
Optionally, the agent may define

    initial_design(n, bounds) -> list of n designs

which is then used to generate the initial dataset (warmstarting).
"""
import inspect
import math
import random
import numpy as np
import torch
from tqdm import trange
from botorch.acquisition import UpperConfidenceBound
from helper_func import (train_gp, optimize_acqf_ucb, ucb_beta, parse_llm_point, find_max_variance_bound,
                         find_gp_maximum, build_fantasy_model, optimize_cgp_ucb)


def _random_points(n, bounds):
    lower, upper = bounds[0], bounds[1]
    return (lower + (upper - lower) * torch.rand(n, bounds.shape[1], dtype=torch.float64)).tolist()


# ---------------------------------------------------------------- agents

class RandomAgent:
    """Uninformative agent: proposes a uniformly random design (the stress-test agent)."""

    def propose(self, history, t, bounds):
        return _random_points(1, bounds)[0]

    def initial_design(self, n, bounds):
        return _random_points(n, bounds)


class BBOLLMAgent:
    """LLM agents of the black-box optimization task (maximization on the unit cube [0, 1]^D).

    kind = "llambo_light" (one LLM query per iteration) or "llambo" (candidate sampling + LLM surrogate + EI).
    """

    def __init__(self, desc, dim, kind="llambo_light"):
        if kind not in ("llambo_light", "llambo"):
            raise ValueError("kind must be 'llambo_light' or 'llambo'.")
        self.desc, self.dim, self.kind = desc, dim, kind

    def propose(self, history, t, bounds):
        from LLM_agent_BBFO import LLAMAGENT, LLAMAGENT_L
        if self.kind == "llambo":
            return list(LLAMAGENT(history, dim=self.dim, func_desc=self.desc).find_best_candidate())
        return LLAMAGENT_L(history, dim=self.dim, func_desc=self.desc).sample_candidate_points()

    def initial_design(self, n, bounds):
        from LLM_agent_BBFO import LLAMAGENT_L
        # the warmstarting helper pairs each point with objective_function(x); only the points are kept
        points = LLAMAGENT_L([], dim=self.dim, func_desc=self.desc).llm_warmstarting(
            num_warmstart=n, objective_function=lambda x: None)
        return [list(x) for x, _ in points]


class HPTLLMAgent:
    """LLM agents of the hyperparameter-tuning task (minimization of the MSE).

    desc is the description card (see HPT_examples.ipynb); budget, if given, is stated in the
    LLAMBO-light prompt as "You are on iteration t out of budget".
    """

    def __init__(self, desc, kind="llambo_light", budget=None):
        if kind not in ("llambo_light", "llambo"):
            raise ValueError("kind must be 'llambo_light' or 'llambo'.")
        self.desc, self.kind, self.budget = desc, kind, budget

    def propose(self, history, t, bounds):
        from LLM_agent_HPT import LLAMAGENT_HPT, LLAMAGENT_L_HPT
        if self.kind == "llambo":
            return list(LLAMAGENT_HPT(history, func_desc=self.desc, bounds=bounds).find_best_candidate())
        return LLAMAGENT_L_HPT(history, func_desc=self.desc, bounds=bounds,
                               iteration=t, budget=self.budget).sample_candidate_points()

    def initial_design(self, n, bounds):
        from LLM_agent_HPT import LLAMAGENT_L_HPT
        points = LLAMAGENT_L_HPT([], func_desc=self.desc, bounds=bounds).llm_warmstarting(
            num_warmstart=n, objective_function=lambda x: None)
        return [list(x) for x, _ in points]


# ---------------------------------------------------------------- schedules

def _resolve_schedule(schedule, T, name, extra_args=()):
    """Turn a schedule (sequence with at least T entries, or callable of t) into a list of T floats."""
    if callable(schedule):
        try:
            n_args = len(inspect.signature(schedule).parameters)
        except (TypeError, ValueError):
            n_args = 1
        values = [schedule(t, *extra_args[:max(n_args - 1, 0)]) for t in range(1, T + 1)]
    else:
        values = list(schedule)
        if len(values) < T:
            raise ValueError(f"{name} has {len(values)} entries, but T = {T} are needed.")
        values = values[:T]
    values = [float(v) for v in values]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} contains non-finite values.")
    return values


# ---------------------------------------------------------------- runner

class LLINBO:
    """LLINBO with an arbitrary agent.

    method:      "transient", "justify", "constrained", "bo" (GP-UCB) or "agent" (the agent alone).
    agent:       object with propose(history, t, bounds); may be None for method="bo".
    objective:   maps a list x (original units) to a scalar.
    bounds:      (2, D) array-like of lower and upper bounds; the GP works on the rescaled unit cube.
    T:           number of iterations; n_init: size of the initial dataset (default D).
    init:        "agent" (agent.initial_design), "random", or None (agent's initial design if it has
                 one, random otherwise; so method="bo" with an LLM agent gives LLM-initialized BO).
    minimize:    set True for minimization problems (e.g. MSE).
    p_schedule:  LLINBO-Transient's p_t (probability of using GP-UCB), default min(t^2 / T, 1).
    psi_schedule: LLINBO-Justify's tolerance psi_t, default psi_0 / t with psi_0 the largest posterior
                 variance after the initial design; a callable may take (t) or (t, psi_0).
    S_schedule:  LLINBO-Constrained's number of posterior samples S_t, default ceil(10^4 / t^2).
    Each schedule is a sequence with (at least) T entries for t = 1, ..., T, or a callable of t.
    The no-harm guarantees of the paper hold when sum_t (1 - p_t) = O(sqrt(T)), psi_t >= 0 with
    sum_t psi_t = O(sqrt(T)), and 1 <= S_t <= S_max.
    f_opt:       optimal value for the regret (default 0). The returned regret is f_opt - max_i y_i
                 for maximization and min_i y_i - f_opt for minimization.
    T_rep:       number of independent replications; seed: base random seed (replication r uses seed + r).
    """

    METHODS = ("transient", "justify", "constrained", "bo", "agent")

    def __init__(self, method, agent, objective, bounds, T, n_init=None, init=None, minimize=False,
                 p_schedule=None, psi_schedule=None, S_schedule=None, f_opt=None, T_rep=1, seed=None,
                 verbose=True):
        self.method = method.lower()
        if self.method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}.")
        if agent is None and self.method != "bo":
            raise ValueError(f"method '{self.method}' needs an agent.")
        if agent is not None and not callable(getattr(agent, "propose", None)):
            raise TypeError("agent must define propose(history, t, bounds).")

        self.bounds = torch.as_tensor(bounds, dtype=torch.float64)
        if self.bounds.ndim != 2 or self.bounds.shape[0] != 2 or not bool((self.bounds[1] > self.bounds[0]).all()):
            raise ValueError("bounds must have shape (2, D) with lower < upper in every dimension.")
        self.dim = self.bounds.shape[1]
        self.unit_bounds = torch.stack([torch.zeros(self.dim, dtype=torch.float64),
                                        torch.ones(self.dim, dtype=torch.float64)])

        if int(T) < 1:
            raise ValueError("T must be at least 1.")
        self.T = int(T)
        self.n_init = int(n_init) if n_init is not None else self.dim
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1.")

        has_init = agent is not None and callable(getattr(agent, "initial_design", None))
        if init is None:
            init = "agent" if has_init else "random"
        if init not in ("agent", "random"):
            raise ValueError("init must be 'agent', 'random' or None.")
        if init == "agent" and not has_init:
            raise ValueError("init='agent' needs an agent with initial_design(n, bounds).")
        self.init = init

        self.agent = agent
        self.obj = objective
        self.sign = -1.0 if minimize else 1.0
        self.f_opt = float(f_opt) if f_opt is not None else 0.0
        self.T_rep = int(T_rep)
        self.seed = seed
        self.verbose = verbose

        T_ = self.T
        self.p = _resolve_schedule(p_schedule if p_schedule is not None else (lambda t: min(t**2 / T_, 1.0)),
                                   self.T, "p_schedule")
        if not all(0.0 <= v <= 1.0 for v in self.p):
            raise ValueError("p_schedule values must lie in [0, 1].")
        S = _resolve_schedule(S_schedule if S_schedule is not None else (lambda t: math.ceil(1e4 / t**2)),
                              self.T, "S_schedule")
        self.S = [int(math.ceil(v)) for v in S]
        if not all(v >= 1 for v in self.S):
            raise ValueError("S_schedule values must be at least 1.")
        self.psi_schedule = psi_schedule if psi_schedule is not None else (lambda t, psi_0: psi_0 / t)
        if not callable(self.psi_schedule):
            self._check_psi(_resolve_schedule(self.psi_schedule, self.T, "psi_schedule"))

    # ---- helpers
    @staticmethod
    def _check_psi(values):
        if not all(v >= 0.0 for v in values):
            raise ValueError("psi_schedule values must be non-negative.")
        return values

    def _to_unit(self, x):
        x = torch.as_tensor(x, dtype=torch.float64).view(-1)
        return (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

    def _from_unit(self, u):
        u = torch.as_tensor(u, dtype=torch.float64).view(-1).clamp(0.0, 1.0)
        return (u * (self.bounds[1] - self.bounds[0]) + self.bounds[0]).tolist()

    def _clip(self, x):
        # accept lists, tuples, numpy arrays and tensors; project onto the box
        if isinstance(x, torch.Tensor):
            x = x.detach().reshape(-1).tolist()
        elif isinstance(x, np.ndarray):
            x = x.reshape(-1).tolist()
        return parse_llm_point(list(x), self.dim, self.bounds[0].tolist(), self.bounds[1].tolist())

    def _fit_gp(self, history):
        # the GP works on the unit cube and always maximizes sign * y
        return train_gp([(self._to_unit(x).tolist(), self.sign * y) for x, y in history])

    def _ucb_design(self, model, t):
        return self._from_unit(optimize_acqf_ucb(model, bounds=self.unit_bounds, beta=ucb_beta(t, self.dim)))

    def _agent_design(self, history, t):
        return self._clip(self.agent.propose(list(history), t, self.bounds.clone()))

    def _regret(self, history):
        ys = [y for _, y in history]
        return self.f_opt - max(ys) if self.sign > 0 else min(ys) - self.f_opt

    def _initial_history(self):
        if self.init == "agent":
            points = list(self.agent.initial_design(self.n_init, self.bounds.clone()))
            if len(points) != self.n_init:
                raise ValueError(f"initial_design returned {len(points)} designs, expected {self.n_init}.")
            points = [self._clip(x) for x in points]
        else:
            points = _random_points(self.n_init, self.bounds)
        return [(tuple(x), float(self.obj(x))) for x in points]

    # ---- one iteration of each method
    def _step_transient(self, history, t, psi):
        if np.random.rand() < self.p[t - 1]:
            return self._ucb_design(self._fit_gp(history), t)  # z_t = 1
        return self._agent_design(history, t)                   # z_t = 0: only now is the agent queried

    def _step_justify(self, history, t, psi):
        model = self._fit_gp(history)
        beta_t = ucb_beta(t, self.dim)
        u_gp = optimize_acqf_ucb(model, bounds=self.unit_bounds, beta=beta_t)
        x_agent = self._agent_design(history, t)
        ucb = UpperConfidenceBound(model, beta=beta_t)
        with torch.no_grad():
            ucb_gp = ucb(u_gp).item()
            ucb_agent = ucb(self._to_unit(x_agent).view(1, -1)).item()
        # reject the agent's design if its UCB is at least psi_t below the UCB maximum
        if ucb_agent <= ucb_gp - psi[t - 1]:
            return self._from_unit(u_gp)
        return x_agent

    def _step_constrained(self, history, t, psi):
        model = self._fit_gp(history)
        beta_t = ucb_beta(t, self.dim)
        x_agent = self._agent_design(history, t)
        u_agent = self._to_unit(x_agent).view(1, -1)
        # kappa_{t-1}: maximum of the posterior mean; keep the S_t posterior samples at the agent's design above it
        kappa = find_gp_maximum(model, self.unit_bounds, num_restarts=10, raw_samples=100)
        with torch.no_grad():
            samples = model.posterior(u_agent).rsample(sample_shape=torch.Size([self.S[t - 1]])).view(-1)
        retained = samples[samples > kappa]
        if len(retained) == 0:
            return self._ucb_design(model, t)  # I_t is empty: fall back to GP-UCB
        fantasy_model = build_fantasy_model(model, u_agent, retained)
        return self._from_unit(optimize_cgp_ucb(fantasy_model, bounds=self.unit_bounds, beta=beta_t))

    def _step_bo(self, history, t, psi):
        return self._ucb_design(self._fit_gp(history), t)

    def _step_agent(self, history, t, psi):
        return self._agent_design(history, t)

    def run(self):
        """Returns (histories, regrets): T_rep lists of (x, y) pairs and an array of shape (T_rep, T + 1)."""
        step = getattr(self, f"_step_{self.method}")
        histories, regrets = [], []
        for rep in trange(self.T_rep, desc=self.method.upper(), disable=not self.verbose):
            if self.seed is not None:
                random.seed(self.seed + rep)
                np.random.seed(self.seed + rep)
                torch.manual_seed(self.seed + rep)
            history = self._initial_history()
            regret = [self._regret(history)]
            psi = None
            if self.method == "justify":
                psi_0 = find_max_variance_bound(self._fit_gp(history), bounds=self.unit_bounds, dim=self.dim)
                psi = self._check_psi(_resolve_schedule(self.psi_schedule, self.T, "psi_schedule", (psi_0,)))
            for t in range(1, self.T + 1):
                x = step(history, t, psi)
                history.append((tuple(x), float(self.obj(x))))
                regret.append(self._regret(history))
            histories.append(history)
            regrets.append(regret)
        return histories, np.array(regrets)
