from helper_func import *
import openai
import random
import json
from tqdm import trange
from scipy.stats import norm
import numpy as np
import torch
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from botorch.acquisition import UpperConfidenceBound

SYSTEM_PROMPT = "You are an AI assistant that helps people find an maximum of a black box function."

# Black-box function optimization task
# candidate sampling and surrogate modeling prompt for LLAMBO
def _sample_one_candidate(args):
    i, history_variant_str, dim, func_desc, target_score = args
    prompt = f"""
    The following are past evaluations of a black-box function. The function is {func_desc}.
    {history_variant_str}
    The allowable ranges for x is [0, 1]^{dim}.
    Recommend a new x that can achieve the function value of {target_score}.
    Return only a single {dim}-dimensional numerical vector with the highest possible precision.
    Do not include any explanations, labels, formatting, or extra text. The response must be strictly valid JSON.
    """

    from openai import OpenAI  # ensure import in subprocess
    client = OpenAI()
    while True:
        try:
            message = []
            message.append({"role": "system", "content": SYSTEM_PROMPT})
            message.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=message,
                max_tokens=50,
                ).choices[0].message.content.strip()

            return tuple(parse_llm_point(response, dim, [0.0] * dim, [1.0] * dim))

        except (ValueError, TypeError):
            print("Invalid LLM selecting response, retrying...")
            continue

def _predict_llm_score(args):
    x, history_variant_str, dim, func_desc = args
    prompt = f"""
    The following are past evaluations of a black-box function, which is {func_desc}.
    {history_variant_str}
    The allowable ranges for x is [0, 1]^{dim}.
    Predict the function value at x = {x}.
    Return only a single numerical value. Do not include any explanations, labels, formatting, or extra text. The response must be strictly a valid floating-point number.
    """

    from openai import OpenAI
    client = OpenAI()
    while True:
        try:
            message = []
            message.append({"role": "system", "content": SYSTEM_PROMPT})
            message.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=message,
                max_tokens=10
                ).choices[0].message.content.strip()
            score = float(response)
            if not np.isfinite(score):
                raise ValueError("non-finite prediction")
            return score, tuple(x)

        except ValueError:
            print("Invalid LLM sampling response, retrying...")
            continue

# LLM warmstarting shared by LLAMBO and LLAMBO-light: suggest num_warmstart points in [0, 1]^dim
def _llm_warmstarting(query_llm, func_desc, dim, num_warmstart, objective_function):
    if objective_function is None:
        raise ValueError("Objective function must be provided for warm-starting.")

    prompt = f"""
    You are assisting me with maximize a black-box function, which is {func_desc}.
    Suggest {num_warmstart} promising starting points for this task in the range [0, 1]^{dim}.
    Return the points strictly in JSON format as a list of {dim}-dimensional vectors. Do not include any explanations, labels, formatting, or extra text. The response must be strictly valid JSON.
    """

    while True:
        llm_output = query_llm(prompt)
        try:
            warmstart_points = json.loads(llm_output)
            if not isinstance(warmstart_points, list) or len(warmstart_points) != num_warmstart:
                raise ValueError("wrong number of warmstart points")
            points = [parse_llm_point(x, dim, [0.0] * dim, [1.0] * dim) for x in warmstart_points]
            return [(tuple(x), objective_function(x)) for x in points]
        except (ValueError, TypeError):
            print("LLM warmstarting response could not be parsed! Retrying...")
            continue

# LLAMBO agent
class LLAMAGENT:
    def __init__(self, history, dim=2, alpha=0.1, num_cand=10, max_surrogate_eval=10, func_desc = 'good'):
        self.dim = dim
        self.alpha = alpha
        self.history = [(tuple(x), y) for x, y in history]  # Ensure tuple format for keys
        self.grid_results = {}  # Store grid evaluations
        self.num_cand = num_cand
        self.func_desc = func_desc
        self.max_surrogate_eval = max_surrogate_eval

    def query_llm(self, prompt, model=None, max_tokens=4000):
        message = []
        message.append({"role": "system", "content": SYSTEM_PROMPT})
        message.append({"role": "user", "content": prompt})
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=message,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    # Candidate sampling phase: one LLM query per candidate, each with a permuted history
    def sample_candidate_points(self):
        best_y = max(self.history, key=lambda x: x[1])[1]
        worst_y = min(self.history, key=lambda x: x[1])[1]
        target_score = best_y - self.alpha * (best_y - worst_y)

        args_list = []
        for i in range(self.num_cand):
            shuffled = self.history.copy()
            random.shuffle(shuffled)
            history_str = "\n".join([f"x: {h[0]}, f(x): {h[1]}" for h in shuffled])
            args_list.append((i, history_str, self.dim, self.func_desc, target_score))

        with Pool(min(cpu_count(), self.num_cand)) as pool:
            candidates = pool.map(_sample_one_candidate, args_list)
        return candidates

    # warmstarting phase
    def llm_warmstarting(self, num_warmstart=1, objective_function=None):
        return _llm_warmstarting(self.query_llm, self.func_desc, self.dim, num_warmstart, objective_function)

    # determine the next design through EI, given selected candidates
    def find_best_candidate(self):
        if not self.history:
            return None

        best_so_far = max(self.history, key=lambda x: x[1])[1]
        candidates = self.sample_candidate_points()

        self.surrogate_model(candidates)
        best_candidate = None
        best_ei = -np.inf

        for candidate in candidates:
            mean, std = self.grid_results.get(tuple(candidate), (None, None))
            ei = self.expected_improvement(mean, std, best_so_far)

            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        return best_candidate

    # surrogate sampling phase (run it parallelly)
    def surrogate_model(self, candidates):
        permuted_histories = []

        for _ in range(self.max_surrogate_eval):
            shuffled = self.history.copy()
            random.shuffle(shuffled)
            permuted_histories.append(shuffled)

        tasks = []
        for x in candidates:
            for history_variant in permuted_histories:
                history_str = "\n".join([f"x: {h[0]}, f(x): {h[1]}" for h in history_variant])
                tasks.append((x, history_str, self.dim, self.func_desc))

        # Run in parallel
        with Pool(min(cpu_count(), len(tasks))) as pool:
            results = pool.map(_predict_llm_score, tasks)

        # Group results by candidate
        grouped_scores = defaultdict(list)
        for score, x_key in results:
            grouped_scores[x_key].append(score)

        # Store in results
        for x_key, scores in grouped_scores.items():
            mean, std = np.mean(scores), np.std(scores)
            self.grid_results[x_key] = (mean, std)

    def expected_improvement(self, mean, std, best_so_far, xi=0.01):
        if mean is None or std is None:
            return -np.inf
        improvement = mean - best_so_far - xi
        z = improvement / (std + 1e-9)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        return ei

# LLAMBO-light agent
class LLAMAGENT_L:
    def __init__(self, history, dim, func_desc):
        self.dim = dim
        self.func_desc = func_desc
        self.history = [(tuple(x), y) for x, y in history]  # Ensure tuple format for keys

    def query_llm(self, prompt, model=None, max_tokens=2000):
        message = []
        message.append({"role": "system", "content": SYSTEM_PROMPT})
        message.append({"role": "user", "content": prompt})
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=message,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def llm_warmstarting(self, num_warmstart=1, objective_function=None):
        return _llm_warmstarting(self.query_llm, self.func_desc, self.dim, num_warmstart, objective_function)

    # candidate generation phase
    def sample_candidate_points(self):
        shuffled_history = self.history.copy()
        random.shuffle(shuffled_history)

        history_str = "\n".join([f"x: {x}, f(x): {y}" for x, y in shuffled_history])
        prompt = f"""
        The following are past evaluations of a black-box function, which is {self.func_desc}.
        {history_str}
        The allowable ranges for x is [0, 1]^{self.dim}.
        Based on the past data, recommend the next point to evaluate that balances exploration and exploitation:
        - Exploration means selecting a point in an unexplored or less-sampled region that is far from the previously evaluated points.
        - Exploitation means selecting a point close to the previously high-performing evaluations.
        The goal is to eventually find the global maximum. Return only a single {self.dim}-dimensional numerical vector with high precision. The response must be valid JSON with no explanations, labels, or extra formatting.
        Return only a single {self.dim}-dimensional numerical vector with the highest possible precision. Do not include any explanations, labels, formatting, or extra text.
        """

        while True:
            llm_output = self.query_llm(prompt, max_tokens=50)
            try:
                return parse_llm_point(llm_output, self.dim, [0.0] * self.dim, [1.0] * self.dim)
            except (ValueError, TypeError):
                print("LLM candidate response could not be parsed! Retrying...")
                continue

# LLM in BO main function
class LLMIBO_BFO:
    def __init__(self, method, objective, dim, desc, T=20, T_ini=None, T_rep=1, verbose=True,
                 f_opt=None, S1=10000, S_decay=2):
        self.method = method.lower()
        self.obj = objective
        self.dim = dim
        self.desc = desc
        self.T = T
        self.T_ini = T_ini if T_ini is not None else dim
        self.T_rep = T_rep
        self.verbose = verbose
        self.bounds = torch.stack([torch.zeros(dim, dtype=torch.float64), torch.ones(dim, dtype=torch.float64)])
        # f(x*) for the regret G_t = f(x*) - max_{i<=t} y_i (0 if the objective is not in GLOBAL_MAX)
        self.f_opt = f_opt if f_opt is not None else GLOBAL_MAX.get(getattr(objective, '__name__', ''), 0.0)
        # LLINBO-Constrained sample schedule S_t = ceil(S1 / t^S_decay)
        self.S1 = S1
        self.S_decay = S_decay

        self.methods = {
            'rs': self._run_rs,
            'llambo': self._run_llambo,
            'llambo_l': self._run_llambo_l,
            'llmbo': self._run_llambo_l,  # kept for backward compatibility
            'bo': self._run_bo,
            'transient': self._run_transient,
            'justify': self._run_justify,
            'constrained': self._run_constrained
        }

        if self.method not in self.methods:
            raise ValueError(f"Method '{self.method}' is not implemented.")

    def run(self):
        return self.methods[self.method]()

    def _regret(self, history):
        return self.f_opt - max(y for _, y in history)

    def _ucb_step(self, model, t):
        next_x = optimize_acqf_ucb(model, bounds=self.bounds, beta=ucb_beta(t, self.dim))
        return next_x.squeeze(0).tolist()

    def _run_rs(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="RANDOM", disable=not self.verbose):
            history = generate_ini_data(func=self.obj, n=self.T_ini, bounds=self.bounds)
            regret = [self._regret(history)]
            for _ in range(self.T):
                x = torch.rand(self.dim, dtype=torch.float64).tolist()
                y = self.obj(x)
                history.append((tuple(x), y))
                regret.append(self._regret(history))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_llambo(self):
        regrets, histories = [], []
        for g in trange(self.T_rep, desc="LLAMBO", disable=not self.verbose):
            history = LLAMAGENT([], dim=self.dim, func_desc=self.desc).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [self._regret(history)]
            for t in range(self.T):
                next_x = LLAMAGENT(history, dim=self.dim, func_desc=self.desc).find_best_candidate()
                next_y = self.obj(list(next_x))
                history.append((tuple(next_x), next_y))
                regret.append(self._regret(history))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)


    def _run_llambo_l(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep,  desc="LLAMBO-L", disable=not self.verbose):
            history = LLAMAGENT_L([], dim=self.dim, func_desc=self.desc).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [self._regret(history)]
            for _ in range(self.T):
                next_x = LLAMAGENT_L(history, dim=self.dim, func_desc=self.desc).sample_candidate_points()
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(self._regret(history))
            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_bo(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="BO", disable=not self.verbose):
            history = generate_ini_data(func=self.obj, n=self.T_ini, bounds=self.bounds)
            regret = [self._regret(history)]
            for t in range(1, self.T + 1):
                model = train_gp(history)
                next_x = self._ucb_step(model, t)
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(self._regret(history))
            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_transient(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="TRANSIENT", disable=not self.verbose):
            history = LLAMAGENT_L([], dim=self.dim, func_desc=self.desc).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [self._regret(history)]
            for t in range(1, self.T + 1):
                p_t = min(t**2 / self.T, 1)
                if np.random.rand() < p_t:
                    # z_t = 1: GP-UCB design
                    next_x = self._ucb_step(train_gp(history), t)
                else:
                    # z_t = 0: LLM design (the LLM is only queried in this case)
                    next_x = LLAMAGENT_L(history, dim=self.dim, func_desc=self.desc).sample_candidate_points()
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(self._regret(history))
            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_justify(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="JUSTIFY", disable=not self.verbose):
            history = LLAMAGENT_L([], dim=self.dim, func_desc=self.desc).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [self._regret(history)]
            # psi_0: maximum posterior variance of the GP fitted on the initial data
            psi_0 = find_max_variance_bound(train_gp(history), dim=self.dim, bounds=self.bounds)
            for t in range(1, self.T + 1):
                psi_t = psi_0 / t
                model = train_gp(history)
                beta_t = ucb_beta(t, self.dim)
                next_x_gp = optimize_acqf_ucb(model, bounds=self.bounds, beta=beta_t)
                next_x_LLM = LLAMAGENT_L(history, dim=self.dim, func_desc=self.desc).sample_candidate_points()
                ucb = UpperConfidenceBound(model, beta=beta_t)
                with torch.no_grad():
                    ucb_gp = ucb(next_x_gp).item()
                    ucb_llm = ucb(torch.tensor([next_x_LLM], dtype=torch.float64)).item()

                # reject x_LLM if its UCB is at least psi_t below the UCB maximum
                if ucb_llm <= ucb_gp - psi_t:
                    next_x = next_x_gp.squeeze(0).tolist()
                else:
                    next_x = next_x_LLM
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(self._regret(history))
            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_constrained(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="CONSTRAINED", disable=not self.verbose):
            history = LLAMAGENT_L([], dim=self.dim, func_desc=self.desc).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [self._regret(history)]
            for t in range(1, self.T + 1):
                model = train_gp(history)
                beta_t = ucb_beta(t, self.dim)
                next_x_LLM = LLAMAGENT_L(history, dim=self.dim, func_desc=self.desc).sample_candidate_points()
                x_llm = torch.tensor([next_x_LLM], dtype=torch.float64)

                # kappa_{t-1}: maximum of the posterior mean
                kappa = find_gp_maximum(model, self.bounds, num_restarts=10, raw_samples=100)
                # draw S_t samples of f(x_LLM) from F_{t-1} and retain those exceeding kappa
                S_t = max(1, int(np.ceil(self.S1 / t**self.S_decay)))
                with torch.no_grad():
                    samples = model.posterior(x_llm).rsample(sample_shape=torch.Size([S_t])).view(-1)
                retained = samples[samples > kappa]

                if len(retained) == 0:
                    # I_t is empty: fall back to GP-UCB on F_{t-1}
                    next_x = self._ucb_step(model, t)
                else:
                    fantasy_model = build_fantasy_model(model, x_llm, retained)
                    next_x = optimize_cgp_ucb(fantasy_model, bounds=self.bounds, beta=beta_t).squeeze(0).tolist()
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(self._regret(history))
            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)
