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
from helper_func import *

SYSTEM_PROMPT_HPT = "You are an AI assistant that helps me reducing the mean square error by tuning the hyperparameter in the machine learning model."

# names of the tunable hyperparameters of each supported model (used in the Data Card)
PARAM_NAMES = {
    "Random Forest": "(max_depth, min_samples_split, min_samples_leaf, max_features)",
    "Support Vector Regression": "(C, epsilon, gamma)",
    "XGBoost": "(max_depth, learning_rate, subsample, colsample_bytree)",
    "Neural Net": "(hidden_layer_sizes, alpha, learning_rate_init)",
}

def _param_names(func_desc):
    if func_desc['md_name'] not in PARAM_NAMES:
        raise ValueError(f"Unsupported model '{func_desc['md_name']}'.")
    return PARAM_NAMES[func_desc['md_name']]

# Data Card: past configurations and their MSE
def _format_history(func_desc, history):
    names = _param_names(func_desc)
    return "\n".join([f"{names}: {h[0]}, mean square error: {h[1]}" for h in history])

# functions used to make LLAMBO parallelly
def _sample_one_candidate_HPT(args):
    i, history_variant_str, func_desc, target_score, lower, upper = args
    prompt = f"""
    The following are examples of the performance of a {func_desc['md_name']} measured in mean square error and the corresponding model hyperparameter configurations.
    {history_variant_str}
    The model is evaluated on a regression task. {func_desc['data_desc']}
    The allowable ranges for the hyperparameters are: {func_desc['md_param']}.
    Recommend a configuration that can achieve the target mean square error of {target_score}, and each dimension must strictly within the allowable range specified above.
    Return only a single {func_desc['md_ndim']}-dimensional numerical vector with the highest possible precision.
    The response need to be a list and must be strictly valid JSON.
    Do not include any explanations, labels, formatting, or extra text like jsom.
    """
    from openai import OpenAI  # ensure import in subprocess

    client = OpenAI()

    while True:
        try:
            messages = [
            {"role": "system", "content": SYSTEM_PROMPT_HPT},
            {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=50
            ).choices[0].message.content.strip()
            return tuple(parse_llm_point(response, func_desc['md_ndim'], lower, upper))

        except (ValueError, TypeError):
            print("Invalid LLM sampling response, retrying...")
            continue

def _predict_llm_score_HPT(args):
    x, history_variant_str, func_desc = args
    pred_card = f"{_param_names(func_desc)}: {x}"

    prompt = f"""
    The following are examples of the performance of a {func_desc['md_name']} measured in mean square error and the corresponding model hyperparameter configurations.
    {history_variant_str}
    The model is evaluated on a regression task. {func_desc['data_desc']}
    The dataset contains {func_desc['data_nsamp']} samples and {func_desc['data_nfeature']} features and all of the features are continuous.
    Predict the mean square error when the model hyperparameter configurations is set to be {pred_card}. Do not include any explanations, labels, formatting, or extra text. The response must be strictly a valid floating-point number.
    """

    from openai import OpenAI
    client = OpenAI()
    while True:
        try:
            messages = [
            {"role": "system", "content": SYSTEM_PROMPT_HPT},
            {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=50
            ).choices[0].message.content.strip()
            score = float(response)
            if not np.isfinite(score):
                raise ValueError("non-finite prediction")
            return score, tuple(x)

        except ValueError:
            print("Invalid LLM selecting response, retrying...")
            continue

# LLM warmstarting shared by LLAMBO and LLAMBO-light: suggest num_warmstart configurations
def _llm_warmstarting_HPT(query_llm, func_desc, bounds, num_warmstart, objective_function):
    if objective_function is None:
        raise ValueError("Objective function must be provided for warm-starting.")

    prompt = f"""
    You are assisting with automated hyperparameter tuning using {func_desc['md_name']} for a regression task. {func_desc['data_desc']}
    Model performance is evaluated using mean square error (MSE).
    The dataset contains {func_desc['data_nsamp']} samples and {func_desc['data_nfeature']} continuous features.
    The following hyperparameters are tunable: {func_desc['md_param']}.

    Please suggest {num_warmstart} diverse yet effective configurations to initiate a Bayesian Optimization process for hyperparameter tuning.
    Format your response strictly as a JSON array of {func_desc['md_ndim']}-dimensional numerical vectors (lists).
    Do not include explanations, comments, or any extra text outside the JSON.
    """

    while True:
        llm_output = query_llm(prompt)
        try:
            warmstart_points = json.loads(llm_output)
            if not isinstance(warmstart_points, list) or len(warmstart_points) != num_warmstart:
                raise ValueError("wrong number of warmstart points")
            points = [parse_llm_point(x, func_desc['md_ndim'], bounds[0].tolist(), bounds[1].tolist()) for x in warmstart_points]
            return [(tuple(x), objective_function(x)) for x in points]
        except (ValueError, TypeError):
            print("LLM warmstarting response could not be parsed! Retrying...")
            continue

# LLAMBO agent function (minimizes the MSE)
class LLAMAGENT_HPT:
    def __init__(self, history, func_desc, bounds, alpha=0.1, num_cand=10, max_surrogate_eval=10):
        self.alpha = alpha
        self.history = [(tuple(x), y) for x, y in history]  # Ensure tuple format for keys
        self.grid_results = {}  # Store grid evaluations
        self.num_cand = num_cand
        self.func_desc = func_desc
        self.bounds = bounds
        self.max_surrogate_eval = max_surrogate_eval

    def query_llm(self, prompt, model=None, max_tokens=4000):
        message = []
        message.append({"role": "system", "content": SYSTEM_PROMPT_HPT})
        message.append({"role": "user", "content": prompt})
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=message,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def sample_candidate_points(self):
        # target score for minimization: min + alpha * (max - min)
        best_y = min(self.history, key=lambda x: x[1])[1]
        worst_y = max(self.history, key=lambda x: x[1])[1]
        target_score = best_y + self.alpha * (worst_y - best_y)

        lower, upper = self.bounds[0].tolist(), self.bounds[1].tolist()
        args_list = []
        for i in range(self.num_cand):
            shuffled = self.history.copy()
            random.shuffle(shuffled)
            args_list.append((i, _format_history(self.func_desc, shuffled), self.func_desc, target_score, lower, upper))

        with Pool(min(cpu_count(), self.num_cand)) as pool:
            candidates = pool.map(_sample_one_candidate_HPT, args_list)

        return candidates

    def llm_warmstarting(self, num_warmstart=None, objective_function=None):
        num_warmstart = num_warmstart if num_warmstart is not None else self.func_desc['md_ndim']
        return _llm_warmstarting_HPT(self.query_llm, self.func_desc, self.bounds, num_warmstart, objective_function)

    def find_best_candidate(self):
        if not self.history:
            return None

        best_so_far = min(self.history, key=lambda x: x[1])[1]
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

    def surrogate_model(self, candidates):
        # Prepare all tasks (candidate x permutation)
        permuted_histories = []
        for _ in range(self.max_surrogate_eval):
            shuffled = self.history.copy()
            random.shuffle(shuffled)
            permuted_histories.append(shuffled)
        tasks = []
        for x in candidates:
            for history_variant in permuted_histories:
                tasks.append((x, _format_history(self.func_desc, history_variant), self.func_desc))

        # Run in parallel
        with Pool(min(cpu_count(), len(tasks))) as pool:
            results = pool.map(_predict_llm_score_HPT, tasks)
        # Group results by candidate
        grouped_scores = defaultdict(list)
        for score, x_key in results:
            grouped_scores[x_key].append(score)

        # Store in grid_results
        for x_key, scores in grouped_scores.items():
            mean, std = np.mean(scores), np.std(scores)
            self.grid_results[x_key] = (mean, std)

    # EI for minimization
    def expected_improvement(self, mean, std, best_so_far, xi=0.01):
        if mean is None or std is None:
            return -np.inf
        improvement = best_so_far - mean - xi
        z = improvement / (std + 1e-9)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        return ei

class LLAMAGENT_L_HPT:
    def __init__(self, history, func_desc, bounds, iteration=None, budget=None):
        self.func_desc = func_desc
        self.bounds = bounds
        self.history = [(tuple(x), y) for x, y in history]  # Ensure tuple format for keys
        self.iteration = iteration
        self.budget = budget

    def query_llm(self, prompt, model=None, max_tokens=2000):
        message = []
        message.append({"role": "system", "content": SYSTEM_PROMPT_HPT})
        message.append({"role": "user", "content": prompt})
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=message,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def llm_warmstarting(self, num_warmstart=None, objective_function=None):
        num_warmstart = num_warmstart if num_warmstart is not None else self.func_desc['md_ndim']
        return _llm_warmstarting_HPT(self.query_llm, self.func_desc, self.bounds, num_warmstart, objective_function)

    def sample_candidate_points(self):
        history_variant = self.history.copy()
        random.shuffle(history_variant)
        history_str = _format_history(self.func_desc, history_variant)
        iteration_str = ""
        if self.iteration is not None and self.budget is not None:
            iteration_str = f"You are on iteration {self.iteration} out of {self.budget}."

        prompt = f"""
        The following are examples of the performance of a {self.func_desc['md_name']} measured in mean square error and the corresponding model hyperparameter configurations.
        {history_str}
        The model is evaluated on a regression task. {self.func_desc['data_desc']}
        The dataset contains {self.func_desc['data_nsamp']} samples and {self.func_desc['data_nfeature']} features and all of the features are continuous.
        The allowable ranges for the hyperparameters are: {self.func_desc['md_param']}.
        Your goal is to recommend the next setting to evaluate that balances **exploration** and **exploitation**:
        - **Exploration** favors regions that are less-sampled or farther from existing evaluations.
        - **Exploitation** favors regions near previously low mean square error.
        To encourage exploration, avoid suggesting values too close to past evaluations.

        {iteration_str}
        The ultimate objective is to find the global minimum prediction mean square error. The ideal prediction mean square error is 0.
        Return only a single {self.func_desc['md_ndim']}-dimensional numerical vector with the highest possible precision. Do not include any explanations, labels, formatting, or extra text like json. The response must be strictly valid JSON.
       """

        while True:
            llm_output = self.query_llm(prompt, max_tokens=50)
            try:
                return parse_llm_point(llm_output, self.func_desc['md_ndim'], self.bounds[0].tolist(), self.bounds[1].tolist())
            except (ValueError, TypeError):
                print("LLM candidate response could not be parsed! Retrying...")
                continue

class LLMIBO_HPT:
    def __init__(self, method, bounds, objective, dim, desc, T=20, T_ini=None, T_rep=1, verbose=True,
                 S1=10000, S_decay=2):
        self.method = method.lower()
        self.obj = objective
        self.dim = dim
        self.desc = desc
        self.T = T
        self.T_ini = T_ini if T_ini is not None else dim
        self.T_rep = T_rep
        self.verbose = verbose
        self.bounds = bounds.to(torch.float64)
        self.unit_bounds = torch.stack([torch.zeros(dim, dtype=torch.float64), torch.ones(dim, dtype=torch.float64)])
        # LLINBO-Constrained sample schedule S_t = ceil(S1 / t^S_decay)
        self.S1 = S1
        self.S_decay = S_decay
        self.methods = {
            'rs': self._run_rs,
            'llambo': self._run_llambo,
            'llambo_l': self._run_llambo_l,
            'bo': self._run_bo,
            'transient': self._run_transient,
            'justify': self._run_justify,
            'constrained': self._run_constrained
        }

        if self.method not in self.methods:
            raise ValueError(f"Method '{self.method}' is not implemented.")

    def run(self):
        return self.methods[self.method]()

    # the GP works on the unit cube and maximizes -MSE
    def _to_unit(self, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        return (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

    def _from_unit(self, u):
        u = torch.as_tensor(u, dtype=torch.float64).view(-1)
        return (u * (self.bounds[1] - self.bounds[0]) + self.bounds[0]).tolist()

    def _fit_gp(self, history):
        return train_gp([(self._to_unit(x).tolist(), -y) for x, y in history])

    def _ucb_step(self, model, t):
        next_u = optimize_acqf_ucb(model, bounds=self.unit_bounds, beta=ucb_beta(t, self.dim))
        return self._from_unit(next_u)

    def _llm_l(self, history, t):
        return LLAMAGENT_L_HPT(history, func_desc=self.desc, bounds=self.bounds, iteration=t, budget=self.T)

    @staticmethod
    def _regret(history):
        return min(y for _, y in history)

    def _run_rs(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="RANDOM", disable=not self.verbose):
            history = generate_ini_data(func=self.obj, n=self.T_ini, bounds=self.bounds)
            regret = [self._regret(history)]
            for _ in range(self.T):
                x = torch.rand(self.dim, dtype=torch.float64)
                x = (self.bounds[0] + (self.bounds[1] - self.bounds[0]) * x).tolist()
                y = self.obj(x)
                history.append((tuple(x), y))
                regret.append(self._regret(history))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_llambo(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="LLAMBO", disable=not self.verbose):
            history = LLAMAGENT_HPT([], func_desc=self.desc, bounds=self.bounds).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [self._regret(history)]
            for _ in range(self.T):
                next_x = list(LLAMAGENT_HPT(history, func_desc=self.desc, bounds=self.bounds).find_best_candidate())
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(self._regret(history))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_llambo_l(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="LLAMBO-L", disable=not self.verbose):
            history = self._llm_l([], None).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [self._regret(history)]
            for t in range(1, self.T + 1):
                next_x = self._llm_l(history, t).sample_candidate_points()
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
                next_x = self._ucb_step(self._fit_gp(history), t)
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(self._regret(history))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_transient(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="TRANSIENT", disable=not self.verbose):
            history = self._llm_l([], None).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [self._regret(history)]
            for t in range(1, self.T + 1):
                p_t = min(t**2 / self.T, 1)
                if np.random.rand() < p_t:
                    # z_t = 1: GP-UCB design
                    next_x = self._ucb_step(self._fit_gp(history), t)
                else:
                    # z_t = 0: LLM design (the LLM is only queried in this case)
                    next_x = self._llm_l(history, t).sample_candidate_points()
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(self._regret(history))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_justify(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="JUSTIFY", disable=not self.verbose):
            history = self._llm_l([], None).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [self._regret(history)]
            # psi_0: maximum posterior variance of the GP fitted on the initial data
            psi_0 = find_max_variance_bound(self._fit_gp(history), bounds=self.unit_bounds, dim=self.dim, resolution=10)
            for t in range(1, self.T + 1):
                model = self._fit_gp(history)
                beta_t = ucb_beta(t, self.dim)
                next_u_gp = optimize_acqf_ucb(model, bounds=self.unit_bounds, beta=beta_t)
                next_x_LLM = self._llm_l(history, t).sample_candidate_points()

                ucb = UpperConfidenceBound(model, beta=beta_t)
                psi_t = psi_0 / t
                with torch.no_grad():
                    ucb_gp = ucb(next_u_gp).item()
                    ucb_llm = ucb(self._to_unit(next_x_LLM).view(1, -1)).item()
                # reject x_LLM if its UCB is at least psi_t below the UCB maximum
                if ucb_llm <= ucb_gp - psi_t:
                    next_x = self._from_unit(next_u_gp)
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
            # warmstarting
            history = self._llm_l([], None).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [self._regret(history)]
            for t in range(1, self.T + 1):
                # train F_{t-1} on the unit cube
                model = self._fit_gp(history)
                beta_t = ucb_beta(t, self.dim)
                # find LLM's suggestions
                next_x_LLM = self._llm_l(history, t).sample_candidate_points()
                u_llm = self._to_unit(next_x_LLM).view(1, -1)

                # kappa_{t-1}: maximum of the posterior mean over the (unit-cube) design space
                kappa = find_gp_maximum(model, self.unit_bounds, num_restarts=10, raw_samples=100)
                # draw S_t samples of f(x_LLM) from F_{t-1} and retain those exceeding kappa
                S_t = max(1, int(np.ceil(self.S1 / t**self.S_decay)))
                with torch.no_grad():
                    samples = model.posterior(u_llm).rsample(sample_shape=torch.Size([S_t])).view(-1)
                retained = samples[samples > kappa]

                if len(retained) == 0:
                    # case 1: |I_t|=0, fall back to GP-UCB on F_{t-1}
                    next_x = self._ucb_step(model, t)
                else:
                    # fantasy models sharing the hyperparameters of F_{t-1}, then CGP-UCB
                    fantasy_model = build_fantasy_model(model, u_llm, retained)
                    next_u = optimize_cgp_ucb(fantasy_model, bounds=self.unit_bounds, beta=beta_t)
                    next_x = self._from_unit(next_u)
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(self._regret(history))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)
