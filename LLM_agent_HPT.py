import openai
import random
import json
from tqdm import trange
from scipy.stats import norm
import numpy as np
import torch
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import trange
from botorch.acquisition import UpperConfidenceBound
from helper_func import *

# functions used to make LLAMBO parallelly
def _sample_one_candidate_HPT(args):
    i, history_variant_str, func_desc, target_score = args
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
    import json

    client = OpenAI()
    
    while True:
        try:
            messages = [
            {"role": "system", "content": "You are an AI assistant that helps me maximizing the accuracy by tuning the hyperparameter in the machine learning model."},
            {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=50
            ).choices[0].message.content.strip()
            extracted_value = json.loads(response)
            extracted_value = [np.float64(v) for v in extracted_value]
            return tuple(extracted_value)

        except (ValueError, json.JSONDecodeError):
            print("Invalid LLM sampling response, retrying...")
            continue
                
def _predict_llm_score_HPT(args):
    x, history_variant_str, func_desc = args
    if func_desc['md_name']== "Random Forest":
        pred_card = f""""(max_depth, min_samples_split, min_samples_leaf, max_features): {x}"""
    elif func_desc['md_name']== "Support Vector Regression":
        pred_card = f""""(C, epsilon, gamma): {x}"""
    elif func_desc['md_name']== "XGBoost":
        pred_card = f""""(max_depth, learning_rate, subsample, colsample_bytree): {x}"""
    elif func_desc['md_name']== "Neural Net":
        pred_card = f""""(hidden_layer_sizes, alpha, learning_rate_init): {x}"""
        
    prompt = f"""
    The following are examples of the performance of a {func_desc['md_name']} measured in mean square error and the corresponding model hyperparameter configurations. 
    {history_variant_str}     
    The model is evaluated on a regression task. {func_desc['data_desc']}
    {func_desc['data_desc']}
    The dataset contains {func_desc['data_nsamp']} samples and {func_desc['data_nfeature']} features and all of the features are continuous. 
    Predict the mean square error when the model hyperparameter configurations is set to be {pred_card}. Do not include any explanations, labels, formatting, or extra text. The response must be strictly a valid floating-point number.
    """

    import json
    from openai import OpenAI
    client = OpenAI()
    while True:
        try:
            messages = [
            {"role": "system", "content": "You are an AI assistant that helps me maximizing the accuracy by tuning the hyperparameter in the machine learning model."},
            {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=50
            ).choices[0].message.content.strip()
            return float(response), tuple(x)
        
        except ValueError:
            print("Invalid LLM selecting response, retrying...")
            continue

def build_gp_model(args):
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll

    next_x_LLM, sample_val, history, lower_bounds, upper_bounds = args

    extended_history = history + [(tuple(next_x_LLM), sample_val)]
    X = torch.tensor([list(x) for x, _ in extended_history], dtype=torch.double)
    X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
    Y = torch.tensor([[y] for _, y in extended_history], dtype=torch.double)

    model = SingleTaskGP(X_scaled, Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model
# LLAMBO agent function
class LLAMAGENT_HPT:
    def __init__(self, history, func_desc, alpha=0.1, num_cand=10, max_surrogate_eval=10):
        self.alpha = alpha
        self.history = [(tuple(x), y) for x, y in history]  # Ensure tuple format for keys
        self.grid_results = {}  # Store grid evaluations
        self.num_cand = num_cand
        self.func_desc = func_desc
        self.max_surrogate_eval = max_surrogate_eval
    
    def query_llm(self, prompt, model="gpt-3.5-turbo", max_tokens=4000):
        message = []
        message.append({"role": "system", "content": "You are an AI assistant that helps me maximizing the accuracy by tuning the hyperparameter in the machine learning model."})
        message.append({"role": "user", "content": prompt})
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def sample_candidate_points(self):
        best_y = max(self.history, key=lambda x: x[1])[1]
        worst_y = min(self.history, key=lambda x: x[1])[1]
        target_score = best_y - self.alpha * (best_y - worst_y)

        permuted_histories = []

        for _ in range(self.max_surrogate_eval):
            shuffled = self.history.copy()
            random.shuffle(shuffled)
            permuted_histories.append(shuffled)
    
        # Prepare args for parallel calls
        args_list = []
        for i, history_variant in enumerate(permuted_histories[:self.num_cand]):
            if self.func_desc['md_name']== "Random Forest":
                history_str = "\n".join([f"(max_depth, min_samples_split, min_samples_leaf, max_features): {h[0]}, mean square error: {h[1]}" for h in history_variant])
            elif self.func_desc['md_name']== "Support Vector Regression":
                history_str = "\n".join([f"(C, epsilon, gamma): {h[0]}, mean square error: {h[1]}" for h in history_variant])
            elif self.func_desc['md_name']== "XGBoost":
                history_str = "\n".join([f"(max_depth, learning_rate, subsample, colsample_bytree): {h[0]}, mean square error: {h[1]}" for h in history_variant])
            elif self.func_desc['md_name']== "Neural Net":
                history_str = "\n".join([f"(hidden_layer_sizes, alpha, learning_rate_init): {h[0]}, mean square error: {h[1]}" for h in history_variant])
            args_list.append((i, history_str, self.func_desc, target_score))

        with Pool(min(cpu_count(), self.num_cand)) as pool:
            candidates = pool.map(_sample_one_candidate_HPT, args_list)

        return candidates

    def llm_warmstarting(self, objective_function=None):
        if objective_function is None:
            raise ValueError("Objective function must be provided for warm-starting.")

        prompt = f"""
        You are assisting with automated hyperparameter tuning using {self.func_desc['md_name']} for a regression task. {self.func_desc['data_desc']}
        Model performance is evaluated using mean square error.
        The following hyperparameters are tunable: {self.func_desc['md_param']}. 

        Please suggest {self.func_desc['md_ndim']} diverse yet effective configurations to initiate a Bayesian Optimization process for hyperparameter tuning. 
        **Format your response strictly as a JSON array** of {self.func_desc['md_ndim']}-dimensional numerical vectors (lists). 
        Do not include explanations, comments, or any extra text outside the JSON. The output must be strictly valid JSON.
        """

        while True:
            llm_output = self.query_llm(prompt)
            try:
                warmstart_points = json.loads(llm_output)
                if isinstance(warmstart_points, list) and all(isinstance(x, list) and len(x) == self.func_desc['md_ndim'] for x in warmstart_points):
                    history = [(tuple(x), objective_function(x)) for x in warmstart_points]
                    return history
            except json.JSONDecodeError:
                print("LLM warmstarting response could not be parsed! Retrying...")
                continue
            
    def find_best_candidate(self):
        if not self.history:
            return None
        
        best_so_far = max(self.history, key=lambda x: x[1])[1]
        candidates_nontuple = self.sample_candidate_points()
        candidates = []
        for item in candidates_nontuple:
            if isinstance(item, tuple) and len(item) == 1 and isinstance(item[0], np.ndarray):
                # unpack the array inside the tuple
                candidates.append(tuple(float(x) for x in item[0]))
            elif isinstance(item, np.ndarray):
                candidates.append(tuple(float(x) for x in item))
            else:
                candidates.append(tuple(item))

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
            for history_variant in permuted_histories[:self.max_surrogate_eval]:
                if self.func_desc['md_name']== "Random Forest":
                    history_str = "\n".join([f"(max_depth, min_samples_split, min_samples_leaf, max_features): {h[0]}, mean square error: {h[1]}" for h in history_variant])
                elif self.func_desc['md_name']== "Support Vector Regression":
                    history_str = "\n".join([f"(C, epsilon, gamma): {h[0]}, mean square error: {h[1]}" for h in history_variant])
                elif self.func_desc['md_name']== "XGBoost":
                    history_str = "\n".join([f"(max_depth, learning_rate, subsample, colsample_bytree): {h[0]}, mean square error: {h[1]}" for h in history_variant])
                elif self.func_desc['md_name']== "Neural Net":
                    history_str = "\n".join([f"(hidden_layer_sizes, alpha, learning_rate_init): {h[0]}, mean square error: {h[1]}" for h in history_variant])

                tasks.append((x, history_str, self.func_desc))

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

    def expected_improvement(self, mean, std, best_so_far, xi=0.01):
        if mean is None or std is None:
            return -np.inf
        improvement = best_so_far - mean - xi

        #improvement = mean - best_so_far - xi
        z = improvement / (std + 1e-9)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        return ei
      
class LLAMAGENT_L_HPT:
    def __init__(self, history, func_desc):
        self.func_desc = func_desc
        self.history = [(tuple(x), y) for x, y in history]  # Ensure tuple format for keys
    
    def query_llm(self, prompt, model="gpt-3.5-turbo", max_tokens=2000):
        message = []
        message.append({"role": "system", "content": "You are an AI assistant that helps me reducing the mean square error by tuning the hyperparameter in the machine learning model."})
        message.append({"role": "user", "content": prompt})
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def llm_warmstarting(self, num_warmstart=1, objective_function=None):
        if objective_function is None:
            raise ValueError("Objective function must be provided for warm-starting.")

        prompt = f"""
        You are assisting with automated hyperparameter tuning using {self.func_desc['md_name']} for a regression task. {self.func_desc['data_desc']}
        Model performance is evaluated using mean square error (MSE).
        The dataset contains {self.func_desc['data_nsamp']} samples and {self.func_desc['data_nfeature']} continuous features. 
        The following hyperparameters are tunable: {self.func_desc['md_param']}. 

        Please suggest {self.func_desc['md_ndim']} diverse yet effective configurations to initiate a Bayesian Optimization process for hyperparameter tuning. 
        Format your response strictly as a JSON array of {self.func_desc['md_ndim']}-dimensional numerical vectors (lists). 
        Do not include explanations, comments, or any extra text outside the JSON.
        """
        
        while True:
            llm_output = self.query_llm(prompt)
            try:
                warmstart_points = json.loads(llm_output)
                if isinstance(warmstart_points, list) and all(isinstance(x, list) and len(x) == self.func_desc['md_ndim'] for x in warmstart_points):
                    history = [(tuple(x), objective_function(x)) for x in warmstart_points]
                    return history
            except json.JSONDecodeError:
                print("LLM warmstarting response could not be parsed! Retrying...")
                continue
            
    def sample_candidate_points(self):
        history_variant = self.history.copy()
        random.shuffle(history_variant)

        if self.func_desc['md_name']== "Random Forest":
            history_str = "\n".join([f"(max_depth, min_samples_split, min_samples_leaf, max_features): {h[0]}, mean square error: {h[1]}" for h in history_variant])
        elif self.func_desc['md_name']== "Support Vector Regression":
            history_str = "\n".join([f"(C, epsilon, gamma): {h[0]}, mean square error: {h[1]}" for h in history_variant])
        elif self.func_desc['md_name']== "XGBoost":
            history_str = "\n".join([f"(max_depth, learning_rate, subsample, colsample_bytree): {h[0]}, mean square error: {h[1]}" for h in history_variant])
        elif self.func_desc['md_name']== "Neural Net":
            history_str = "\n".join([f"(hidden_layer_sizes, alpha, learning_rate_init): {h[0]}, mean square error: {h[1]}" for h in history_variant])

             
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

        You are on iteration {len(history_str)} out of {10*self.func_desc['md_ndim']}).
        The ultimate objective is to find the global minimum prediction mean square error. The ideal prediction mean square error is 0.
        Return only a single {self.func_desc['md_ndim']}-dimensional numerical vector with the highest possible precision. Do not include any explanations, labels, formatting, or extra text like json. The response must be strictly valid JSON.
       """
       
        while True:
            llm_output = self.query_llm(prompt, max_tokens=50)
            try:
                cand_points = json.loads(llm_output)
                return cand_points
            except json.JSONDecodeError:
                print("LLM warmstarting response could not be parsed! Retrying...")
                continue

class LLMIBO_HPT:
    def __init__(self, method, bounds, objective, dim, desc, T=20, T_ini=None, T_rep=1, verbose=True):
        self.method = method.lower()
        self.obj = objective
        self.dim = dim
        self.desc = desc
        self.T = T
        self.T_ini = T_ini if T_ini is not None else dim
        self.T_rep = T_rep
        self.verbose = verbose
        self.bounds = bounds
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

    def _run_rs(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="RANDOM", disable=not self.verbose):
            history = generate_ini_data(func=self.obj, n=self.T_ini, bounds=self.bounds)
            regret = [np.min([y for _, y in history])]
            for _ in range(self.T):
                x = torch.rand(self.dim)
                x = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * x
                x = x.tolist()
                y = self.obj(x)
                history.append((tuple(x), y))
                regret.append(np.min([y for _, y in history]))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_llambo(self):
        regrets, histories = [], []
        for g in trange(self.T_rep, desc="LLAMBO", disable=not self.verbose):
            history = LLAMAGENT_HPT([], func_desc=self.desc).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [np.min([y for _, y in history])]
            for t in range(self.T):
                while True:
                    try:
                        next_x = LLAMAGENT_HPT(history, func_desc=self.desc).find_best_candidate()
                        break  # success
                    except Exception as e:
                        print(f"Retrying at iteration {t} due to error: {e}")
                        continue  # keep retrying
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(np.min([y for _, y in history]))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_llambo_l(self):
        regrets, histories = [], []
        for _ in trange(self.T_rep, desc="LLAMBO-L", disable=not self.verbose):
            history = LLAMAGENT_L_HPT([], func_desc=self.desc).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [np.min([y for _, y in history])]
            for _ in range(self.T):
                next_x = LLAMAGENT_L_HPT(history, func_desc=self.desc).sample_candidate_points()
                next_y = self.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(np.min([y for _, y in history]))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_bo(self):
        regrets, histories = [], []
        for t in trange(self.T_rep, desc="BO", disable=not self.verbose):
            history = generate_ini_data(func=self.obj, n=self.T_ini, bounds=self.bounds)
            regret = [np.min([y for _, y in history])]
            for i in range(self.T):
                # Convert bounds to tensors
                lower_bounds = self.bounds[0]
                upper_bounds = self.bounds[1]
                X = torch.tensor([x for x, y in history], dtype=torch.float64)  # shape (n, d)
                Y = [y for x, y in history]  # leave as is
                X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
                history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
                model = train_gp(history_gp)
                beta_t = np.log((i+1)*self.dim*np.pi**2/0.1*6)*2
                next_x = optimize_acqf_ucb(model, bounds=torch.stack([torch.zeros_like(lower_bounds),torch.ones_like(upper_bounds)]), beta=beta_t)
                next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds

                next_y = self.obj(next_x.squeeze(0))
                history.append((tuple(next_x.squeeze(0).tolist()), next_y))
                regret.append(np.min([y for _, y in history]))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_transient(self):
        regrets, histories = [], []
        for t in trange(self.T_rep, desc="TRANSIENT", disable=not self.verbose):
            history = LLAMAGENT_L_HPT([], func_desc=self.desc).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [np.min([y for _, y in history])]
            for i in range(self.T):
                p_t = min(i**2/self.T,1)
                if np.random.rand() < p_t:
                    lower_bounds = self.bounds[0]
                    upper_bounds = self.bounds[1]
                    X = torch.tensor([x for x, y in history], dtype=torch.float64)  # shape (n, d)
                    Y = [y for x, y in history]  # leave as is
                    X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
                    history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
                    model = train_gp(history_gp)
                    beta_t = np.log((i+1)*self.dim*np.pi**2/0.1*6)*2
                    next_x = optimize_acqf_ucb(model, bounds=torch.stack([torch.zeros_like(lower_bounds),torch.ones_like(upper_bounds)]), beta=beta_t)
                    next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds
                    next_y = self.obj(next_x.squeeze(0))
                    history.append((tuple(next_x.squeeze(0).tolist()), next_y))
                else:
                    while True:
                        try:
                            next_x = LLAMAGENT_L_HPT(history, func_desc=self.desc).sample_candidate_points()
                            next_y_LLM = self.obj(next_x)
                            break
                        except:
                            print("call llambo failed, retrying...")
                            continue
                        
                    next_y = self.obj(next_x)
                    history.append((tuple(next_x), next_y))

                regret.append(np.min([y for _, y in history]))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_justify(self):
        regrets, histories = [], []
        for rep in trange(self.T_rep, desc="GPJ", disable=not self.verbose):
            history = LLAMAGENT_L_HPT([], func_desc=self.desc).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj)
            regret = [np.min([y for _, y in history])]
            lower_bounds = self.bounds[0]
            upper_bounds = self.bounds[1]
            X = torch.tensor([x for x, y in history], dtype=torch.float64)  # shape (n, d)
            Y = [y for x, y in history]  # leave as is
            X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
            history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
            model = train_gp(history_gp)
            max_var = find_max_variance_bound(model, bounds=torch.stack([torch.zeros_like(lower_bounds),torch.ones_like(upper_bounds)]), dim=self.dim, resolution=10)
            for t in range(self.T):
                X = torch.tensor([x for x, y in history], dtype=torch.float64)  # shape (n, d)
                Y = [y for x, y in history]  # leave as is
                X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
                history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
                model = train_gp(history_gp)
                beta_t = np.log((t+1)*self.dim*np.pi**2/0.1*6)*2
                next_x = optimize_acqf_ucb(model, bounds=torch.stack([torch.zeros_like(lower_bounds),torch.ones_like(upper_bounds)]), beta=beta_t)
                
                while True:
                    try:
                        next_x_LLM = LLAMAGENT_L_HPT(history, func_desc=self.desc).sample_candidate_points()
                        next_y_LLM = self.obj(next_x_LLM)
                        break
                    except:
                        print("call llambo_l failed, retrying...")
                        continue
                
                next_x_LLM_rescaled = ((torch.tensor(next_x_LLM, dtype=torch.float64) - lower_bounds) / (upper_bounds - lower_bounds)).tolist()
                ucb = UpperConfidenceBound(model, beta=beta_t)
                psi_t = max_var / (t+1)
                if ucb(next_x).item() > ucb(torch.tensor([next_x_LLM_rescaled], dtype=torch.float64)).item() + psi_t:
                    next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds
                    next_y = self.obj(next_x.squeeze(0).tolist())
                    history.append((tuple(next_x.squeeze(0).tolist()), next_y))

                else:
                    next_x = next_x_LLM
                    next_y = self.obj(next_x_LLM)
                    history.append((tuple(next_x), next_y))
                regret.append(np.min([y for _, y in history]))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)

    def _run_constrained(self):
        regrets, histories = [], []
        lower_bounds = self.bounds[0]
        upper_bounds = self.bounds[1]
        
        for rep in trange(self.T_rep, desc="CONSTRAINED", disable=not self.verbose):
            sraw_new = 10000
            # warmstarting
            history = LLAMAGENT_L_HPT([], func_desc=self.desc).llm_warmstarting(num_warmstart=self.T_ini, objective_function=self.obj) # generate initial dataset
            regret = [np.min([y for _, y in history])]
            for t in range(self.T):
                sraw = int(np.floor(sraw_new/(t+1)**2))
                X = torch.tensor([x for x, y in history], dtype=torch.float64)
                Y = [y for x, y in history]
                # rescale the history into unit cube
                X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
                # train F_{t-1}
                history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
                model = train_gp(history_gp)
                beta_t = np.log((t+1)*self.dim*np.pi**2/0.1*6)*2
                next_x = optimize_acqf_ucb(model, bounds=torch.stack([torch.zeros_like(lower_bounds),torch.ones_like(upper_bounds)]), beta=beta_t)
                # find LLM's suggestions
                while True:
                    try:
                        next_x_LLM = LLAMAGENT_L_HPT(history, func_desc=self.desc).sample_candidate_points()
                        next_y_LLM = self.obj(next_x_LLM)
                        break
                    except:
                        print("call LLAMBO-L failed, retrying...")
                        continue

                next_x_LLM_rescaled = ((torch.tensor(next_x_LLM, dtype=torch.float64) - lower_bounds) / (upper_bounds - lower_bounds)).tolist()
                better_samples = []
                post_max = find_gp_maximum(model, self.bounds, num_restarts=10, raw_samples=100)
                # resample s_raw times
                if sraw > 1:
                    with torch.no_grad():
                        posterior = model.posterior(torch.tensor(next_x_LLM_rescaled, dtype=torch.float64).unsqueeze(0))
                        samples = posterior.rsample(sample_shape=torch.Size([sraw]))
                    for s in samples.view(-1):
                        if s.item() > post_max:
                            better_samples.append(s.item())

                # case 1: |I_t|=0
                if len(better_samples) == 0:
                    next_x = optimize_acqf_ucb(model, bounds=torch.stack([torch.zeros_like(lower_bounds),torch.ones_like(upper_bounds)]), beta=beta_t)
                    next_x = (next_x* (upper_bounds - lower_bounds)+ lower_bounds)
                    next_y = self.obj(next_x.squeeze(0).tolist())
                    history.append((tuple(next_x.squeeze(0).tolist()), next_y))
                # if someone were retained
                else:
                      
                    args_list = [
                    (next_x_LLM, sample_val, history, lower_bounds, upper_bounds)
                    for sample_val in better_samples
                    ]

                    with Pool(min(cpu_count(), len(args_list))) as pool:
                        models = pool.map(build_gp_model, args_list)

                    # 4. Store in dictionary
                    model_dict = {i: model for i, model in enumerate(models)}
                    # processing cgp-ucb
                    next_x = select_next_design_point_bound(model_dict=model_dict, bounds=torch.stack([torch.zeros_like(lower_bounds),torch.ones_like(upper_bounds)]), beta_t=beta_t, dim=self.dim)
                    # scale back next_x
                    next_x = ((torch.tensor(next_x, dtype=torch.float64)) * (upper_bounds - lower_bounds)+ lower_bounds).tolist()
                    next_y = self.obj(next_x)
                    history.append((tuple(next_x), next_y))
                regret.append(np.min([y for _, y in history]))
                
            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)
