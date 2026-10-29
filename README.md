# LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization

A general framework that wraps existing LLM-based Bayesian optimization (BO) methods with a statistical surrogate (a Gaussian process), deciding when and how to act on the LLM's suggestions and when to revert to classical BO.

[Paper (arXiv:2505.14756)](https://arxiv.org/abs/2505.14756)

## Abstract
Bayesian optimization (BO) is a sequential decision-making tool widely used for optimizing expensive black-box functions. Recently, Large Language Models (LLMs) have shown remarkable adaptability in low-data regimes, making them promising tools for black-box optimization by leveraging contextual knowledge to propose high-quality query points. However, LLMs lack explicit surrogate modeling and calibrated uncertainty, and their internal mechanisms remain opaque. This structural opacity can result in unnecessary experiments, which are costly in BO and may prevent the search from approaching the optimum. To address this, we propose LLINBO: LLM-in-the-Loop BO, a general framework that wraps existing LLM-based BO approaches, deciding when and how to act on their suggestions and when to revert to classical BO. We introduce three mechanisms for this interaction, each accompanied by a no-harm theoretical guarantee that ensures sublinear cumulative regret for common kernels while requiring no performance assumptions on the LLM. Beyond these theoretical guarantees, we show through simulation studies that LLINBO makes existing LLM-based BO more reliable while still exploiting the LLM's few-shot learning ability to outperform classical BO. We further demonstrate its practical use on a real-world, expensive 3D printing experiment.

## What is in this repository
This repository releases the implementation of the three LLINBO mechanisms and the baselines used in the paper, with an example notebook for each task. It is a code release, not a script-by-script reproduction of every experiment in the paper.

| Method key | Method |
|---|---|
| `transient` | LLINBO-Transient: the LLM design is used with probability 1 − p_t, the GP-UCB design otherwise (p_t = min(t²/T, 1)) |
| `justify` | LLINBO-Justify: the LLM design is rejected if its UCB is at least ψ_t below the UCB maximum (ψ_t = ψ_0 / t) |
| `constrained` | LLINBO-Constrained: the GP is reshaped by fantasy models conditioned on the LLM design, then CGP-UCB is maximized (S_t = ⌈S_1 / t²⌉) |
| `bo` | GP-UCB with random initial designs |
| `llambo` | LLAMBO (LLM candidate sampling + LLM surrogate + EI) |
| `llambo_l` | LLAMBO-light (the LLM directly proposes the next design); also the LLM agent used inside the three LLINBO variants |
| `rs` | Random search |

```
├── llinbo.py                      # general LLINBO runner for any agent, with user-specified schedules
├── helper_func.py                 # GP surrogate, UCB / CGP-UCB, LLM-response parsing, BBO test functions
├── LLM_agent_BBFO.py              # black-box optimization: LLM agents and LLMIBO_BFO runner
├── LLM_agent_HPT.py               # hyperparameter tuning: LLM agents and LLMIBO_HPT runner
├── AM_par_func.py                 # parallel LLM calls for the 3D printing study
├── BBFO_examples.ipynb            # example: black-box optimization
├── HPT_examples.ipynb             # example: hyperparameter tuning (data generation, loss functions, description cards)
├── 3D_printing_experiment.ipynb   # 3D printing pipeline and recorded prints
├── Black-box-opt_task_data/       # saved regret curves for the BBO task
├── Hyperparameter-tuning_task_data/  # saved regret curves and histories for the HPT task
└── 3D-printing_data/              # recorded 3D printing trajectories
```

## Installation
```bash
git clone https://github.com/UMDataScienceLab/LLM-in-the-Loop-BO.git
cd LLM-in-the-Loop-BO
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."     # your OpenAI API key
```
All LLM calls use the OpenAI API with `gpt-3.5-turbo`, as in the paper. To use another OpenAI model, change `LLM_MODEL` in `helper_func.py`. See the [OpenAI API documentation](https://platform.openai.com/docs/overview) for how to obtain a key.

**API cost:** LLAMBO makes 10 candidate queries plus 100 surrogate queries per iteration, while LLAMBO-light and the LLINBO variants make at most one query per iteration (plus one warm-start query per run). Try a small `T` and `T_rep` first.

## Using LLINBO with any agent
LLINBO's guarantees require nothing of the agent except that it proposes one design per iteration from the data observed so far. `llinbo.py` therefore wraps **any** agent (an LLM, an LLM-based BO method, a human, another optimizer, or a dummy) and makes it safe to use: whatever the agent proposes, the cumulative regret stays within the order of GP-UCB's (Theorem 1 of the paper).

An agent is any object with a `propose(history, t, bounds)` method that returns one design:
```python
from llinbo import LLINBO, RandomAgent, BBOLLMAgent, HPTLLMAgent

class MyAgent:
    def propose(self, history, t, bounds):
        # history: list of (x, y) pairs so far; t: iteration (1, 2, ...); bounds: (2, D) tensor
        best_x, _ = max(history, key=lambda h: h[1])
        return [min(max(v + 0.05, lo), hi) for v, lo, hi in zip(best_x, bounds[0].tolist(), bounds[1].tolist())]

    # optional: used to generate the initial dataset (warmstarting)
    # def initial_design(self, n, bounds): return [...]

runner = LLINBO(
    method="justify",             # "transient" | "justify" | "constrained" | "bo" | "agent" (the agent alone)
    agent=MyAgent(),
    objective=f,                  # maps a list x to a scalar
    bounds=[[0, 0], [1, 1]],      # any box; the GP works on the rescaled unit cube
    T=20,
    minimize=False,               # set True for e.g. MSE
)
histories, regrets = runner.run()
```
- The agent must not evaluate the objective itself, and its proposals are projected onto the box. LLINBO-Transient only queries the agent when it uses the agent's design, so the agent should rebuild any state it needs from `history`.
- Built-in agents: `RandomAgent()` (uninformative, as in the stress test), `BBOLLMAgent(desc, dim, kind="llambo_light" | "llambo")` for functions on [0, 1]^D, and `HPTLLMAgent(card, kind=..., budget=T)` for hyperparameter tuning.
- `init=None` (default) builds the initial dataset with the agent's `initial_design` if it has one and randomly otherwise; `n_init` sets its size (default D). With `method="bo"`, an LLM agent therefore gives LLM-initialized BO.
- Other options: `f_opt` (optimal value used for the regret, default 0), `T_rep` (replications) and `seed`.

### Custom schedules
The schedules of the three mechanisms can be given as a sequence with one value per iteration t = 1, ..., T, or as a function of t:

| Argument | Meaning | Default (paper) |
|---|---|---|
| `p_schedule` | LLINBO-Transient: probability p_t of using the GP-UCB design | min(t² / T, 1) |
| `psi_schedule` | LLINBO-Justify: tolerance ψ_t; a function may take `(t)` or `(t, psi_0)` | ψ_0 / t, with ψ_0 the largest posterior variance after the initial design |
| `S_schedule` | LLINBO-Constrained: number of posterior samples S_t | ⌈10⁴ / t²⌉ |

```python
LLINBO(method="transient", ..., T=20, p_schedule=[0.0] * 5 + [1.0] * 15)       # agent for 5 iterations, then GP
LLINBO(method="justify", ..., psi_schedule=lambda t, psi_0: psi_0 / t**0.5)
LLINBO(method="constrained", ..., S_schedule=lambda t: 1000)                   # constant S_t
```
The no-harm guarantee holds when Σ_t (1 − p_t) = O(√T), ψ_t ≥ 0 with Σ_t ψ_t = O(√T), and 1 ≤ S_t ≤ S_max. Values outside [0, 1] for p_t, negative ψ_t, or S_t < 1 raise an error.

## Black-box optimization
```python
from LLM_agent_BBFO import *

runner = LLMIBO_BFO(
    method='transient',     # any method key from the table above
    objective=bukin_n6,     # maps a list x in [0, 1]^D to a scalar to be MAXIMIZED
    dim=2,
    desc="steep, narrow, and highly non-convex landscape with a sharp valley and a unique global maximum",
    T=20,                   # number of iterations (the paper uses T = 10D)
    T_ini=2,                # size of the initial dataset (default: D)
    T_rep=1,                # number of independent replications
)
histories, regrets = runner.run()
```
- `histories` is a list of `T_rep` histories, each a list of `(x, y)` pairs; `regrets` is an array of shape `(T_rep, T + 1)` holding the best-observed regret G_t = f(x\*) − max_{i≤t} y_i.
- Built-in test functions (in `helper_func.py`): `branin_function`, `bukin_n6`, `levy_function`, `rastrigin2d` (2D), `hartmann4` (4D), `ackley6` (6D). Their global maxima are listed in `GLOBAL_MAX`.
- **Your own function:** write it on the unit cube [0, 1]^D as a maximization problem, give a short text description of its patterns (`desc`), and pass its global maximum as `f_opt=...` if known (otherwise the regret is reported relative to 0).
- LLINBO-Constrained's sample schedule can be changed with `S1` (default 10000) and `S_decay` (default 2), i.e. S_t = ⌈S1 / t^S_decay⌉.
- See `BBFO_examples.ipynb` for the description cards used in the paper.

## Hyperparameter tuning
```python
from LLM_agent_HPT import *

runner = LLMIBO_HPT(
    method='transient',
    bounds=bounds_xgb,      # torch tensor of shape (2, D): lower and upper bounds of the hyperparameters
    objective=loss,         # maps a hyperparameter list to a scalar to be MINIMIZED (e.g. CV MSE)
    dim=4,
    desc=pi_XGB,            # description card (dict), see HPT_examples.ipynb
    T=20,                   # the paper uses T = 5D
    T_ini=4,
    T_rep=1,
)
histories, regrets = runner.run()
```
- Here `regrets[r, t]` is the best (lowest) MSE observed up to iteration t.
- `HPT_examples.ipynb` generates the Piston and Robot datasets, defines the cross-validated loss functions for RF, SVR and XGBoost, and contains the bounds and description cards used in the paper.
- **Your own tuning problem:** define a loss function, its bounds, and a description card with the keys `md_name` (one of `"Random Forest"`, `"Support Vector Regression"`, `"XGBoost"`, `"Neural Net"`), `md_ndim`, `md_param`, `data_nsamp`, `data_nfeature` and `data_desc`. To tune a different model, add its hyperparameter names to `PARAM_NAMES` in `LLM_agent_HPT.py`.

## 3D printing experiment
`3D_printing_experiment.ipynb` contains the pipeline used for the physical study: warm-starting, LLAMBO, LLAMBO-light, BO over the grid of admissible slicer settings, and LLINBO-Transient. Each function proposes the next print setting from the history of recorded prints. Keep `AM_par_func.py` in the same directory; it runs LLAMBO's LLM calls in parallel. The recorded trajectories are in the notebook and in `3D-printing_data/`.

## Saved results
- `Black-box-opt_task_data/<Function>/<method>_<function>.npy`: regret arrays of shape `(10, T + 1)` (10 replications). Some Transient files are named `trensient_*`.
- `Hyperparameter-tuning_task_data/<Model>+<Data>/<model>_<data>_<method>_regret.pkl`: best-MSE arrays of shape `(10, T + 1)`; the matching `_history.pkl` files hold the evaluated configurations.
- `3D-printing_data/*.xlsx`: one recorded trajectory per method (five parameters and the measured stringing percentage).

## Citation
If you use this code, please cite:

```bibtex
@misc{chang2025llinbo,
  title={{LLINBO}: Trustworthy {LLM}-in-the-Loop {B}ayesian Optimization},
  author={Chang, Chih-Yu and Azvar, Milad and Okwudire, Chinedum and Al Kontar, Raed},
  year={2025},
  eprint={2505.14756},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.14756}
}
```

## License
MIT. See [LICENSE](LICENSE).
