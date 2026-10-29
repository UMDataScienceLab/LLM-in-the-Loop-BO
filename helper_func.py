import json
import math
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import get_matern_kernel_with_gamma_prior
from botorch.acquisition import UpperConfidenceBound, AcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from botorch.optim import optimize_acqf
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import warnings
import os

warnings.filterwarnings("ignore")

# Fallback: Set the key directly in the script (for debugging only)
if not os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = 'your key'

# LLM used by all agents (the paper's main experiments use gpt-3.5-turbo)
LLM_MODEL = "gpt-3.5-turbo"

# exploration schedule of Srinivas et al. (2009) used in the experiments:
# beta_t = 2 log(t D pi^2 / (6 delta)). BoTorch's UCB uses mu + sqrt(beta_t) * sigma.
def ucb_beta(t, dim, delta=0.1):
    return 2 * np.log(t * dim * np.pi**2 / (6 * delta))

# AF optimizers (for BO)
def optimize_acqf_ucb(model, bounds, beta):
    ucb = UpperConfidenceBound(model, beta=beta)
    candidate, _ = optimize_acqf(
        ucb,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=200,
    )
    return candidate

# surrogates (for BO): constant mean, Matern-5/2 kernel with ARD, standardized outputs.
# The kernel is set explicitly because SingleTaskGP's default kernel changed in BoTorch 0.12.
def train_gp(history):
    X = torch.tensor([list(x) for x, _ in history], dtype=torch.float64)
    Y = torch.tensor([[y] for _, y in history], dtype=torch.float64)  # Shape: (n, 1)
    gp = SingleTaskGP(
        X, Y,
        covar_module=get_matern_kernel_with_gamma_prior(ard_num_dims=X.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    return gp

# perform random sampling (in BO)
def generate_ini_data(func, n, bounds):
    X_train = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, bounds.shape[1])
    Y_train = torch.tensor([func(x.tolist()) for x in X_train], dtype=torch.float64).unsqueeze(-1)
    # Store as history (tuple of inputs and corresponding outputs)
    history = [(tuple(x.tolist()), y.item()) for x, y in zip(X_train, Y_train)]
    return history

# validate an LLM-suggested design and project it onto the box [lower, upper]
# (the framework requires x_LLM in X); raises ValueError if the response is not a dim-vector
def parse_llm_point(point, dim, lower, upper):
    if isinstance(point, str):
        point = json.loads(point)
    if isinstance(point, (list, tuple)) and len(point) == 1 and isinstance(point[0], (list, tuple)):
        point = point[0]
    if not isinstance(point, (list, tuple)) or len(point) != dim:
        raise ValueError(f"Expected a {dim}-dimensional vector, got {point!r}")
    values = [float(v) for v in point]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"Non-finite value in {point!r}")
    lower = [float(v) for v in lower]
    upper = [float(v) for v in upper]
    return [min(max(v, lo), hi) for v, lo, hi in zip(values, lower, upper)]

# fantasy models of LLINBO-Constrained: condition F_{t-1} on (x_LLM, f_s) for every retained
# sample f_s, keeping the GP hyperparameters and noise level of F_{t-1}. Returns a batched model
# with batch shape (|I_t|,); all fantasy models share the same posterior variance sigma^+_{t-1}.
def build_fantasy_model(model, x_llm, retained_samples):
    x_llm = x_llm.view(1, -1)
    model.posterior(x_llm)  # populate prediction caches required for conditioning
    n_s = len(retained_samples)
    X_f = x_llm.expand(n_s, 1, x_llm.shape[-1])
    Y_f = torch.as_tensor(retained_samples, dtype=x_llm.dtype).view(n_s, 1, 1)
    return model.condition_on_observations(X=X_f, Y=Y_f)

# CGP-UCB: mean of the fantasy means + beta^{1/2} * sqrt(sigma^+(x)^2 + s^2(x)),
# using the same beta^{1/2} convention as BoTorch's UCB; s^2 = 0 when |I_t| = 1
class CGPUCB(AcquisitionFunction):
    def __init__(self, fantasy_model, beta):
        super().__init__(model=fantasy_model)
        self.beta = beta

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        posterior = self.model.posterior(X.unsqueeze(-3))  # batch x n_fantasy x 1 x 1
        means = posterior.mean.squeeze(-1).squeeze(-1)
        var_in = posterior.variance.squeeze(-1).squeeze(-1).mean(dim=-1)
        if means.shape[-1] > 1:
            var_between = means.var(dim=-1, unbiased=True)
        else:
            var_between = torch.zeros_like(var_in)
        return means.mean(dim=-1) + math.sqrt(self.beta) * torch.sqrt(var_in + var_between)

# perform CGP-UCB (in LLINBO-Constrained)
def optimize_cgp_ucb(fantasy_model, bounds, beta, num_restarts=20, raw_samples=200):
    candidate, _ = optimize_acqf(
        CGPUCB(fantasy_model, beta),
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidate

# psi_0 of LLINBO-Justify: maximum posterior variance over a grid of resolution^dim points
# (a scrambled Sobol set of max_points points is used when the grid is larger than max_points)
def find_max_variance_bound(model, bounds, dim=2, resolution=10, max_points=10**6):
    bounds = bounds.to(torch.float64)
    if resolution ** dim <= max_points:
        axes = [torch.linspace(bounds[0, i], bounds[1, i], resolution, dtype=torch.float64) for i in range(dim)]
        mesh = torch.meshgrid(*axes, indexing='ij')
        points = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    else:
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
        points = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(100000, dtype=torch.float64)

    model.eval()
    max_var = -np.inf
    with torch.no_grad():
        for chunk in torch.split(points, 50000):
            # evaluate each point as its own batch so only marginal variances are computed
            variances = model.posterior(chunk.unsqueeze(-2)).variance
            max_var = max(max_var, variances.max().item())
    return max_var

# determine the posterior maximum (in LLINBO-Constrained)
def find_gp_maximum(model, bounds, num_restarts=10, raw_samples=100):

    posterior_mean = PosteriorMean(model)

    best_x, best_obj = optimize_acqf(
        acq_function=posterior_mean,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return best_obj.item()  # Return as a Python float

# simulation functions (all rescaled to [0, 1]^D and written as maximization problems)
def branin_function(x):

    x = np.array(x)
    assert x.shape == (2,), "Input must be a 2D vector"

    # Rescale from [0, 1]^2 to [-5, 10] × [0, 15]
    x1 = x[0] * 15 - 5
    x2 = x[1] * 15

    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)
    return -(term1 + term2 + s)  # Negate for maximization

def hartmann4(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10.0, 3.0, 17.0, 3.5],
        [0.05, 10.0, 17.0, 0.1],
        [3.0, 3.5, 1.7, 10.0],
        [17.0, 8.0, 0.05, 10.0]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124],
        [2329, 4135, 8307, 3736],
        [2348, 1451, 3522, 2883],
        [4047, 8828, 8732, 5743]
    ])

    x = np.array(x)
    assert x.shape == (4,), "Input must be a 4-dimensional vector"

    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        outer += alpha[i] * np.exp(-inner)

    # Hartmann is usually minimized as -outer, so the maximization form is +outer
    return outer

def levy_function(x):
    x = np.array(x) * 20 - 10
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return -term1 - term2 - term3

def ackley6(x, a=20, b=0.2, c=2 * np.pi):
    x = np.array(x)
    assert x.shape == (6,), "Input must be a 6-dimensional vector"

    # Rescale from [0, 1] to [-32.768, 32.768]
    x = x * 65.536 - 32.768

    d = x.shape[0]
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)

    return -sum_sq_term - cos_term - a - np.exp(1)

def rastrigin2d(x, A=10):
    x = np.array(x)
    assert x.shape == (2,), "Input must be a 2-dimensional vector"

    # Rescale from [0,1] to [-5, 5.24]
    x = x * 10.24 - 5

    d = x.shape[0]
    value = A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    return -value  # For maximization

def bukin_n6(x):
    x = np.array(x)
    assert x.shape == (2,), "Input must be a 2D vector"
    # Rescale [0, 1] → [-15, -5] × [-3, 3]
    x1 = x[0] * 10 - 15
    x2 = x[1] * 6 - 3
    term1 = 100 * np.sqrt(abs(x2 - 0.01 * x1**2))
    term2 = 0.01 * abs(x1 + 10)
    return -(term1 + term2)  # convert to maximization

# global maximum f(x*) of each simulation function on [0, 1]^D, used for the regret G_t
GLOBAL_MAX = {
    'branin_function': -0.397887,
    'hartmann4': 3.729841,
    'levy_function': 0.0,
    'ackley6': 0.0,
    'rastrigin2d': 0.0,
    'bukin_n6': 0.0,
}
