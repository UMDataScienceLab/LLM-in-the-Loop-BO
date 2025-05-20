import numpy as np
import torch
import pickle
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import warnings
import os
import matplotlib.pyplot as plt
from botorch.acquisition.analytic import PosteriorMean

warnings.filterwarnings("ignore")

# Fallback: Set the key directly in the script (for debugging only)
if not os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = 'your key'

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

# surrogates (for BO)
def train_gp(history):
    # Extract and convert to tensors
    X = torch.tensor([list(x) for x, _ in history], dtype=torch.float64)
    Y = torch.tensor([[y] for _, y in history], dtype=torch.float64)  # Shape: (n, 1)
    # Fit GP
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp

# perform random sampling (in BO)
def generate_ini_data(func, n, bounds):
    X_train = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, bounds.shape[1])
    Y_train = torch.tensor([func(x.tolist()) for x in X_train], dtype=torch.float64).unsqueeze(-1)
    # Store as history (tuple of inputs and corresponding outputs)
    history = [(tuple(x.tolist()), y.item()) for x, y in zip(X_train, Y_train)]
    return history

# perform CGP-UCB (in LLINBO-Constrained)
def select_next_design_point_bound(model_dict, bounds, beta_t, dim):
    # construct a grid
    axes = [torch.linspace(bounds[0, i], bounds[1, i], 40) for i in range(dim)]
    mesh = torch.meshgrid(*axes, indexing='ij')  
    grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    sampled_points = grid[torch.randperm(grid.shape[0])[:20000]]

    n_models = len(model_dict)
    n_grid = sampled_points.shape[0]
        
    mu_matrix = torch.zeros(n_models, n_grid, dtype=torch.float64)
    shared_variance = torch.zeros(n_grid, dtype=torch.float64)
    for i, model in model_dict.items():
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(sampled_points)
            mu_matrix[i] = posterior.mean.view(-1).to(torch.float64)
            if i == 0:
                shared_variance = posterior.variance.view(-1).to(torch.float64)

    mu_mean = mu_matrix.mean(dim=0)
    mu_sample_var = mu_matrix.var(dim=0, unbiased=True)
    acq = mu_mean + beta_t * torch.sqrt(shared_variance + mu_sample_var)
    best_idx = torch.argmax(acq)
    best_x = sampled_points[best_idx]
    return best_x.tolist()

# determine the posterior variance (in LLINBO-Justify)
def find_max_variance_bound(model, bounds, dim=2, resolution=10):
    # Create a grid in [0,1]^dim
    axes = [torch.linspace(bounds[0, i], bounds[1, i], 10) for i in range(dim)]
    mesh = torch.meshgrid(*axes, indexing='ij')  
    grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)

    # Evaluate variance
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(grid)
        variances = posterior.variance  # shape: (num_points,)
        max_var, idx = torch.max(variances, dim=0)
    return max_var.item()

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

# simulation functions
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
        [2649, 4135, 8307, 3736],
        [2348, 1451, 3522, 2883],
        [4047, 8828, 8732, 5743]
    ])

    x = np.array(x)
    assert x.shape == (4,), "Input must be a 4-dimensional vector"

    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        outer += alpha[i] * np.exp(-inner)

    return -outer  # It's usually minimized, so return negative

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
    
    # Rescale from [0,1] to [-5.12, 5.12]
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