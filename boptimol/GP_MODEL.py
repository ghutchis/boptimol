from typing import List, Optional
import torch
import numpy as np

from botorch.utils import standardize
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import *
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch import kernels as gpykernels


# TODO: add custom models for the GP

def select_next_points_botorch( bounds: torch.Tensor,
        observed_X: List[List[float]], 
        observed_y: List[float]) -> np.ndarray:
    
    """Generate the next sample to evaluate

    Uses BOTorch to pick the next sample using Expected Improvement

    Args:
        observed_X: Observed coordinates
        observed_y: Observed energies
    Returns:
        Next coordinates to try
    """

    # Clipping the energies if needed
    observed_y = np.clip(observed_y, -np.inf, 2 + np.log10(np.clip(observed_y, 1, np.inf)))

    # Tracking the torch device
    #  .. unfortuantely "MPS" for Apple Silicon doesn't support float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs to torch arrays
    train_X = torch.tensor(observed_X, dtype=torch.float64, device=device)
    train_y = torch.tensor(observed_y, dtype=torch.float64, device=device)
    
    # Making this a maximization problem
    train_y = standardize(-1* train_y[:, None])

    # Setting up the GP
    gp = SingleTaskGP(train_X, train_y,
        covar_module=gpykernels.ScaleKernel(gpykernels.ProductStructureKernel(
        num_dims=train_X.shape[1],
        base_kernel=gpykernels.MaternKernel())))
    
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=device)
    fit_gpytorch_mll(mll)

    # Solve the optimization problem
    n_sampled, n_dim = train_X.shape
    
    kappa = 0.05
    #aq = UpperConfidenceBound(gp, kappa)
    # aq = qAnalyticProbabilityOfImprovement(gp, best_f=torch.max(train_y))
    # aq = ExpectedImprovement(gp, best_f=torch.max(train_y), maximize=True) -> Log is better
    aq = LogExpectedImprovement(gp, best_f=torch.max(train_y))
    # aq = LogProbabilityOfImprovement(gp, best_f=torch.max(train_y))
    # aq = LogProbabilityOfImprovement(gp, best_f=torch.max(train_y))
    # aq = PosteriorMean(gp)
    # aq = ScalarizedPosteriorMean(gp)
    # aq = qExpectedImprovement(gp) -> Not Implemented: Requires a sampler and constraints
    
    ''' The best acquisition function was LogExoected Improvement. Entropy search was the potential
    candidate for Acquisition Function '''
    
    # q=1 means we only pick one point - TODO: try q>1 -> DONE
    candidate, acq_value = optimize_acqf(
        aq, 
        bounds=bounds,
        q=1, 
        num_restarts=64, 
        raw_samples=64
    )
    return candidate.detach().numpy()[0, :]