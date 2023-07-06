from typing import List, Optional
import torch
import gpytorch
import numpy as np

from botorch.utils import standardize
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import *
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qKnowledgeGradient
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch import kernels as gpykernels
from gpytorch import means as gpymeans

from boptimol.molecule import Molecule
from boptimol.mean_potentials import MorseMean, QuadraticMean

class Bond_Lengths(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(Bond_Lengths, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def select_next_point_bond_lengths_derivative(bounds: torch.Tensor,
        observed_X: List[List[float]], 
        observed_y: List[float],
        device: str) -> np.ndarray:
    
    """Generate the next sample to evaluate

    Uses BOTorch to pick the next sample using Expected Improvement

    Args:
        observed_X: Observed coordinates
        observed_y: Observed energies
    Returns:
        Next coordinates to try
    """
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1).to(device=device)
    gp = Bond_Lengths(observed_X, observed_y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    qKG = qKnowledgeGradient(gp, num_fantasies=4)

    
    aq = LogExpectedImprovement(gp, best_f=torch.max(observed_y))
    candidate, acq_value = optimize_acqf(
        qKG,
        bounds=bounds,
        q=2,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    return candidate.detach().numpy()[0, :]


def select_next_point_bond_lengths(bounds: torch.Tensor,
        observed_X: List[List[float]], 
        observed_y: List[float], 
        device: str) -> np.ndarray:
    
    """Generate the next sample to evaluate

    Uses BOTorch to pick the next sample using Expected Improvement

    Args:
        observed_X: Observed coordinates
        observed_y: Observed energies
    Returns:
        Next coordinates to try
    """
    
    mean_module = gpymeans.ConstantMean(batch_shape=torch.Size())
    covar_module = gpykernels.ScaleKernel(gpykernels.RBFKernel())
    
    gp = SingleTaskGP(observed_X, observed_y, covar_module=covar_module, mean_module=mean_module)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=device)
    fit_gpytorch_mll(mll)
    
    aq = LogProbabilityOfImprovement(gp, best_f=torch.max(observed_y))
    candidate, acq_value = optimize_acqf(
        aq, 
        bounds=bounds,
        q=1, 
        num_restarts=64, 
        raw_samples=64
    )
    return candidate.detach().numpy()[0, :]


def select_next_point_bond_angles(bounds: torch.Tensor,
        observed_X: List[List[float]], 
        observed_y: List[float], 
        device: str) -> np.ndarray:
    
    """Generate the next sample to evaluate

    Uses BOTorch to pick the next sample using Expected Improvement

    Args:
        observed_X: Observed coordinates
        observed_y: Observed energies
    Returns:
        Next coordinates to try
    """
    
    mean_module = gpymeans.ConstantMean(batch_shape=torch.Size())
    covar_module = gpykernels.ScaleKernel(gpykernels.RBFKernel())
    
    gp = SingleTaskGP(observed_X, observed_y, covar_module=covar_module, mean_module=mean_module)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=device)
    fit_gpytorch_mll(mll)
    
    aq = LogProbabilityOfImprovement(gp, best_f=torch.max(observed_y))
    candidate, acq_value = optimize_acqf(
        aq, 
        bounds=bounds,
        q=1, 
        num_restarts=64, 
        raw_samples=64
    )
    return candidate.detach().numpy()[0, :]



def select_next_point_dihedrals(bounds: torch.Tensor,
        observed_X: List[List[float]], 
        observed_y: List[float], 
        device: str) -> np.ndarray:
    
    """Generate the next sample to evaluate

    Uses BOTorch to pick the next sample using Expected Improvement

    Args:
        observed_X: Observed coordinates
        observed_y: Observed energies
    Returns:
        Next coordinates to try
    """
    
    mean_module = gpymeans.ConstantMean(batch_shape=torch.Size())
    covar_module = gpykernels.ScaleKernel(gpykernels.ProductStructureKernel(
                num_dims = observed_X.shape[1],
                base_kernel=gpykernels.PeriodicKernel()))
    
    gp = SingleTaskGP(observed_X, observed_y, covar_module=covar_module, mean_module=mean_module)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=device)
    fit_gpytorch_mll(mll)
    
    aq = LogExpectedImprovement(gp, best_f=torch.max(observed_y))
    candidate, acq_value = optimize_acqf(
        aq, 
        bounds=bounds,
        q=1, 
        num_restarts=64, 
        raw_samples=64
    )
    return candidate.detach().numpy()[0, :]
    
    

def select_next_points_botorch( bounds: torch.Tensor,
        observed_X: List[List[float]], 
        observed_y: List[float],
        mol: Molecule) -> np.ndarray:
    
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
    train_Y = standardize(-1* train_y[:, None])

    candidate_bond_lengths = select_next_point_bond_lengths(bounds[:, : mol.end_bonds],
                             train_X[:, : mol.end_bonds], train_Y, device)
    
    candidate_bond_angles = select_next_point_bond_angles(bounds[:, mol.end_bonds: mol.end_angles],
                             train_X[:, mol.end_bonds: mol.end_angles], train_Y, device)
    
    candidate_dihedrals = select_next_point_dihedrals(bounds[:, mol.end_angles: mol.degrees_of_freedom],
                             train_X[:, mol.end_angles: mol.degrees_of_freedom], train_Y, device)
    
    candidate = np.concatenate((candidate_bond_lengths, candidate_bond_angles, candidate_dihedrals))
    
    
    # Solve the optimization problem
    #n_sampled, n_dim = train_X.shape
    
    #kappa = 0.05
    #aq = UpperConfidenceBound(gp, kappa)
    # aq = qAnalyticProbabilityOfImprovement(gp, best_f=torch.max(train_y))
    # aq = ExpectedImprovement(gp, best_f=torch.max(train_y), maximize=True) -> Log is better
    #aq = LogExpectedImprovement(gp, best_f=torch.max(train_y))
    # aq = LogProbabilityOfImprovement(gp, best_f=torch.max(train_y))
    # aq = PosteriorMean(gp)
    # aq = ScalarizedPosteriorMean(gp)
    # aq = qExpectedImprovement(gp) -> Not Implemented: Requires a sampler and constraints
    
    ''' The best acquisition function was LogExoected Improvement. Entropy search was the potential
    candidate for Acquisition Function '''
    
    # q=1 means we only pick one point - TODO: try q>1 -> DONE
#     candidate, acq_value = optimize_acqf(
#         aq, 
#         bounds=bounds,
#         q=1, 
#         num_restarts=64, 
#         raw_samples=64
#     )
    return candidate