"""Methods for optimizing using internal coordinates"""
import logging
from csv import DictWriter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional

from boptimol import Molecule

import torch

from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import *
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch import kernels as gpykernels

import numpy as np

logger = logging.getLogger(__name__)

# TODO: add custom models for the GP

def select_next_points_botorch(bounds: torch.Tensor,
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

    # Clip the energies if needed
    observed_y = np.clip(observed_y, -np.inf, 2 + np.log10(np.clip(observed_y, 1, np.inf)))

    # we should track the torch device
    #  .. unfortuantely "MPS" for Apple Silicon doesn't support float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs to torch arrays
    train_X = torch.tensor(observed_X, dtype=torch.float64, device=device)
    train_y = torch.tensor(observed_y, dtype=torch.float64, device=device)
    train_y = train_y[:, None]
    train_y = standardize(-1 * train_y) # make this a maximization problem

    # Make the GP
    gp = SingleTaskGP(train_X, train_y,
        covar_module=gpykernels.ScaleKernel(gpykernels.ProductStructureKernel(
        num_dims=train_X.shape[1],
        base_kernel=gpykernels.MaternKernel()
    )))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=device)
    fit_gpytorch_mll(mll)

    # Solve the optimization problem
    n_sampled, n_dim = train_X.shape
    # TODO: compare different acquisition functions
    # - for example kappa parameter for UCB
    kappa = 0.01
    aq = UpperConfidenceBound(gp, kappa)
    #aq = ExpectedImprovement(gp, best_f=torch.max(train_y), maximize=True)
    # q=1 means we only pick one point - TODO: try q>1
    candidate, acq_value = optimize_acqf(
        aq, 
        bounds=bounds,
        q=1, 
        num_restarts=64, 
        raw_samples=64
    )
    return candidate.detach().numpy()[0, :]


def run_optimization(mol: Molecule,
                     n_steps: int, 
                     init_steps: int, 
                     out_dir: Optional[Path]) -> np.ndarray:
    """Optimize the structure of a molecule by iteratively changing the dihedral angles

    Args:
        mol: Molecule object with the current geometry
        n_steps: Number of optimization steps to perform
        init_steps: Number of initial guesses to evaluate
        out_dir: Output path for logging information
    Returns:
        (Atoms) optimized parameters
    """

    # Evaluate initial point
    start_coords = mol.parameters
    start_energy = mol.energy(start_coords)
    logger.info(f'Computed initial energy: {start_energy}')

    # Begin a structure log, if output available
    if out_dir is not None:
        log_path = out_dir.joinpath('structures.csv')
        with log_path.open('w') as fp:
            writer = DictWriter(fp, ['time', 'coords', 'energy', 'ediff'])
            writer.writeheader()

        def add_entry(coords, energy):
            with log_path.open('a') as fp:
                writer = DictWriter(fp, ['time', 'coords', 'energy', 'ediff'])
                writer.writerow({
                    'time': datetime.now().timestamp(),
                    'coords': coords,
                    'energy': energy,
                    'ediff': energy - start_energy
                })
            # TODO: write the XYZ file

        add_entry(start_coords, start_energy)

    # Make some initial guesses
    # TODO: make this a function and make better guesses
    init_guesses = np.random.normal(start_coords, 0.1, size=(init_steps, len(start_coords)))
    init_energies = []
    for i, guess in enumerate(init_guesses):
        energy = mol.energy(guess)
        init_energies.append(energy - start_energy)
        logger.info(f'Evaluated initial guess {i+1}/{init_steps}. Energy-E0: {energy-start_energy}')

        if out_dir is not None:
            add_entry(guess, energy)

    # Save the initial guesses
    observed_coords = np.array([start_coords, *init_guesses.tolist()])
    observed_energies = [0.] + init_energies

    bounds = torch.tensor(mol.bounds, dtype=torch.float64)

    # Loop over many steps
    best_energy = np.min(observed_energies)
    for step in range(n_steps):
        # Make a new search space
        next_coords = select_next_points_botorch(bounds, observed_coords, observed_energies)

        # Compute the energies of those points
        energy = mol.energy(next_coords)
        logger.info(f'Evaluated energy in step {step+1}/{n_steps}. Energy-E0: {energy-start_energy}')
        if energy - start_energy < np.min(observed_energies) and out_dir is not None:
            filename = out_dir.joinpath(f'best_{step}.xyz')
            mol.set_parameters(next_coords)
            mol.write_xyz(filename)

        # Update the log
        if out_dir is not None:
            add_entry(next_coords, energy)

        # Update the search space
        observed_coords = np.vstack([observed_coords, next_coords])
        observed_energies.append(energy - start_energy)
        best_energy = np.min(observed_energies)

        # Check for convergence
        # TODO: check for RMSD of the parameters
        if np.abs(energy - best_energy) < 1e-3:
            logger.info('Converged!')
            break

    best_coords = observed_coords[np.argmin(observed_energies)]
    energy = mol.energy(best_coords)

    # Final result
    logger.info(f'Finished in step {step+1}/{n_steps}. Energy-E0: {energy-start_energy}')

    # Write the final result
    filename = out_dir.joinpath(f'best.xyz')
    mol.set_parameters(best_coords)
    mol.write_xyz(filename)
    return best_coords
