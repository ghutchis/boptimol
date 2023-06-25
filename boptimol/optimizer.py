"""Methods for optimizing using internal coordinates"""
import logging
from csv import DictWriter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional

import chemcoord as cc
import torch
import numpy as np

from boptimol import Molecule
from boptimol.GP_MODEL import select_next_points_botorch
from boptimol.initializer import initial_guess

logger = logging.getLogger(__name__)


def run_optimization(mol: Molecule, n_steps: int, 
                     init_steps: int, out_dir: Optional[Path]) -> np.ndarray:
    
    """Optimize the structure of a molecule by iteratively changing the dihedral angles

    Args:
        mol: Molecule object with the current geometry
        n_steps: Number of optimization steps to perform
        init_steps: Number of initial guesses to evaluate
        out_dir: Output path for logging information
        
    Returns:
        (Atoms) optimized parameters
    """
    
    def add_entry(coords, energy, i):
        with log_path.open('a') as fp:
            writer = DictWriter(fp, ['time', 'coords', 'energy', 'ediff'])
            writer.writerow({
                'time': datetime.now().timestamp(),
                'coords': coords,
                'energy': energy,
                'ediff': energy - start_energy
            })
        mol.write_xyz(f'{out_dir}/training_loop/{i}.xyz')

        
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

        add_entry(start_coords, start_energy, 0)
    
    init_guesses = initial_guess(mol, init_steps)
    init_energies = []
    
    for i, guess in enumerate(init_guesses):
        energy = mol.energy(guess)
        init_energies.append(energy - start_energy)
        logger.info(f'Evaluated initial guess {i+1}/{init_steps}. Energy-E0: {energy-start_energy}')

        if out_dir is not None:
            add_entry(guess, energy, f'InitialGuess_{i}')

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
            filename = out_dir.joinpath(f'/training_loop/best_{step}.xyz')
            mol.set_parameters(next_coords)
            mol.write_xyz(f'{out_dir}/training_loop/best_{step}.xyz')

        # Update the log
        if out_dir is not None:
            add_entry(next_coords, energy, f'Optim_{step+1}')

        # Update the search space
        observed_coords = np.vstack([observed_coords, next_coords])
        observed_energies.append(energy - start_energy)
        best_energy = np.min(observed_energies)

        # Check for convergence
        # TODO: check for RMSD of the parameters
        # and cc.xyz_functions.isclose(,)
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