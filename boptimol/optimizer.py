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
    
    
    # Begin a structure log, if output available
    if out_dir is not None:
        log_path = out_dir.joinpath('structures.csv')
        with log_path.open('w') as fp:
            writer = DictWriter(fp, ['time', 'coords', 'energy', 'ediff'])
            writer.writeheader()
        
  
    # Evaluate initial point
    start_coords = mol.parameters
    start_energy = mol.energy(start_coords)
    logger.info(f'''Initial Coordinates: {start_coords} \n 
                Computed initial energy: {start_energy}''')
    bounds = torch.tensor(mol.bounds, dtype=torch.float64)
    print(f'Bounds: {bounds}')

    
    # Guessing internal structure (Active Learning)
    observed_coords, observed_energies = initial_guess(mol, init_steps, out_dir)
    best_energy = np.min(observed_energies)

    
    ''' Bayesian Optimization Loop '''
    for step in range(n_steps):
        
        # Make a new search space
        next_coords = select_next_points_botorch(bounds, 
                                                 observed_coords, 
                                                 observed_energies,
                                                 mol)
        
        energy = mol.energy(next_coords)
        logger.info(f'''Evaluated energy in step {step+1}/{n_steps}. 
                    Energy-E0: {energy-start_energy}''')
        
        # Logging the structure
        if out_dir is not None:
            add_entry(next_coords, energy, f'Optim_{step+1}')
        
        # Logging the structure is it's the current best
        if energy - start_energy < np.min(observed_energies) and out_dir is not None:
            filename = out_dir.joinpath(f'/training_loop/best_{step}.xyz')
            mol.set_parameters(next_coords)
            mol.write_xyz(f'{out_dir}/training_loop/best_{step+1}.xyz')

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

    # Saving the best observed coordinates and energies
    best_coords = observed_coords[np.argmin(observed_energies)]
    energy = mol.energy(best_coords)
    logger.info(f'Finished in step {step+1}/{n_steps}. Energy-E0: {energy-start_energy}')

    # Writing the final result
    filename = out_dir.joinpath(f'best.xyz')
    mol.set_parameters(best_coords)
    mol.write_xyz(filename)
    
    return best_coords