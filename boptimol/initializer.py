from typing import List, Optional
import logging

import numpy as np
from boptimol import Molecule

logger = logging.getLogger(__name__)

# Making initial guesses for the molecule (Active Learning)
def initial_guess(mol: Molecule, init_steps: int, out_dir: str) -> np.ndarray:
    
    ''' Return initial coordinates guess based on the molecule (Acti)
    Args:
        mol: A Molecule Class Object
        init_step: No of initial guesses to be returned
    Returns:
        Initial Guess Coordinates to be tried
    '''
    
    start_coords = mol.parameters
    start_energy = mol.energy(start_coords)
    
    init_guesses = np.random.normal(start_coords, 0.1, size=(init_steps, len(start_coords))) #TODO
    init_energies = []
    
    # Enumerating and storing all the initial internal coordinate guesses
    for i, guess in enumerate(init_guesses):
        energy = mol.energy(guess)
        init_energies.append(energy - start_energy)
        logger.info(f'Evaluated initial guess {i+1}/{init_steps}. Energy-E0: {energy-start_energy}')

        if out_dir is not None:
            mol.write_xyz(f'{out_dir}/training_loop/InitialGuess_{i}.xyz')

    init_coords_guesses = np.array([start_coords, *init_guesses.tolist()])
    init_energy_guesses = [0.] + init_energies
    
    return init_coords_guesses, init_energy_guesses