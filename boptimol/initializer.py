from typing import List, Optional

import numpy as np

from boptimol import Molecule


# Making initial guesses for the molecule (Active Learning)
def initial_guess(mol: Molecule, init_steps: int) -> np.ndarray:
    
    ''' Return initial coordinates guess based on the molecule (Acti)
    Args:
        mol: A Molecule Class Object
        init_step: No of initial guesses to be returned
    Returns:
        Initial Guess Coordinates to be tried
    '''
    
    start_coords = mol.parameters
    return np.random.normal(start_coords, 0.1, size=(init_steps, len(start_coords)))