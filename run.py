import hashlib
import json
import logging
import sys

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from boptimol.molecule import Molecule
from boptimol.solver import run_optimization

logger = logging.getLogger('botimol')

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument('file', type=str,
                        help='File containing the structure to optimize')
    parser.add_argument('-e', '--energy', choices=['ani', 'b3lyp', 'b97',
                        'gfn0', 'gfn2', 'gfnff'], default='gfn2', help='Energy method')
    args = parser.parse_args()

    if args.file is None:
        raise ValueError('Must specify --file to optimize')

    name = Path(args.file).stem

    # Make an output directory
    params_hash = hashlib.sha256(str(args.__dict__).encode()).hexdigest()
    out_dir = Path(__file__).parent.joinpath(
        f'solutions/{name}-{args.energy}-{params_hash[-6:]}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_dir.joinpath('run_params.json').open('w') as fp:
        json.dump(args.__dict__, fp)

    # Set up the logging
    handlers = [logging.FileHandler(out_dir.joinpath('runtime.log')),
                logging.StreamHandler(sys.stdout)]

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)
    logger.info(f'Started optimizing file {name}')

    # Set up the calculator
    if args.energy == 'ani':
        import torchani
        calc = torchani.models.ANI2x().ase()
    elif args.energy == 'xtb' or args.energy == 'gfn2':
        from xtb.ase.calculator import XTB
        calc = XTB()  # gfn2
    elif args.energy == 'gfn0':
        from xtb.ase.calculator import XTB
        calc = XTB(method='gfn0')
    elif args.energy == 'gfnff':
        from xtb.ase.calculator import XTB
        calc = XTB(method='gfnff')
    elif args.energy == 'b3lyp':
        from ase.calculators.psi4 import Psi4
        calc = Psi4(method='b3lyp-D3MBJ2B', basis='def2-svp',
                    num_threads=4, multiplicity=1, charge=0)
    elif args.energy == 'b97':
        from ase.calculators.psi4 import Psi4
        calc = Psi4(method='b97-d3bj', basis='def2-svp',
                    num_threads=4, multiplicity=1, charge=0)
    else:
        raise ValueError(f'Unrecognized QC method: {args.energy}')
    
    # Load the molecule
    molecule = Molecule(args.file, calc)

    init_steps = 1
    n_steps = 100
    final_parameters = run_optimization(molecule, n_steps, init_steps,
                                   out_dir)

    logger.info(f'Done. Files are stored in {str(out_dir)}')
