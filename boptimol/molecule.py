"""Tools for computing the energy of a molecule"""
from typing import List, Tuple, Union
import os

from ase.calculators.calculator import Calculator
from ase import Atoms
import chemcoord as cc
import numpy as np

class Molecule:
    """Track the internal and Cartesian coordinates of a molecule
        during optimization.

        At the moment, this uses chemcoord to convert between
        internal and Cartesian coordinates, and ASE to compute
        energies and optionally forces.
    """

    def __init__(self, filename: str, calculator: Calculator = None):
        """Initialize the molecule from a file.

        If the file is an XYZ file, the molecule is initialized
        from the Cartesian coordinates. If the file is a zmatrix,
        the molecule is initialized from the internal coordinates.

        Args:
            filename (str): File containing the molecule.
            calculator (Calculator): ASE energy calculator.
        """

        # Check the file extension as an XYZ file or zmatrix
        # is required.
        if not filename.endswith(".xyz") and not filename.endswith(".zmat"):
            raise ValueError("Molecule must be an XYZ or zmatrix file")
        
        # Check that the file exists.
        if not os.path.exists(filename):
            raise FileNotFoundError("Molecule file not found")
        
        # Read the molecule from the file.
        if filename.endswith(".xyz"):
            self.xyz = cc.Cartesian.read_xyz(filename)
            self.zmat = self.xyz.get_zmat()
        else:
            # not sure how well this works
            # we may need to write our own zmat reader
            self.zmat = cc.ZMatrix.read_zmat(filename)
            self.xyz = self.zmat.get_cartesian()

        # store the ASE atoms object
        self.atoms = Atoms(self.xyz['atom'].to_list(), 
                           positions=self.xyz[['x','y','z']].to_numpy())
        
        # TODO: set total charge and spin multiplicity

        # Set the energy calculator.
        if calculator is None:
            raise ValueError("Energy calculator must be provided")
        
        self.calculator = calculator
        self.atoms.calc = self.calculator

        # build up the arrays
        # element, atom idx, length, atom idx, angle, idx, dihedral
        self.bonds = self.zmat.iloc[1:,2].to_numpy()
        self.angles = self.zmat.iloc[2:,4].to_numpy()
        self.dihedrals = self.zmat.iloc[3:,6].to_numpy()

        # when we stack, track the indices
        self.end_bonds = len(self.bonds)
        self.end_angles = self.end_bonds + len(self.angles)

        # get the bounds for the current parameters
        # right now, this gives bounds based on the current
        #  .. geometry with a trust range

        # TODO: adjust this based on bond elements / types
        degrees_of_freedom = self.end_angles + len(self.dihedrals)

        lower_bounds = np.zeros(degrees_of_freedom)
        upper_bounds = np.zeros(degrees_of_freedom)

        for i in range(self.end_bonds):
            lower_bounds[i] = self.bonds[i] - 0.35
            upper_bounds[i] = self.bonds[i] + 0.35

        for i in range(self.end_bonds, self.end_angles):
            idx = i - self.end_bonds
            lower_bounds[i] = self.angles[idx] - 10.0
            upper_bounds[i] = self.angles[idx] + 10.0

        for i in range(self.end_angles, degrees_of_freedom):
            idx = i - self.end_angles
            lower_bounds[i] = self.dihedrals[idx] - 30.0
            upper_bounds[i] = self.dihedrals[idx] + 30.0

        # stack the bounds to 2 x n array
        self._bounds = np.vstack((lower_bounds, upper_bounds))

    @property
    def parameters(self) -> np.ndarray:
        """Get the parameters for the molecule.
        
        Parameters are stacked as follows:
        bond lengths, angles, dihedrals

        Returns:
            np.ndarray: Parameters for the molecule.
        """

        # stack the parameters
        self.bonds = self.zmat.iloc[1:,2].to_numpy()
        self.angles = self.zmat.iloc[2:,4].to_numpy()
        self.dihedrals = self.zmat.iloc[3:,6].to_numpy()

        return np.hstack((self.bonds, self.angles, self.dihedrals))
    
    def set_parameters(self, parameters: np.ndarray) -> None:
        """Set the parameters for the molecule.
        
        Parameters are stacked as follows:
        bond lengths, angles, dihedrals

        Args:
            parameters (np.ndarray): Parameters for the molecule.
        """

        # set the bond lengths
        self.zmat.safe_iloc[1:,2] = parameters[:self.end_bonds]
        # angles
        self.zmat.safe_iloc[2:,4] = parameters[self.end_bonds:self.end_angles]
        # dihedrals
        self.zmat.safe_iloc[3:,6] = parameters[self.end_angles:]

        # update the Cartesian coordinates
        self.xyz = self.zmat.get_cartesian()
        self.atoms.positions = self.xyz[['x','y','z']].to_numpy()

    @property
    def bounds(self) -> np.ndarray:
        """Get the bounds for the parameters.        

        Parameters are stacked as follows:
        bond lengths, angles, dihedrals

        - bonds: 0.6 to 2.5 Angstroms

        Returns:
            ndarray: Bounds for the parameters.
        """

        return self._bounds
        

    def energy(self, parameters: np.ndarray) -> float:
        """Get the energy of the molecule at the given parameters.
        
        Parameters are stacked as follows:
        bond lengths, angles, dihedrals

        Args:
            parameters (np.ndarray): Parameters for the molecule.

        Returns:
            float: Energy of the molecule using the current calculator
        """

        # set the bond lengths
        self.zmat.safe_iloc[1:,2] = parameters[:self.end_bonds]
        # angles
        self.zmat.safe_iloc[2:,4] = parameters[self.end_bonds:self.end_angles]
        # dihedrals
        self.zmat.safe_iloc[3:,6] = parameters[self.end_angles:]

        # update the Cartesian coordinates
        self.xyz = self.zmat.get_cartesian()
        self.atoms.positions = self.xyz[['x','y','z']].to_numpy()

        # return the energy
        energy = 999999.0
        try:
            energy = self.atoms.get_potential_energy()
        except:
            pass # bad geometry
        return energy
    
    def forces(self, parameters: np.ndarray) -> np.ndarray:
        """Get the forces on the molecule at the given parameters.
        
        Parameters are stacked as follows:
        bond lengths, angles, dihedrals

        Args:
            parameters (np.ndarray): Parameters for the molecule.

        Returns:
            np.ndarray: Forces on the molecule using the current calculator
        """

        # set the bond lengths
        self.zmat.iloc[1:,2] = parameters[:self.end_bonds]
        # angles
        self.zmat.iloc[2:,4] = parameters[self.end_bonds:self.end_angles]
        # dihedrals
        self.zmat.iloc[3:,6] = parameters[self.end_angles:]

        # update the Cartesian coordinates
        self.xyz = self.zmat.get_cartesian()
        self.atoms.positions = self.xyz[['x','y','z']].to_numpy()
    
        # return the forces
        forces = self.atoms.get_forces()
        # TODO: convert to Cartesian
        return forces
    
    def write_xyz(self, filename: str):
        """Write the current Cartesian coordinates to an XYZ file.

        Args:
            filename (str): Name / path of the file to write.
        """

        self.xyz.to_xyz(filename)