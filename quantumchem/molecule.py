from typing import List

import numpy as np


class Molecule:
	"""
	A class to represent a molecular system with atomic coordinates and nuclear charges.
	"""

	def __init__(self, atomic_numbers: List[int], coordinates: np.ndarray):
		"""
		Initialize a molecule with atomic numbers and coordinates.

		Args:
		    atomic_numbers: List of atomic numbers for each atom
		    coordinates: Numpy array of shape (n_atoms, 3) with XYZ coordinates in Angstroms
		"""
		self.atomic_numbers = np.array(atomic_numbers)
		self.coordinates = np.array(coordinates)
		self.n_atoms = len(atomic_numbers)
		self.n_electrons = sum(atomic_numbers)

	@classmethod
	def from_xyz(cls, filename: str) -> "Molecule":
		"""
		Create a Molecule instance from an XYZ file.

		Args:
		    filename: Path to the XYZ file

		Returns:
		    Molecule instance
		"""
		atomic_numbers = []
		coordinates = []

		with open(filename, "r") as f:
			n_atoms = int(f.readline())
			f.readline()  # Skip comment line

			for _ in range(n_atoms):
				line = f.readline().strip().split()
				symbol = line[0]
				coords = [float(x) for x in line[1:4]]
				atomic_numbers.append(cls._symbol_to_atomic_number(symbol))
				coordinates.append(coords)

		return cls(atomic_numbers, np.array(coordinates))

	@staticmethod
	def _symbol_to_atomic_number(symbol: str) -> int:
		"""Convert atomic symbol to atomic number."""
		symbol_to_number = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10}
		return symbol_to_number[symbol]

	def nuclear_repulsion_energy(self) -> float:
		"""
		Calculate the nuclear repulsion energy.

		Returns:
		    Nuclear repulsion energy in atomic units
		"""
		energy = 0.0
		for i in range(self.n_atoms):
			for j in range(i + 1, self.n_atoms):
				r_ij = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
				energy += (self.atomic_numbers[i] * self.atomic_numbers[j]) / r_ij
		return energy
