"""Parser for quantum chemistry input files."""

from dataclasses import dataclass
from typing import List

import numpy as np

from .basis import BasisSet
from .molecule import Molecule


@dataclass
class InputData:
	"""Container for parsed input data."""

	title: str
	basis_set: str
	method: str
	charge: int
	multiplicity: int
	atomic_symbols: List[str]
	coordinates: np.ndarray
	max_iterations: int
	convergence_threshold: float


def parse_input_file(filename: str) -> InputData:
	"""
	Parse a quantum chemistry input file.

	Format:
	```
	title = Water molecule HF calculation
	basis = STO-3G
	method = HF
	charge = 0
	multiplicity = 1
	max_iterations = 100
	convergence = 1e-8

	geometry
	O    0.0000    0.0000    0.0000
	H    0.7574    0.5860    0.0000
	H   -0.7574    0.5860    0.0000
	end
	```
	"""
	# Default values
	title = "Quantum Chemistry Calculation"
	basis_set = "STO-3G"
	method = "HF"
	charge = 0
	multiplicity = 1
	max_iterations = 100
	convergence = 1e-8

	atomic_symbols = []
	coordinates = []

	with open(filename, "r") as f:
		lines = [line.strip() for line in f.readlines()]

	reading_geometry = False

	for line in lines:
		if not line or line.startswith("#"):
			continue

		if reading_geometry:
			if line.lower() == "end":
				reading_geometry = False
				continue
			# Parse geometry line
			try:
				symbol, x, y, z = line.split()
				atomic_symbols.append(symbol)
				coordinates.append([float(x), float(y), float(z)])
			except ValueError:
				raise ValueError(f"Invalid geometry line: {line}")
		else:
			if line.lower() == "geometry":
				reading_geometry = True
				continue

			if "=" in line:
				key, value = [x.strip() for x in line.split("=")]
				key = key.lower()

				if key == "title":
					title = value
				elif key == "basis":
					if value not in BasisSet.available_basis_sets():
						raise ValueError(
							f"Unsupported basis set: {value}. Available: {BasisSet.available_basis_sets()}"
						)
					basis_set = value
				elif key == "method":
					if value.upper() != "HF":
						raise ValueError("Only HF method is currently supported")
					method = value.upper()
				elif key == "charge":
					charge = int(value)
				elif key == "multiplicity":
					multiplicity = int(value)
				elif key == "max_iterations":
					max_iterations = int(value)
				elif key == "convergence":
					convergence = float(value)

	if not atomic_symbols:
		raise ValueError("No geometry found in input file")

	return InputData(
		title=title,
		basis_set=basis_set,
		method=method,
		charge=charge,
		multiplicity=multiplicity,
		atomic_symbols=atomic_symbols,
		coordinates=np.array(coordinates),
		max_iterations=max_iterations,
		convergence_threshold=convergence,
	)


def symbol_to_atomic_number(symbol: str) -> int:
	"""Convert atomic symbol to atomic number."""
	symbol_to_number = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10}

	symbol = symbol.title()  # Convert to title case (e.g., 'h' -> 'H')
	if symbol not in symbol_to_number:
		raise ValueError(f"Unsupported atomic symbol: {symbol}")

	return symbol_to_number[symbol]


def create_molecule_from_input(input_data: InputData) -> Molecule:
	"""Create a Molecule instance from input data."""
	atomic_numbers = [symbol_to_atomic_number(symbol) for symbol in input_data.atomic_symbols]
	return Molecule(atomic_numbers, input_data.coordinates)
