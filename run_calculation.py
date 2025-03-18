#!/usr/bin/env python3
"""
Main script for running quantum chemistry calculations from input files.
"""

import argparse
import sys

import numpy as np

from quantumchem import HartreeFock
from quantumchem.input_parser import create_molecule_from_input, parse_input_file


def format_matrix(matrix: np.ndarray, precision: int = 6) -> str:
	"""Format a matrix for pretty printing."""
	if matrix.ndim == 1:
		matrix = matrix.reshape(-1, 1)

	n_rows, n_cols = matrix.shape
	col_width = precision + 4

	# Create format string for each number
	number_format = f"{{:>{col_width}.{precision}f}}"

	# Format each row
	rows = []
	for i in range(n_rows):
		row = " ".join(number_format.format(x) for x in matrix[i])
		rows.append(row)

	return "\n".join(rows)


def print_results(
	input_file: str, energy: float, orbital_energies: np.ndarray, orbital_coefficients: np.ndarray
) -> None:
	"""Print calculation results in a nice format."""
	print("\n" + "=" * 80)
	print(f"Results for input file: {input_file}")
	print("=" * 80)

	print(f"\nFinal HF Energy: {energy:.10f} Hartree")
	print(f"                 {energy * 27.211386:.10f} eV")
	print(f"                 {energy * 627.509474:.10f} kcal/mol")

	print("\nOrbital Energies (Hartree):")
	print("-" * 40)
	for i, e in enumerate(orbital_energies):
		print(f"Orbital {i + 1:2d}: {e:12.6f}")

	print("\nMolecular Orbital Coefficients:")
	print("-" * 80)
	print(format_matrix(orbital_coefficients))
	print("=" * 80 + "\n")


def main():
	parser = argparse.ArgumentParser(description="Run quantum chemistry calculations.")
	parser.add_argument("input_file", help="Path to the input file")
	args = parser.parse_args()

	try:
		# Parse input file
		input_data = parse_input_file(args.input_file)

		# Create molecule
		molecule = create_molecule_from_input(input_data)

		# Run calculation
		hf = HartreeFock(
			molecule,
			basis=input_data.basis_set,
			max_iterations=input_data.max_iterations,
			convergence_threshold=input_data.convergence_threshold,
		)

		energy, orbital_energies, orbital_coefficients = hf.compute()

		# Print results
		print_results(args.input_file, energy, orbital_energies, orbital_coefficients)

	except Exception as e:
		print(f"Error: {str(e)}", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main()
