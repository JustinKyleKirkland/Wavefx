"""
Example DFT calculation for a water molecule using different functionals.
"""

import numpy as np

from quantumchem.dft import KohnShamDFT
from quantumchem.molecule import Molecule


def run_dft_calculation(molecule, basis_set: str, functional: str) -> tuple:
	"""Run DFT calculation with specified functional."""
	print(f"\n{functional} Calculation")
	print("-" * 50)

	try:
		dft = KohnShamDFT(molecule, basis_set, xc=functional)
		energy, orb_energies, orb_coeff = dft.compute()

		print(f"{functional} Total Energy: {energy:.8f} Hartree")
		print("\nOrbital Energies (Hartree):")
		for i, e in enumerate(orb_energies):
			print(f"ε_{i} = {e:.6f}")

		return energy, orb_energies, orb_coeff
	except Exception as e:
		print(f"{functional} calculation failed: {str(e)}")
		return None, None, None


def main():
	# Define water molecule geometry (in Bohr)
	atomic_numbers = [8, 1, 1]  # O, H, H
	coordinates = np.array(
		[
			[0.0, 0.0, 0.0],  # O
			[0.0, -1.43, 1.108],  # H
			[0.0, 1.43, 1.108],  # H
		]
	)

	# Create molecule object
	water = Molecule(atomic_numbers, coordinates)

	# Perform DFT calculations with different functionals
	basis_set = "STO-3G"  # Start with minimal basis

	print("Performing DFT calculations for water molecule...")
	print(f"Basis set: {basis_set}")
	print("\nGeometry:")
	print("Atom    X        Y        Z")
	print("-" * 25)
	for Z, (x, y, z) in zip(atomic_numbers, coordinates):
		symbol = "H" if Z == 1 else "O"
		print(f"{symbol:4s} {x:8.3f} {y:8.3f} {z:8.3f}")

	# Run calculations with all available functionals
	functionals = ["LDA", "PBE", "BLYP", "B3LYP", "TPSS", "M06", "PW91"]
	results = {}

	for functional in functionals:
		energy, orb_energies, _ = run_dft_calculation(water, basis_set, functional)
		if energy is not None:
			results[functional] = energy

	# Compare results
	if len(results) > 1:
		print("\nComparison of Total Energies:")
		print("-" * 50)
		print("Functional      Energy (Hartree)     ΔE (kcal/mol)")
		print("-" * 50)

		# Use LDA as reference if available, otherwise use the first available functional
		ref_functional = "LDA" if "LDA" in results else list(results.keys())[0]
		ref_energy = results[ref_functional]

		for functional, energy in results.items():
			delta_e = (energy - ref_energy) * 627.509
			print(f"{functional:8s}    {energy:15.8f}    {delta_e:10.2f}")


if __name__ == "__main__":
	main()
