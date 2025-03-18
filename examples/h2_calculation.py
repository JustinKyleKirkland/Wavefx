import numpy as np

from quantumchem import HartreeFock
from quantumchem.molecule import Molecule


def main():
	# Create H2 molecule with 0.74 Angstrom bond length
	atomic_numbers = [1, 1]  # Two hydrogen atoms
	coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

	# Create molecule instance
	molecule = Molecule(atomic_numbers, coordinates)

	# Initialize Hartree-Fock calculation
	hf = HartreeFock(molecule, basis="STO-3G")

	# Run the calculation
	energy, orbital_energies, orbital_coefficients = hf.compute()

	# Print results
	print(f"Nuclear repulsion energy: {molecule.nuclear_repulsion_energy():.6f} Hartree")
	print(f"Total HF energy: {energy:.6f} Hartree")
	print("\nOrbital energies (Hartree):")
	for i, e in enumerate(orbital_energies):
		print(f"Orbital {i + 1}: {e:.6f}")

	print("\nOrbital coefficients:")
	print(orbital_coefficients)


if __name__ == "__main__":
	main()
