# ruff: noqa: E741  # Allow ambiguous variable names (l, I, O) due to physics notation


"""
Geometry optimizer for quantum chemistry calculations.
Implements BFGS optimization algorithm for both Hartree-Fock and DFT methods.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from .dft import KohnShamDFT
from .hartree_fock import HartreeFock


class GeometryOptimizer:
	"""Geometry optimizer using BFGS algorithm."""

	def __init__(
		self,
		calculator: Union[HartreeFock, KohnShamDFT],
		max_iterations: int = 100,
		convergence_threshold: float = 1e-6,
		step_size: float = 0.1,
		use_constraints: bool = False,
		constraints: Optional[List[Tuple[int, int, float]]] = None,
	):
		"""
		Initialize geometry optimizer.

		Args:
		    calculator: Hartree-Fock or DFT calculator instance
		    max_iterations: Maximum number of optimization iterations
		    convergence_threshold: Convergence threshold for forces (Hartree/Bohr)
		    step_size: Initial step size for optimization
		    use_constraints: Whether to use constraints in optimization
		    constraints: List of (atom1, atom2, distance) tuples for bond length constraints
		"""
		self.calculator = calculator
		self.molecule = calculator.molecule
		self.max_iterations = max_iterations
		self.convergence_threshold = convergence_threshold
		self.step_size = step_size
		self.use_constraints = use_constraints
		self.constraints = constraints or []

		# Initialize BFGS parameters
		n_atoms = len(self.molecule.atomic_numbers)
		self.H = np.eye(3 * n_atoms)
		self.energy_history = []
		self.force_history = []

	def compute_forces(self, x: np.ndarray) -> np.ndarray:
		"""
		Compute forces for given atomic coordinates.

		Args:
		    x: Flattened array of atomic coordinates

		Returns:
		    Flattened array of forces
		"""
		# Update molecule coordinates
		self.molecule.coordinates = x.reshape(-1, 3)

		# Compute energy and wavefunction
		if isinstance(self.calculator, HartreeFock):
			energy, _, _ = self.calculator.compute()
		else:  # DFT
			energy, _, _ = self.calculator.compute()

		# Compute forces using Hellmann-Feynman theorem
		forces = np.zeros_like(x)
		n_atoms = len(self.molecule.coordinates)

		# Nuclear-nuclear repulsion forces
		for i in range(n_atoms):
			for j in range(i + 1, n_atoms):
				R = self.molecule.coordinates[i] - self.molecule.coordinates[j]
				R_norm = np.linalg.norm(R)
				force = self.molecule.atomic_numbers[i] * self.molecule.atomic_numbers[j] * R / (R_norm**3)
				forces[3 * i : 3 * (i + 1)] += force
				forces[3 * j : 3 * (j + 1)] -= force

		# Electronic forces (Hellmann-Feynman)
		if isinstance(self.calculator, HartreeFock):
			# Get density matrix and integrals
			_, _, C = self.calculator.compute()
			D = 2 * (C[:, : self.calculator.n_occupied] @ C[:, : self.calculator.n_occupied].T)
			h_core = self.calculator.compute_kinetic_matrix() + self.calculator.compute_nuclear_attraction_matrix()
			eri = self.calculator.compute_electron_repulsion_integrals()
			F = self.calculator.compute_fock_matrix(D, h_core, eri)

			# Compute electronic forces
			for i in range(n_atoms):
				# Derivative of nuclear attraction integrals
				dV = self.calculator.compute_nuclear_attraction_matrix_derivative(i)
				# Derivative of overlap matrix
				dS = self.calculator.compute_overlap_matrix_derivative(i)
				# Derivative of kinetic energy matrix
				dT = self.calculator.compute_kinetic_matrix_derivative(i)
				# Derivative of electron repulsion integrals
				dERI = self.calculator.compute_electron_repulsion_integrals_derivative(i)

				# Compute force contribution
				force = np.zeros(3)
				for j in range(self.calculator.n_basis):
					for k in range(self.calculator.n_basis):
						if abs(D[j, k]) > 1e-10:
							# Core Hamiltonian contribution
							force += D[j, k] * (dT[j, k] + dV[j, k])
							# Two-electron contribution
							for l in range(self.calculator.n_basis):
								for m in range(self.calculator.n_basis):
									force += 0.5 * D[j, k] * D[l, m] * dERI[j, k, l, m]
							# Overlap contribution
							force -= 0.5 * D[j, k] * dS[j, k]

				forces[3 * i : 3 * (i + 1)] += force

		else:  # DFT
			# Get density matrix and integrals
			_, _, C = self.calculator.compute()
			D = 2 * (C[:, : self.calculator.n_occupied] @ C[:, : self.calculator.n_occupied].T)
			h_core = (
				self.calculator.hf.compute_kinetic_matrix() + self.calculator.hf.compute_nuclear_attraction_matrix()
			)
			eri = self.calculator.hf.compute_electron_repulsion_integrals()
			F = h_core + self.calculator.compute_vxc_matrix(D)

			# Compute electronic forces
			for i in range(n_atoms):
				# Derivative of nuclear attraction integrals
				dV = self.calculator.hf.compute_nuclear_attraction_matrix_derivative(i)
				# Derivative of overlap matrix
				dS = self.calculator.hf.compute_overlap_matrix_derivative(i)
				# Derivative of kinetic energy matrix
				dT = self.calculator.hf.compute_kinetic_matrix_derivative(i)
				# Derivative of electron repulsion integrals
				dERI = self.calculator.hf.compute_electron_repulsion_integrals_derivative(i)
				# Derivative of XC potential
				dVxc = self.calculator.compute_vxc_matrix_derivative(D, i)

				# Compute force contribution
				force = np.zeros(3)
				for j in range(self.calculator.n_basis):
					for k in range(self.calculator.n_basis):
						if abs(D[j, k]) > 1e-10:
							# Core Hamiltonian contribution
							force += D[j, k] * (dT[j, k] + dV[j, k])
							# Two-electron contribution
							for l in range(self.calculator.n_basis):
								for m in range(self.calculator.n_basis):
									force += 0.5 * D[j, k] * D[l, m] * dERI[j, k, l, m]
							# XC contribution
							force += D[j, k] * dVxc[j, k]
							# Overlap contribution
							force -= 0.5 * D[j, k] * dS[j, k]

				forces[3 * i : 3 * (i + 1)] += force

		# Apply constraints if specified
		if self.use_constraints:
			forces = self._apply_constraints(forces)

		return -forces.flatten()

	def objective_function(self, x: np.ndarray) -> float:
		"""
		Objective function for optimization (total energy).

		Args:
		    x: Flattened array of atomic coordinates

		Returns:
		    Total energy in Hartree
		"""
		# Update molecule coordinates
		self.molecule.coordinates = x.reshape(-1, 3)

		# Compute energy
		if isinstance(self.calculator, HartreeFock):
			energy, _, _ = self.calculator.compute()
		else:  # DFT
			energy, _, _ = self.calculator.compute()

		# Store energy and forces
		self.energy_history.append(energy)
		self.force_history.append(self.compute_forces(x))

		return energy

	def _apply_constraints(self, forces: np.ndarray) -> np.ndarray:
		"""
		Apply bond length constraints to forces.

		Args:
		    forces: Forces on nuclei

		Returns:
		    Constrained forces
		"""
		n_atoms = len(self.molecule.coordinates)
		constrained_forces = forces.copy()

		for atom1, atom2, target_distance in self.constraints:
			# Compute current bond vector and length
			R = self.molecule.coordinates[atom1] - self.molecule.coordinates[atom2]
			current_distance = np.linalg.norm(R)

			# Compute force correction
			force_correction = R / current_distance
			force_magnitude = (
				np.linalg.norm(forces[atom1] - forces[atom2]) * (current_distance - target_distance) / current_distance
			)

			# Apply force correction
			constrained_forces[3 * atom1 : 3 * (atom1 + 1)] -= force_magnitude * force_correction
			constrained_forces[3 * atom2 : 3 * (atom2 + 1)] += force_magnitude * force_correction

		return constrained_forces

	def optimize(self) -> Tuple[np.ndarray, float, List[float], List[np.ndarray]]:
		"""
		Optimize molecular geometry using BFGS algorithm.

		Returns:
		    Tuple of (optimized coordinates, final energy, energy history, force history)
		"""
		# Initial coordinates
		x0 = self.molecule.coordinates.flatten()

		# Define optimization options
		options = {
			"maxiter": self.max_iterations,
			"disp": True,
		}

		# Run optimization
		result = minimize(
			self.objective_function,
			x0,
			method="BFGS",
			jac=self.compute_forces,
			options=options,
		)

		# Update molecule with optimized coordinates
		self.molecule.coordinates = result.x.reshape(-1, 3)

		return (
			self.molecule.coordinates,
			result.fun,
			self.energy_history,
			self.force_history,
		)

	def print_optimization_summary(self):
		"""Print summary of the optimization process."""
		print("\nOptimization Summary:")
		print("-" * 50)
		print(f"Final Energy: {self.energy_history[-1]:.6f} Hartree")
		print(f"Number of Iterations: {len(self.energy_history)}")
		print("Final Forces:")
		for i, force in enumerate(self.force_history[-1]):
			print(f"  Atom {i}: {force:.6f}")
		print("\nOptimized Geometry:")
		for i, (atom, coord) in enumerate(zip(self.molecule.atomic_numbers, self.molecule.coordinates)):
			print(f"  {atom:2d} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}")
		print("-" * 50)
