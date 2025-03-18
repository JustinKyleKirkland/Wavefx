# ruff: noqa: E741  # Allow ambiguous variable names (l, I, O) due to physics notation
import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.special import comb, factorial2

from .basis_sets.data import BASIS_SETS

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
logging.basicConfig(
	filename=os.path.join(log_dir, "quantum_chem.log"),
	level=logging.DEBUG,
	format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class PrimitiveGaussian:
	"""Represents a primitive Gaussian function."""

	exponent: float
	coefficient: float


@dataclass
class ContractedGaussian:
	"""Represents a contracted Gaussian function."""

	primitives: List[PrimitiveGaussian]
	angular_momentum: Tuple[int, int, int]  # (l, m, n)
	center: np.ndarray

	def evaluate(self, points: np.ndarray) -> np.ndarray:
		"""
		Evaluate the contracted Gaussian at given points.

		Args:
		    points: Array of shape (n_points, 3) containing the coordinates

		Returns:
		    Array of shape (n_points,) containing the values
		"""
		result = np.zeros(len(points))
		for prim in self.primitives:
			dx = points[:, 0] - self.center[0]
			dy = points[:, 1] - self.center[1]
			dz = points[:, 2] - self.center[2]
			r2 = dx * dx + dy * dy + dz * dz

			# Angular part
			l, m, n = self.angular_momentum
			angular = (dx**l) * (dy**m) * (dz**n)

			# Radial part
			radial = np.exp(-prim.exponent * r2)

			result += prim.coefficient * angular * radial
		return result


def gaussian_product_center(alpha1: float, center1: np.ndarray, alpha2: float, center2: np.ndarray) -> np.ndarray:
	"""Compute the Gaussian product center of two primitive Gaussians."""
	return (alpha1 * center1 + alpha2 * center2) / (alpha1 + alpha2)


def overlap_1d(l1: int, l2: int, a: float, b: float, gamma: float) -> float:
	"""
	Compute 1D overlap integral between two Gaussian basis functions.

	Args:
		l1, l2: Angular momentum quantum numbers
		a, b: Distances from Gaussian product center to centers 1 and 2
		gamma: Sum of Gaussian exponents
	"""
	sum = 0.0
	for i in range(l1 + 1):
		for j in range(l2 + 1):
			if (i + j) % 2 == 1:  # Odd power terms are zero by symmetry
				continue

			try:
				# Use log arithmetic for better numerical stability
				log_term = (np.log(factorial2(i + j - 1)) if i + j > 0 else 0.0) - ((i + j) / 2.0) * np.log(2.0 * gamma)

				# Add binomial terms
				if i <= l1 and j <= l2:  # Ensure valid binomial coefficients
					coef = np.exp(log_term)
					if abs(coef) < 1e-15:  # Skip negligible terms
						continue

					term = coef * (a ** (l1 - i)) * (b ** (l2 - j))
					# Calculate binomial coefficients using scipy.special.comb
					term *= comb(l1, i, exact=True) * comb(l2, j, exact=True)

					if np.isfinite(term):
						sum += term

			except (OverflowError, FloatingPointError):
				continue

	return sum


class BasisSet:
	"""Class to handle the complete basis set for a molecule."""

	def __init__(self, name: str = "STO-3G"):
		"""
		Initialize a basis set.

		Args:
		    name: Name of the basis set (e.g., "STO-3G", "3-21G", "6-31G")
		"""
		self.name = name
		if name not in BASIS_SETS:
			raise ValueError(f"Basis set '{name}' not found. Available basis sets: {list(BASIS_SETS.keys())}")
		self.basis_data = BASIS_SETS[name]

	def _create_p_orbitals(
		self, exponents: List[float], coefficients: List[float], center: np.ndarray
	) -> List[ContractedGaussian]:
		"""Create px, py, and pz orbitals from the same exponents and coefficients."""
		p_orbitals = []
		angular_momenta = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # px, py, pz

		# Create primitives with proper normalization
		primitives = []
		for exp, coef in zip(exponents, coefficients):
			# The coefficient needs to be the same for px, py, and pz
			primitives.append(PrimitiveGaussian(exp, coef))

		# Create contracted Gaussians for each p orbital
		for l, m, n in angular_momenta:
			p_orbitals.append(ContractedGaussian(primitives.copy(), (l, m, n), center.copy()))

		return p_orbitals

	def get_basis_functions(self, atomic_number: int, center: np.ndarray) -> List[ContractedGaussian]:
		"""
		Get the basis functions for an atom.

		Args:
		    atomic_number: Atomic number of the atom
		    center: Coordinates of the atom center

		Returns:
		    List of ContractedGaussian objects representing the basis functions
		"""
		if atomic_number not in self.basis_data:
			raise ValueError(f"Atom with Z={atomic_number} not supported in basis {self.name}")

		functions = []
		atom_data = self.basis_data[atomic_number]

		logging.debug(f"\nCreating basis functions for atom Z={atomic_number} at center {center}")

		for orbital_name, orbital_data in atom_data.items():
			exponents = orbital_data["exponents"]
			coefficients = orbital_data["coefficients"]
			angular_momentum = orbital_data["angular_momentum"]

			logging.debug(f"\nOrbital: {orbital_name}")
			logging.debug(f"Exponents: {exponents}")
			logging.debug(f"Coefficients: {coefficients}")
			logging.debug(f"Angular momentum: {angular_momentum}")

			# For p orbitals, create px, py, and pz
			if orbital_name.endswith("p"):
				p_functions = self._create_p_orbitals(exponents, coefficients, center)
				logging.debug(f"Created {len(p_functions)} p-orbitals")
				functions.extend(p_functions)
			else:
				primitives = [PrimitiveGaussian(exp, coef) for exp, coef in zip(exponents, coefficients)]
				functions.append(ContractedGaussian(primitives, angular_momentum, center.copy()))
				logging.debug(f"Created s-orbital with {len(primitives)} primitives")

		logging.debug(f"Total basis functions created: {len(functions)}")
		return functions

	def compute_overlap(self, basis1: ContractedGaussian, basis2: ContractedGaussian) -> float:
		"""
		Compute overlap integral between two contracted Gaussian basis functions.
		Uses the Gaussian product theorem and analytical formulas with enhanced numerical stability.
		"""
		# Early exit for very distant centers
		center_dist = np.linalg.norm(basis1.center - basis2.center)
		if center_dist > 10.0:  # Arbitrary cutoff for negligible overlap
			logging.debug("Centers too far apart, returning 0.0")
			return 0.0

		# Store terms for stable summation
		terms = []

		# Debug info
		logging.debug("\nComputing overlap between:")
		logging.debug(f"Basis1: center={basis1.center}, angular_momentum={basis1.angular_momentum}")
		logging.debug(f"Basis2: center={basis2.center}, angular_momentum={basis2.angular_momentum}")

		for prim1 in basis1.primitives:
			for prim2 in basis2.primitives:
				try:
					# Get centers and exponents
					a1, a2 = prim1.exponent, prim2.exponent
					c1, c2 = basis1.center, basis2.center

					logging.debug("\nPrimitive pair:")
					logging.debug(f"  exp1={a1:.6f}, coef1={prim1.coefficient:.6f}")
					logging.debug(f"  exp2={a2:.6f}, coef2={prim2.coefficient:.6f}")

					# Skip if exponents are too different (poor overlap)
					if max(a1, a2) / min(a1, a2) > 1e4:
						logging.debug("  Skipping: exponents too different")
						continue

					# Gaussian product center and exponent
					gamma = a1 + a2
					P = gaussian_product_center(a1, c1, a2, c2)

					# Center-center term
					r2 = np.sum((c1 - c2) ** 2)

					# Skip if centers are too far apart relative to exponents
					if r2 * min(a1, a2) > 30.0:
						logging.debug("  Skipping: centers too far apart")
						continue

					# Compute everything in log space
					try:
						# Coefficient term
						log_coef = np.log(abs(prim1.coefficient)) + np.log(abs(prim2.coefficient))

						# Exponential term
						log_exp = -a1 * a2 * r2 / gamma

						# Normalization terms
						l1, m1, n1 = basis1.angular_momentum
						l2, m2, n2 = basis2.angular_momentum

						logging.debug("  Computing normalization:")
						logging.debug(f"    Angular momentum 1: ({l1}, {m1}, {n1})")
						logging.debug(f"    Angular momentum 2: ({l2}, {m2}, {n2})")

						norm1 = self._gaussian_normalization(a1, l1, m1, n1)
						norm2 = self._gaussian_normalization(a2, l2, m2, n2)
						logging.debug(f"    Norm1: {norm1:.6e}")
						logging.debug(f"    Norm2: {norm2:.6e}")

						log_norm = np.log(norm1) + np.log(norm2)

						# Gaussian overlap term
						log_gaussian = 1.5 * np.log(np.pi / gamma)

						# Angular momentum terms
						overlap_product = 1.0
						for i, (l1, l2) in enumerate(zip(basis1.angular_momentum, basis2.angular_momentum)):
							overlap_i = overlap_1d(l1, l2, P[i] - c1[i], P[i] - c2[i], gamma)
							logging.debug(f"    Overlap_{i}: {overlap_i:.6e}")
							if abs(overlap_i) < 1e-15:
								overlap_product = 0.0
								break
							overlap_product *= overlap_i

						if overlap_product == 0.0:
							logging.debug("  Skipping: zero angular overlap")
							continue

						log_angular = np.log(abs(overlap_product))

						# Combine all logarithmic terms
						log_term = log_coef + log_exp + log_norm + log_gaussian + log_angular
						logging.debug(f"  Log term: {log_term:.6f}")

						# Store the term for later summation
						terms.append(log_term)

					except (ValueError, OverflowError, FloatingPointError) as e:
						logging.debug(f"  Error in computation: {str(e)}")
						continue

				except (OverflowError, FloatingPointError) as e:
					logging.debug(f"  Error in primitive calculation: {str(e)}")
					continue

		if not terms:
			logging.debug("No valid terms found")
			return 0.0

		# Perform stable summation of logarithmic terms
		max_term = max(terms)
		sum_terms = 0.0

		for term in terms:
			# Subtract max_term to avoid overflow
			sum_terms += np.exp(term - max_term)

		# Final result
		result = np.exp(max_term) * sum_terms
		logging.debug(f"Final overlap: {result:.6e}")

		return result if np.isfinite(result) else 0.0

	def _gaussian_normalization(self, alpha: float, l: int, m: int, n: int) -> float:
		"""
		Calculate the normalization constant for a Gaussian primitive.
		Uses logarithmic arithmetic to avoid overflow/underflow.
		"""

		def safe_factorial2(n: int) -> float:
			if n <= 0:
				return 1.0
			log_result = 0.0
			for i in range(n, 0, -2):
				log_result += np.log(i)
			return np.exp(log_result)

		# Calculate in log space to avoid overflow
		log_numerator = np.log(2) * (2.0 * (l + m + n) + 3.0) / 4.0 + (3.0 / 4.0) * np.log(alpha)

		# Calculate denominator terms
		log_denominator = 0.5 * (
			np.log(np.pi)
			+ np.log(safe_factorial2(2 * l - 1))
			+ np.log(safe_factorial2(2 * m - 1))
			+ np.log(safe_factorial2(2 * n - 1))
		)

		# Combine and exponentiate
		try:
			return np.exp(log_numerator - log_denominator)
		except OverflowError:
			raise ValueError(f"Normalization constant overflow for alpha={alpha}, l={l}, m={m}, n={n}")

	@classmethod
	def available_basis_sets(cls) -> List[str]:
		"""Return a list of available basis set names."""
		return list(BASIS_SETS.keys())
