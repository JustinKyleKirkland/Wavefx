# ruff: noqa: E741  # Allow ambiguous variable names (l, I, O) due to physics notation


"""
Implementation of Density Functional Theory (DFT) methods.
This module provides Kohn-Sham DFT functionality with LDA and PBE functionals.
"""

import logging
import multiprocessing as mp
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .hartree_fock import HartreeFock
from .molecule import Molecule

# Global variables for multiprocessing
MAX_WORKERS = max(1, mp.cpu_count() - 1)  # Leave one CPU free
CHUNK_SIZE = 1000  # Number of grid points to process at once


def _compute_grid_chunk(args):
	"""Helper function for parallel grid computations."""
	points, weights, basis_functions, D = args
	try:
		# Evaluate basis functions for this chunk
		basis_vals = np.zeros((len(points), len(basis_functions)))
		for i, basis in enumerate(basis_functions):
			for p in range(len(points)):
				basis_vals[p, i] = basis.evaluate(points[p])

		# Compute density for this chunk
		rho = np.einsum("pi,ij,pj->p", basis_vals, D, basis_vals, optimize=True)

		return rho * weights
	except Exception as e:
		logging.error(f"Error in grid chunk computation: {str(e)}")
		return np.zeros(len(points))


class DFTGrid:
	"""Numerical integration grid for DFT calculations using Lebedev-Laikov quadrature."""

	def __init__(self, molecule: Molecule, radial_points: int = 50, angular_points: int = 302):
		"""
		Initialize integration grid.

		Args:
			molecule: Molecule object containing geometry
			radial_points: Number of radial points per atom
			angular_points: Number of angular points (valid values: 6, 14, 26, 38, 50, 86, 110, 146, 170, 194, 302)
		"""
		self.molecule = molecule
		self.radial_points = radial_points
		self.angular_points = angular_points
		self.points = None
		self.weights = None
		self.setup_grid()

	def setup_grid(self):
		"""Create molecular grid using atom-centered grids."""
		all_points = []
		all_weights = []

		for i, (Z, coord) in enumerate(zip(self.molecule.atomic_numbers, self.molecule.coordinates)):
			# Get Bragg-Slater radius for atom
			radius = self._get_bragg_slater_radius(Z)

			# Generate radial points and weights using Euler-Maclaurin scheme
			r, w_rad = self._generate_radial_grid(radius)

			# Get Lebedev points and weights
			theta, phi, w_ang = self._generate_angular_grid()

			# Convert spherical to Cartesian coordinates
			for ir, (rad, w_r) in enumerate(zip(r, w_rad)):
				for ia, (t, p, w_a) in enumerate(zip(theta, phi, w_ang)):
					x = rad * np.sin(t) * np.cos(p)
					y = rad * np.sin(t) * np.sin(p)
					z = rad * np.cos(t)

					# Transform to molecular coordinates
					point = np.array([x, y, z]) + coord
					weight = w_r * w_a * rad * rad

					# Apply Becke partitioning weights
					becke_weight = self._becke_weight(point, i)

					all_points.append(point)
					all_weights.append(weight * becke_weight)

		self.points = np.array(all_points)
		self.weights = np.array(all_weights)

		# Normalize weights
		self.weights /= np.sum(self.weights)

	def _get_bragg_slater_radius(self, Z: int) -> float:
		"""Get Bragg-Slater radius for atom."""
		# Simplified table of Bragg-Slater radii in Bohr
		radii = {1: 0.35, 6: 0.77, 7: 0.75, 8: 0.73, 9: 0.71, 16: 1.02}
		return radii.get(Z, 1.0)

	def _generate_radial_grid(self, R: float) -> Tuple[np.ndarray, np.ndarray]:
		"""Generate radial grid points and weights using Euler-Maclaurin scheme."""
		# Parameters for radial grid
		N = self.radial_points
		a = 5e-4  # Parameter controlling the smallest radial distance

		# Generate points and weights
		i = np.arange(1, N + 1)
		x = i / (N + 1)
		t = -np.log(a * (1 - x * x * x))  # Transformation for better sampling
		r = R * t / (1 - t)

		# Compute weights
		dt_dx = 3 * x * x / (1 - x * x * x)
		dr_dt = R / (1 - t) ** 2
		w = dt_dx * dr_dt / (N + 1)

		return r, w

	def _generate_angular_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Generate Lebedev angular grid points and weights."""
		# Simplified Lebedev grid for demonstration
		# In practice, would use full Lebedev-Laikov quadrature
		phi = np.linspace(0, 2 * np.pi, self.angular_points)
		costheta = np.linspace(-1, 1, self.angular_points)
		weights = np.ones(self.angular_points) * 4 * np.pi / self.angular_points

		theta = np.arccos(costheta)
		return theta, phi, weights

	def _becke_weight(self, point: np.ndarray, center_idx: int) -> float:
		"""Compute Becke partitioning weight for a grid point."""
		w = 1.0
		for j, coord in enumerate(self.molecule.coordinates):
			if j != center_idx:
				# Compute confocal elliptical coordinates
				mu = (
					np.linalg.norm(point - self.molecule.coordinates[center_idx]) - np.linalg.norm(point - coord)
				) / np.linalg.norm(coord - self.molecule.coordinates[center_idx])

				# Becke's smoothing function
				for _ in range(3):  # Three iterations of smoothing
					mu = 1.5 * mu - 0.5 * mu**3

				w *= 0.5 * (1 - mu)

		return w


class DFTFunctional:
	"""Base class for exchange-correlation functionals."""

	def __init__(self, name: str):
		self.name = name

	def compute_exc(self, rho: np.ndarray, grad_rho: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute exchange-correlation energy density and potential."""
		raise NotImplementedError


class LDAFunctional(DFTFunctional):
	"""Local Density Approximation functional."""

	def __init__(self):
		super().__init__("LDA")

	def compute_exc(self, rho: np.ndarray, grad_rho: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute LDA exchange-correlation energy density and potential."""
		# Ensure positive density
		rho = np.maximum(rho, 1e-10)

		# Exchange part (Slater)
		Cx = -0.7385587663820224  # -(3/4)(3/π)^(1/3)
		ex = Cx * np.power(rho, 1 / 3)
		vx = 4 / 3 * ex

		# Simple correlation (VWN parameters)
		A, x0 = 0.0621814, -0.10498
		b, c = 3.72744, 12.9352

		x = np.sqrt(rho)
		X = x * x + b * x + c
		Q = np.sqrt(4 * c - b * b)

		ec = A * (np.log(x * x / X) + 2 * b / Q * np.arctan(Q / (2 * x + b)))
		vc = ec - A / 6 * (c * (x - x0) - b * x * x) / (X * (x - x0))

		return ex + ec, vx + vc


class PBEFunctional(DFTFunctional):
	"""Perdew-Burke-Ernzerhof (PBE) functional."""

	def __init__(self):
		super().__init__("PBE")

	def compute_exc(self, rho: np.ndarray, grad_rho: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute PBE exchange-correlation energy density and potential."""
		if grad_rho is None:
			raise ValueError("PBE functional requires density gradients")

		# Ensure positive density
		rho = np.maximum(rho, 1e-10)
		grad_rho_mag = np.linalg.norm(grad_rho, axis=1)

		# PBE parameters
		kappa = 0.804  # Exchange parameter
		mu = 0.21951  # Exchange parameter
		beta = 0.066725  # Correlation parameter
		gamma = 0.031091  # Correlation parameter

		# Reduced gradient s = |∇ρ|/(2(3π²)^(1/3)ρ^(4/3))
		kf = (3 * np.pi * np.pi * rho) ** (1 / 3)
		s = grad_rho_mag / (2 * kf * rho)

		# Exchange enhancement factor
		Fx = 1 + kappa - kappa / (1 + mu * s * s / kappa)

		# Exchange energy density and potential
		ex_unif = -0.7385587663820224 * np.power(rho, 1 / 3)  # LDA exchange
		ex = ex_unif * Fx
		vx = ex * (4 / 3 + (mu * s * s) / (kappa + mu * s * s))

		# Correlation
		rs = (3 / (4 * np.pi * rho)) ** (1 / 3)  # Wigner-Seitz radius
		t = grad_rho_mag / (2 * gamma * kf * rho)  # Reduced gradient for correlation

		# PBE correlation parameters
		A = beta / gamma * (1 - np.log(2)) / (2 * np.pi * np.pi)
		alpha = beta / gamma

		# Correlation energy
		ec_unif = -gamma * (1 + alpha * rs) * np.log(1 + beta * rs / gamma)  # Uniform electron gas correlation
		H = gamma * np.log(1 + beta / gamma * t * t * (1 + A * t * t) / (1 + A * t * t + A * A * t * t * t * t))
		ec = ec_unif + H  # Total correlation energy including gradient correction
		vc = ec * (1 - rs / (3 * (1 + alpha * rs)))  # Approximate correlation potential

		return ex + ec, vx + vc


class BLYPFunctional(DFTFunctional):
	"""Becke88 exchange + Lee-Yang-Parr correlation functional."""

	def __init__(self):
		super().__init__("BLYP")

	def compute_exc(self, rho: np.ndarray, grad_rho: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute BLYP exchange-correlation energy density and potential."""
		if grad_rho is None:
			raise ValueError("BLYP functional requires density gradients")

		# Ensure positive density
		rho = np.maximum(rho, 1e-10)
		grad_rho_mag = np.linalg.norm(grad_rho, axis=1)

		# Becke88 exchange parameters
		beta = 0.0042  # Becke88 parameter

		# Reduced gradient x = |∇ρ|/ρ^(4/3)
		x = grad_rho_mag / np.power(rho, 4 / 3)

		# Becke88 exchange enhancement factor
		X = beta * x * x / (1.0 + 6.0 * beta * x * np.arcsinh(x))

		# Exchange energy density
		ex_unif = -0.7385587663820224 * np.power(rho, 1 / 3)  # LDA exchange
		ex = ex_unif * (1.0 + X)

		# Lee-Yang-Parr correlation parameters
		a = 0.04918  # LYP parameters
		b = 0.132
		c = 0.2533
		d = 0.349

		# Compute LYP correlation terms
		omega = np.exp(-c * np.power(rho, -1 / 3)) / (1.0 + d * np.power(rho, -1 / 3))

		# Correlation energy density
		ec = -a * (1.0 + b * grad_rho_mag * grad_rho_mag / (rho * rho * rho * rho * rho * rho * rho * rho)) * omega

		# Potential (simplified form)
		vxc = ex + ec  # This is a simplification; full potential would need more terms

		return ex + ec, vxc


class B3LYPFunctional(DFTFunctional):
	"""B3LYP hybrid functional with exact exchange."""

	def __init__(self):
		super().__init__("B3LYP")
		self.a0 = 0.20  # Exact exchange mixing parameter
		self.ax = 0.72  # Becke88 exchange parameter
		self.ac = 0.81  # LYP correlation parameter
		self.blyp = BLYPFunctional()
		self.lda = LDAFunctional()

	def compute_exc(self, rho: np.ndarray, grad_rho: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute B3LYP exchange-correlation energy density and potential."""
		if grad_rho is None:
			raise ValueError("B3LYP functional requires density gradients")

		# Get BLYP and LDA contributions
		e_blyp, v_blyp = self.blyp.compute_exc(rho, grad_rho)
		e_lda, v_lda = self.lda.compute_exc(rho)

		# Mix according to B3LYP formula
		# Note: This is a simplified version; full implementation would need exact exchange
		exc = (1 - self.a0) * e_lda + self.a0 * e_blyp
		vxc = (1 - self.a0) * v_lda + self.a0 * v_blyp

		return exc, vxc


class TPSSFunctional(DFTFunctional):
	"""Tao-Perdew-Staroverov-Scuseria meta-GGA functional."""

	def __init__(self):
		super().__init__("TPSS")
		self.kappa = 0.804  # Same as PBE
		self.c = 1.59096
		self.e = 1.537
		self.mu = 0.21951
		self.beta = 0.066725

	def compute_exc(
		self, rho: np.ndarray, grad_rho: Optional[np.ndarray] = None, tau: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute TPSS exchange-correlation energy density and potential."""
		if grad_rho is None or tau is None:
			raise ValueError("TPSS functional requires density gradients and kinetic energy density")

		# Ensure positive density
		rho = np.maximum(rho, 1e-10)
		grad_rho_mag = np.linalg.norm(grad_rho, axis=1)

		# Compute reduced quantities
		kf = (3 * np.pi * np.pi * rho) ** (1.0 / 3.0)  # Fermi wavevector
		tau_unif = 0.3 * (3 * np.pi * np.pi) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)  # Uniform electron gas τ
		tau_W = grad_rho_mag * grad_rho_mag / (8 * rho)  # von Weizsäcker kinetic energy density

		# Compute dimensionless quantities
		z = tau_W / tau_unif  # Meta-GGA ingredient
		alpha = (tau - tau_W) / tau_unif  # Non-locality parameter

		# TPSS exchange enhancement factor
		s = grad_rho_mag / (2 * kf * rho)
		Q = (9 / 20) * (alpha - 1) / np.sqrt(1 + 0.5 * (alpha - 1))

		# Exchange enhancement factor
		Fx = 1 + self.kappa - self.kappa / (1 + s * s / self.kappa)
		Fx *= (1 + Q * s * s) / (1 + Q * s * s + self.e * s**4)

		# Exchange energy density
		ex_unif = -0.7385587663820224 * np.power(rho, 1 / 3)
		ex = ex_unif * Fx

		# Correlation energy
		rs = (3 / (4 * np.pi * rho)) ** (1 / 3)  # Wigner-Seitz radius
		ec = -0.1 * np.log(1 + self.beta * rs) * (1 + z * z)  # Simplified correlation

		# Total potential (simplified form)
		vxc = ex + ec

		return ex + ec, vxc


class M06Functional(DFTFunctional):
	"""M06 meta-GGA functional from the Minnesota family."""

	def __init__(self):
		super().__init__("M06")
		# M06 parameters
		self.a = [0.00186726, 0.729974, 0.63611, -3.27287, 6.97169, -6.96797, 2.33812]
		self.d = [0.0010344, 0.317710, -0.373453, 2.23476, -5.51042, 5.78596, -2.48771]
		self.alpha = 0.00186726  # Parameter for kinetic energy density dependence

	def compute_exc(
		self, rho: np.ndarray, grad_rho: Optional[np.ndarray] = None, tau: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute M06 exchange-correlation energy density and potential."""
		if grad_rho is None or tau is None:
			raise ValueError("M06 functional requires density gradients and kinetic energy density")

		# Ensure positive density
		rho = np.maximum(rho, 1e-10)
		grad_rho_mag = np.linalg.norm(grad_rho, axis=1)

		# Compute reduced quantities
		tau_unif = 0.3 * (3 * np.pi * np.pi) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
		tau_W = grad_rho_mag * grad_rho_mag / (8 * rho)

		# Compute working variables
		w = (tau - tau_W) / tau_unif  # Reduced kinetic energy density
		s = grad_rho_mag / (2.0 * (3.0 * np.pi * np.pi * rho) ** (1.0 / 3.0) * rho)

		# M06 exchange enhancement factor using VS98 form
		Fx = np.zeros_like(rho)
		z = s * s

		# Compute polynomial in w and z
		for i, (ai, di) in enumerate(zip(self.a, self.d)):
			gamma = np.exp(-self.alpha * w * w)
			Fx += (ai + di * gamma) * z**i

		# Exchange energy density
		ex_unif = -0.7385587663820224 * np.power(rho, 1 / 3)
		ex = ex_unif * Fx

		# Correlation energy density (simplified)
		rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
		ec = -0.1 * np.log(1 + rs) * (1 + w * w)

		# Total potential (simplified form)
		vxc = ex + ec

		return ex + ec, vxc


class PW91Functional(DFTFunctional):
	"""Perdew-Wang 91 GGA functional."""

	def __init__(self):
		super().__init__("PW91")

	def compute_exc(self, rho: np.ndarray, grad_rho: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute PW91 exchange-correlation energy density and potential."""
		if grad_rho is None:
			raise ValueError("PW91 functional requires density gradients")

		# Ensure positive density
		rho = np.maximum(rho, 1e-10)
		grad_rho_mag = np.linalg.norm(grad_rho, axis=1)

		# PW91 parameters
		alpha = 0.19645
		beta = 7.7956
		gamma = 0.2743

		# Reduced gradient
		s = grad_rho_mag / (2.0 * (3.0 * np.pi * np.pi * rho) ** (1.0 / 3.0) * rho)

		# Exchange enhancement factor
		Fx = (1.0 + s * s * (alpha - beta * np.exp(-gamma * s * s))) / (1.0 + s * s * alpha)

		# Exchange energy density
		ex_unif = -0.7385587663820224 * np.power(rho, 1 / 3)  # LDA exchange
		ex = ex_unif * Fx

		# Correlation (simplified PW91 correlation)
		rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
		ec = -0.1423 / (1.0 + 1.0529 * np.sqrt(rs) + 0.3334 * rs)

		# Potential (simplified)
		vxc = ex + ec

		return ex + ec, vxc


class KohnShamDFT:
	"""Kohn-Sham DFT implementation."""

	def __init__(
		self, molecule: Molecule, basis: str = "STO-3G", xc: str = "LDA", max_iter: int = 100, conv: float = 1e-8
	):
		"""
		Initialize DFT calculator.

		Args:
			molecule: Molecule object
			basis: Basis set name
			xc: Exchange-correlation functional ("LDA", "PBE", "BLYP", "B3LYP", "TPSS", "M06", or "PW91")
			max_iter: Maximum SCF iterations
			conv: Convergence threshold
		"""
		# Initialize system
		self.molecule = molecule
		self.hf = HartreeFock(molecule, basis, max_iter, conv)
		self.basis_functions = self.hf.basis_functions
		self.n_basis = len(self.basis_functions)
		self.n_electrons = molecule.n_electrons

		# Set up grid and functional
		self.grid = DFTGrid(molecule)

		# Initialize functional based on user choice
		xc = xc.upper()
		if xc == "LDA":
			self.xc = LDAFunctional()
		elif xc == "PBE":
			self.xc = PBEFunctional()
		elif xc == "BLYP":
			self.xc = BLYPFunctional()
		elif xc == "B3LYP":
			self.xc = B3LYPFunctional()
		elif xc == "TPSS":
			self.xc = TPSSFunctional()
		elif xc == "M06":
			self.xc = M06Functional()
		elif xc == "PW91":
			self.xc = PW91Functional()
		else:
			raise ValueError(f"Unknown functional: {xc}. Available functionals: LDA, PBE, BLYP, B3LYP, TPSS, M06, PW91")

		# Parameters
		self.max_iter = max_iter
		self.conv = conv

		# Initialize multiprocessing pool
		self.pool = mp.Pool(MAX_WORKERS)
		logging.info(f"Initialized multiprocessing pool with {MAX_WORKERS} workers")

		logging.info(f"\nInitializing Kohn-Sham DFT with {self.xc.name}")
		logging.info(f"Number of basis functions: {self.n_basis}")
		logging.info(f"Number of electrons: {self.n_electrons}")
		logging.info(f"Number of grid points: {len(self.grid.points)}")

	def __del__(self):
		"""Clean up multiprocessing resources."""
		if hasattr(self, "pool"):
			self.pool.close()
			self.pool.join()

	def evaluate_basis_functions(self, points: np.ndarray) -> np.ndarray:
		"""
		Evaluate all basis functions at given points.

		Args:
			points: Array of shape (n_points, 3)

		Returns:
			Array of shape (n_points, n_basis)
		"""
		n_points = len(points)
		values = np.zeros((n_points, self.n_basis))

		try:
			for i, basis in enumerate(self.basis_functions):
				for p in range(n_points):
					try:
						val = basis.evaluate(points[p])
						# Ensure the value is a scalar
						if isinstance(val, np.ndarray):
							if val.size == 1:
								val = float(val.item())
							else:
								val = float(val[0])  # Take first element if multiple values
						elif not isinstance(val, (int, float)):
							val = float(val)  # Convert any other type to float
						values[p, i] = val
					except Exception as e:
						logging.error(f"Error evaluating basis function {i} at point {p}: {str(e)}")
						values[p, i] = 0.0  # Set to zero on error

			logging.debug(f"Basis function values shape: {values.shape}")
			return values

		except Exception as e:
			logging.error(f"Error in evaluate_basis_functions: {str(e)}")
			logging.error(f"Points shape: {points.shape}")
			raise

	def evaluate_basis_gradients(self, points: np.ndarray) -> np.ndarray:
		"""
		Evaluate gradients of all basis functions at given points.

		Args:
			points: Array of shape (n_points, 3)

		Returns:
			Array of shape (n_points, n_basis, 3)
		"""
		n_points = len(points)
		grads = np.zeros((n_points, self.n_basis, 3))

		try:
			for i, basis in enumerate(self.basis_functions):
				for p in range(n_points):
					try:
						grad = basis.evaluate_gradient(points[p])
						# Ensure the gradient is a 1D array of length 3
						if isinstance(grad, np.ndarray):
							if grad.ndim > 1:
								grad = grad.reshape(-1)[:3]  # Take first three elements after flattening
							elif len(grad) > 3:
								grad = grad[:3]  # Take first three elements
							elif len(grad) < 3:
								grad = np.pad(grad, (0, 3 - len(grad)))  # Pad with zeros if too short
						else:
							grad = np.zeros(3)  # Default to zero vector if not array
						grads[p, i] = grad
					except Exception as e:
						logging.error(f"Error evaluating basis gradient {i} at point {p}: {str(e)}")
						grads[p, i] = np.zeros(3)  # Set to zero vector on error

			logging.debug(f"Basis function gradients shape: {grads.shape}")
			return grads

		except Exception as e:
			logging.error(f"Error in evaluate_basis_gradients: {str(e)}")
			logging.error(f"Points shape: {points.shape}")
			raise

	def compute_density(self, D: np.ndarray, points: np.ndarray) -> np.ndarray:
		"""Compute electron density at given points using parallel processing."""
		try:
			# Split points into chunks for parallel processing
			n_points = len(points)
			chunks = [
				(points[i : i + CHUNK_SIZE], self.grid.weights[i : i + CHUNK_SIZE], self.basis_functions, D)
				for i in range(0, n_points, CHUNK_SIZE)
			]

			# Process chunks in parallel
			results = self.pool.map(_compute_grid_chunk, chunks)

			# Combine results
			rho = np.concatenate([r for r in results if r is not None])
			return np.maximum(rho, 1e-10)

		except Exception as e:
			logging.error(f"Error in parallel density computation: {str(e)}")
			raise

	def compute_vxc_matrix(self, D: np.ndarray) -> np.ndarray:
		"""Memory-optimized computation of exchange-correlation potential matrix."""
		try:
			# Use sparse matrices for large systems
			if self.n_basis > 1000:
				Vxc = lil_matrix((self.n_basis, self.n_basis))
			else:
				Vxc = np.zeros((self.n_basis, self.n_basis))

			# Process grid points in chunks to reduce memory usage
			for i in range(0, len(self.grid.points), CHUNK_SIZE):
				chunk_points = self.grid.points[i : i + CHUNK_SIZE]
				chunk_weights = self.grid.weights[i : i + CHUNK_SIZE]

				# Compute density and potential for this chunk
				rho_chunk = self.compute_density(D, chunk_points)

				# Compute gradients if needed
				grad_rho_chunk = None
				tau_chunk = None
				if isinstance(
					self.xc,
					(PBEFunctional, BLYPFunctional, B3LYPFunctional, TPSSFunctional, M06Functional, PW91Functional),
				):
					grad_rho_chunk = self._compute_gradient_chunk(D, chunk_points)

					if isinstance(self.xc, (TPSSFunctional, M06Functional)):
						tau_chunk = self._compute_kinetic_chunk(D, chunk_points)

				# Compute XC potential for this chunk
				if isinstance(self.xc, (TPSSFunctional, M06Functional)):
					_, vxc_chunk = self.xc.compute_exc(rho_chunk, grad_rho_chunk, tau_chunk)
				else:
					_, vxc_chunk = self.xc.compute_exc(rho_chunk, grad_rho_chunk)

				# Evaluate basis functions for this chunk
				basis_vals_chunk = self.evaluate_basis_functions(chunk_points)

				# Update Vxc matrix
				weights_vxc = vxc_chunk * chunk_weights
				chunk_contribution = np.einsum(
					"p,pi,pj->ij", weights_vxc, basis_vals_chunk, basis_vals_chunk, optimize=True
				)

				if isinstance(Vxc, lil_matrix):
					Vxc += csr_matrix(chunk_contribution)
				else:
					Vxc += chunk_contribution

			return Vxc.toarray() if isinstance(Vxc, lil_matrix) else Vxc

		except Exception as e:
			logging.error(f"Error in compute_vxc_matrix: {str(e)}")
			raise

	def _compute_gradient_chunk(self, D: np.ndarray, points: np.ndarray) -> np.ndarray:
		"""Compute density gradient for a chunk of grid points."""
		basis_grads = self.evaluate_basis_gradients(points)
		basis_vals = self.evaluate_basis_functions(points)

		grad_rho = np.zeros((len(points), 3))
		for i in range(self.n_basis):
			for j in range(self.n_basis):
				if abs(D[i, j]) < 1e-10:  # Skip negligible contributions
					continue

				grads_i = basis_grads[:, i, :].reshape(-1, 3)
				vals_j = basis_vals[:, j].reshape(-1, 1)
				grad_rho += 2.0 * D[i, j] * (grads_i * vals_j)

		return grad_rho

	def _compute_kinetic_chunk(self, D: np.ndarray, points: np.ndarray) -> np.ndarray:
		"""Compute kinetic energy density for a chunk of grid points."""
		basis_grads = self.evaluate_basis_gradients(points)
		tau = np.zeros(len(points))

		for i in range(self.n_basis):
			for j in range(self.n_basis):
				if abs(D[i, j]) < 1e-10:  # Skip negligible contributions
					continue

				grad_prod = np.sum(basis_grads[:, i, :] * basis_grads[:, j, :], axis=1)
				tau += 0.5 * D[i, j] * grad_prod

		return tau

	def compute_exc_energy(self, D: np.ndarray) -> float:
		"""Compute exchange-correlation energy."""
		rho = self.compute_density(D, self.grid.points)
		logging.debug(f"Density shape in exc: {rho.shape}")

		grad_rho = None
		tau = None
		if isinstance(
			self.xc, (PBEFunctional, BLYPFunctional, B3LYPFunctional, TPSSFunctional, M06Functional, PW91Functional)
		):
			basis_grads = self.evaluate_basis_gradients(self.grid.points)
			basis_vals = self.evaluate_basis_functions(self.grid.points)

			logging.debug(f"Basis gradients shape in exc: {basis_grads.shape}")
			logging.debug(f"Basis values shape in exc: {basis_vals.shape}")

			grad_rho = np.zeros((len(self.grid.points), 3))

			# Compute gradient using vectorized operations
			for i in range(self.n_basis):
				for j in range(self.n_basis):
					try:
						# Reshape arrays for broadcasting
						grads_i = basis_grads[:, i, :].reshape(-1, 3)  # (n_points, 3)
						vals_j = basis_vals[:, j].reshape(-1, 1)  # (n_points, 1)

						# Compute contribution using broadcasting
						contribution = grads_i * vals_j  # (n_points, 3)
						grad_rho += 2.0 * D[i, j] * contribution

					except Exception as e:
						logging.error(f"Error in gradient computation at i={i}, j={j}: {str(e)}")
						raise

			logging.debug(f"Final gradient shape in exc: {grad_rho.shape}")

			# Compute kinetic energy density for meta-GGA functionals
			if isinstance(self.xc, (TPSSFunctional, M06Functional)):
				tau = self._compute_kinetic_chunk(D, self.grid.points)
				logging.debug(f"Kinetic energy density shape in exc: {tau.shape}")

		# Compute exchange-correlation energy density
		if isinstance(self.xc, (TPSSFunctional, M06Functional)):
			exc, _ = self.xc.compute_exc(rho, grad_rho, tau)
		else:
			exc, _ = self.xc.compute_exc(rho, grad_rho)

		logging.debug(f"Exchange-correlation energy density shape: {exc.shape}")

		result = np.sum(exc * rho * self.grid.weights)
		logging.debug(f"Final exc energy: {result}")

		return result

	def compute_exact_exchange(self, C: np.ndarray, eri: np.ndarray) -> np.ndarray:
		"""
		Compute exact (Hartree-Fock) exchange matrix.

		Args:
			C: Orbital coefficients
			eri: Electron repulsion integrals

		Returns:
			Exchange matrix K
		"""
		nocc = self.n_electrons // 2
		K = np.zeros((self.n_basis, self.n_basis))

		# Compute exact exchange using occupied orbitals
		for i in range(self.n_basis):
			for j in range(self.n_basis):
				for k in range(nocc):
					for l in range(self.n_basis):
						for m in range(self.n_basis):
							K[i, j] -= np.sum(C[l, k] * C[m, k] * eri[i, l, j, m])

		return K

	def compute_kinetic_density(self, D: np.ndarray, points: np.ndarray) -> np.ndarray:
		"""
		Compute kinetic energy density for meta-GGA functionals.

		τ = (1/2)∑ᵢ |∇φᵢ|²

		Args:
			D: Density matrix
			points: Grid points

		Returns:
			Kinetic energy density at each point
		"""
		basis_grads = self.evaluate_basis_gradients(points)
		tau = np.zeros(len(points))

		for i in range(self.n_basis):
			for j in range(self.n_basis):
				# Contract density matrix with basis function gradients
				grad_prod = np.sum(basis_grads[:, i, :] * basis_grads[:, j, :], axis=1)
				tau += 0.5 * D[i, j] * grad_prod

		return tau

	def compute(self) -> Tuple[float, np.ndarray, np.ndarray]:
		"""
		Perform Kohn-Sham DFT calculation.

		Returns:
			Tuple of (total_energy, orbital_energies, orbital_coefficients)
		"""
		# Get core Hamiltonian and 2e integrals
		h_core = self.hf.compute_kinetic_matrix() + self.hf.compute_nuclear_attraction_matrix()
		eri = self.hf.compute_electron_repulsion_integrals()
		S = self.hf.compute_overlap_matrix()

		# Get orthogonalization matrix
		eigval, eigvec = np.linalg.eigh(S)
		X = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T

		# Initial guess
		D = np.zeros((self.n_basis, self.n_basis))
		E_old = 0.0

		# SCF loop
		for iter in range(self.max_iter):
			# Coulomb term
			J = np.zeros_like(h_core)
			for i in range(self.n_basis):
				for j in range(self.n_basis):
					for k in range(self.n_basis):
						for l in range(self.n_basis):
							J[i, j] += D[k, l] * eri[i, j, k, l]

			# Exchange-correlation term
			Vxc = self.compute_vxc_matrix(D)

			# Add exact exchange for hybrid functionals
			K = None
			if isinstance(self.xc, B3LYPFunctional):
				K = self.compute_exact_exchange(eigvec, eri) if iter > 0 else np.zeros_like(h_core)
				F = h_core + J + Vxc + self.xc.a0 * K
			else:
				F = h_core + J + Vxc

			# Solve eigenvalue equation
			Fp = X.T @ F @ X
			eps, Cp = np.linalg.eigh(Fp)
			C = X @ Cp

			# Build new density matrix
			nocc = self.n_electrons // 2
			D_new = 2 * (C[:, :nocc] @ C[:, :nocc].T)

			# Compute energy
			E_scf = np.sum(D_new * (h_core + F)) / 2
			E_xc = self.compute_exc_energy(D_new)

			# Add exact exchange contribution for hybrid functionals
			if K is not None:
				E_x = -0.25 * np.sum(D_new * K)  # Factor of 1/4 due to density matrix definition
				E_tot = E_scf + E_xc + self.xc.a0 * E_x
			else:
				E_tot = E_scf + E_xc

			# Check convergence
			delta_E = abs(E_tot - E_old)
			delta_D = np.max(np.abs(D_new - D))

			logging.info(f"\nIteration {iter + 1}")
			logging.info(f"Energy: {E_tot:.8f} Eh")
			logging.info(f"ΔE: {delta_E:.8e}")
			logging.info(f"ΔD: {delta_D:.8e}")

			if delta_E < self.conv and delta_D < self.conv:
				logging.info("\nSCF converged!")
				break

			D = D_new
			E_old = E_tot
		else:
			logging.warning("\nSCF did not converge!")

		return E_tot, eps, C
