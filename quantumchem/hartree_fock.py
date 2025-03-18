# ruff: noqa: E741  # Allow ambiguous variable names (l, I, O) due to physics notation
from typing import Tuple

import numpy as np
from scipy.special import erf

from .basis import BasisSet, overlap_1d
from .molecule import Molecule


class HartreeFock:
	"""
	Implementation of the Hartree-Fock method.
	"""

	def __init__(
		self, molecule: Molecule, basis: str = "STO-3G", max_iterations: int = 100, convergence_threshold: float = 1e-8
	):
		self.molecule = molecule
		self.basis_set = BasisSet(basis)
		self.max_iterations = max_iterations
		self.convergence_threshold = convergence_threshold

		# Get basis functions for all atoms
		self.basis_functions = []
		for i, Z in enumerate(molecule.atomic_numbers):
			center = molecule.coordinates[i]
			self.basis_functions.extend(self.basis_set.get_basis_functions(Z, center))

		self.n_basis = len(self.basis_functions)
		self.n_electrons = molecule.n_electrons
		self.n_occupied = self.n_electrons // 2  # Assuming closed shell

	def compute_overlap_matrix(self) -> np.ndarray:
		"""Compute the overlap matrix S using analytical integrals."""
		S = np.zeros((self.n_basis, self.n_basis))
		for i, basis_i in enumerate(self.basis_functions):
			for j, basis_j in enumerate(self.basis_functions):
				S[i, j] = self.basis_set.compute_overlap(basis_i, basis_j)
		return S

	def compute_kinetic_matrix(self) -> np.ndarray:
		"""
		Compute the kinetic energy matrix T.
		The kinetic energy integral between two Gaussian basis functions is:
		T_ij = -1/2 ∫ φᵢ(r) ∇² φⱼ(r) dr
		"""
		T = np.zeros((self.n_basis, self.n_basis))

		for i, basis_i in enumerate(self.basis_functions):
			for j, basis_j in enumerate(self.basis_functions):
				# Skip if centers are too far apart (negligible overlap)
				center_dist = np.linalg.norm(basis_i.center - basis_j.center)
				if center_dist > 10.0:
					continue

				# Initialize sum for contracted Gaussians
				t_ij = 0.0

				for prim_i in basis_i.primitives:
					for prim_j in basis_j.primitives:
						alpha_i = prim_i.exponent
						alpha_j = prim_j.exponent

						# Skip if exponents are too different
						if max(alpha_i, alpha_j) / min(alpha_i, alpha_j) > 1e4:
							continue

						# Gaussian product center and exponent
						gamma = alpha_i + alpha_j
						P = (alpha_i * basis_i.center + alpha_j * basis_j.center) / gamma

						# Pre-exponential factor
						K = np.exp(-alpha_i * alpha_j * center_dist * center_dist / gamma)

						# Compute kinetic energy terms for each direction
						for dim in range(3):
							l1, m1, n1 = basis_i.angular_momentum
							l2, m2, n2 = basis_j.angular_momentum

							# Get angular momentum for current dimension
							L1 = [l1, m1, n1][dim]
							L2 = [l2, m2, n2][dim]

							# Position differences
							PA = P[dim] - basis_i.center[dim]
							PB = P[dim] - basis_j.center[dim]

							# Kinetic energy contribution
							term = 0.0

							# Term 1: -2α₂² contribution
							if L2 > 1:
								term += -0.5 * L2 * (L2 - 1) * overlap_1d(L1, L2 - 2, PA, PB, gamma)

							# Term 2: 2α₂(2L₂ + 1) contribution
							term += alpha_j * (2 * L2 + 1) * overlap_1d(L1, L2, PA, PB, gamma)

							# Term 3: -4α₂² r² contribution
							term += -2 * alpha_j * alpha_j * overlap_1d(L1, L2 + 2, PA, PB, gamma)

							t_ij += prim_i.coefficient * prim_j.coefficient * K * term

				T[i, j] = t_ij

		return T

	def compute_nuclear_attraction_matrix(self) -> np.ndarray:
		"""
		Compute the nuclear attraction matrix V.
		The nuclear attraction integral between two Gaussian basis functions is:
		V_ij = -∑ₖ Zₖ ∫ φᵢ(r) (1/|r-Rₖ|) φⱼ(r) dr
		where Zₖ is the nuclear charge and Rₖ is the nuclear position.
		"""
		V = np.zeros((self.n_basis, self.n_basis))

		for i, basis_i in enumerate(self.basis_functions):
			for j, basis_j in enumerate(self.basis_functions):
				# Skip if centers are too far apart
				center_dist = np.linalg.norm(basis_i.center - basis_j.center)
				if center_dist > 10.0:
					continue

				# Initialize nuclear attraction for this basis pair
				v_ij = 0.0

				# Loop over all nuclei
				for k, Z in enumerate(self.molecule.atomic_numbers):
					R = self.molecule.coordinates[k]

					# Compute contribution from each primitive pair
					for prim_i in basis_i.primitives:
						for prim_j in basis_j.primitives:
							alpha_i = prim_i.exponent
							alpha_j = prim_j.exponent

							# Skip if exponents are too different
							if max(alpha_i, alpha_j) / min(alpha_i, alpha_j) > 1e4:
								continue

							# Gaussian product center and exponent
							gamma = alpha_i + alpha_j
							P = (alpha_i * basis_i.center + alpha_j * basis_j.center) / gamma

							# Pre-exponential factor
							K = np.exp(-alpha_i * alpha_j * center_dist * center_dist / gamma)

							# Distance between Gaussian product center and nucleus
							RP = np.linalg.norm(R - P)
							if RP < 1e-10:  # Avoid division by zero
								RP = 1e-10

							# Boys function argument
							x = gamma * RP * RP

							# Compute Boys function (using error function for F₀)
							F0 = np.sqrt(np.pi / (4 * x)) * erf(np.sqrt(x)) if x > 1e-6 else 1.0

							# Compute angular momentum terms
							angular_term = 1.0
							for dim in range(3):
								l1, m1, n1 = basis_i.angular_momentum
								l2, m2, n2 = basis_j.angular_momentum

								# Get angular momentum for current dimension
								L1 = [l1, m1, n1][dim]
								L2 = [l2, m2, n2][dim]

								# Position differences
								PA = P[dim] - basis_i.center[dim]
								PB = P[dim] - basis_j.center[dim]
								PC = P[dim] - R[dim]

								# Compute overlap-like term for this dimension
								term = 0.0
								for t in range(L1 + L2 + 1):
									if t % 2 == 0:  # Only even terms contribute
										term += overlap_1d(L1, L2, PA, PB, gamma) * (PC**t)

							angular_term *= term

							# Add contribution to nuclear attraction integral
							v_ij -= Z * K * angular_term * F0 * 2 * np.pi / gamma

				V[i, j] = v_ij

		return V

	def compute_electron_repulsion_integrals(self) -> np.ndarray:
		"""
		Compute the two-electron repulsion integrals.
		The electron repulsion integral between four Gaussian basis functions is:
		(ij|kl) = ∫∫ φᵢ(r₁)φⱼ(r₁) (1/|r₁-r₂|) φₖ(r₂)φₗ(r₂) dr₁dr₂
		"""
		n = self.n_basis
		eri = np.zeros((n, n, n, n))

		for i, basis_i in enumerate(self.basis_functions):
			for j, basis_j in enumerate(self.basis_functions):
				# First electron center distance
				Rij = np.linalg.norm(basis_i.center - basis_j.center)
				if Rij > 10.0:  # Skip if centers are too far apart
					continue

				for k, basis_k in enumerate(self.basis_functions):
					for l, basis_l in enumerate(self.basis_functions):
						# Second electron center distance
						Rkl = np.linalg.norm(basis_k.center - basis_l.center)
						if Rkl > 10.0:  # Skip if centers are too far apart
							continue

						# Initialize integral for this basis function quartet
						eri_ijkl = 0.0

						# Loop over all primitive combinations
						for prim_i in basis_i.primitives:
							for prim_j in basis_j.primitives:
								alpha_i = prim_i.exponent
								alpha_j = prim_j.exponent

								# Skip if exponents are too different
								if max(alpha_i, alpha_j) / min(alpha_i, alpha_j) > 1e4:
									continue

								# First Gaussian product
								gamma1 = alpha_i + alpha_j
								P = (alpha_i * basis_i.center + alpha_j * basis_j.center) / gamma1
								K1 = np.exp(-alpha_i * alpha_j * Rij * Rij / gamma1)

								for prim_k in basis_k.primitives:
									for prim_l in basis_l.primitives:
										alpha_k = prim_k.exponent
										alpha_l = prim_l.exponent

										# Skip if exponents are too different
										if max(alpha_k, alpha_l) / min(alpha_k, alpha_l) > 1e4:
											continue

										# Second Gaussian product
										gamma2 = alpha_k + alpha_l
										Q = (alpha_k * basis_k.center + alpha_l * basis_l.center) / gamma2
										K2 = np.exp(-alpha_k * alpha_l * Rkl * Rkl / gamma2)

										# Distance between Gaussian products
										RPQ = np.linalg.norm(P - Q)
										if RPQ < 1e-10:
											RPQ = 1e-10

										# Combined exponent
										eta = gamma1 * gamma2 / (gamma1 + gamma2)

										# Boys function argument
										x = eta * RPQ * RPQ

										# Compute Boys function (using error function for F₀)
										F0 = np.sqrt(np.pi / (4 * x)) * erf(np.sqrt(x)) if x > 1e-6 else 1.0

										# Compute angular momentum terms
										angular_term = 1.0
										for dim in range(3):
											l1, m1, n1 = basis_i.angular_momentum
											l2, m2, n2 = basis_j.angular_momentum
											l3, m3, n3 = basis_k.angular_momentum
											l4, m4, n4 = basis_l.angular_momentum

											# Get angular momentum for current dimension
											L1 = [l1, m1, n1][dim]
											L2 = [l2, m2, n2][dim]
											L3 = [l3, m3, n3][dim]
											L4 = [l4, m4, n4][dim]

											# Position differences
											PA = P[dim] - basis_i.center[dim]
											PB = P[dim] - basis_j.center[dim]
											QC = Q[dim] - basis_k.center[dim]
											QD = Q[dim] - basis_l.center[dim]

											# Compute overlap-like terms
											term1 = overlap_1d(L1, L2, PA, PB, gamma1)
											term2 = overlap_1d(L3, L4, QC, QD, gamma2)
											angular_term *= term1 * term2

										# Add contribution to ERI
										prefactor = 2 * np.pi * np.pi / (gamma1 * gamma2 * np.sqrt(gamma1 + gamma2))
										eri_ijkl += (
											prim_i.coefficient
											* prim_j.coefficient
											* prim_k.coefficient
											* prim_l.coefficient
											* K1
											* K2
											* angular_term
											* F0
											* prefactor
										)

						eri[i, j, k, l] = eri_ijkl

		return eri

	def compute_initial_guess(self) -> np.ndarray:
		"""Generate initial guess for density matrix."""
		return np.eye(self.n_basis) * 0.1

	def compute_fock_matrix(self, D: np.ndarray, h_core: np.ndarray, eri: np.ndarray) -> np.ndarray:
		"""Compute the Fock matrix."""
		F = h_core.copy()
		for i_basis in range(self.n_basis):
			for j_basis in range(self.n_basis):
				for k_basis in range(self.n_basis):
					for l_basis in range(self.n_basis):
						F[i_basis, j_basis] += D[k_basis, l_basis] * (
							2.0 * eri[i_basis, j_basis, k_basis, l_basis] - eri[i_basis, k_basis, j_basis, l_basis]
						)
		return F

	def compute(self) -> Tuple[float, np.ndarray, np.ndarray]:
		"""
		Perform the SCF calculation.

		Returns:
		    Tuple[float, np.ndarray, np.ndarray]: A tuple containing:
		        - final_energy: The converged total energy in Hartree
		        - orbital_energies: Array of orbital energies
		        - orbital_coefficients: Matrix of orbital coefficients
		"""
		# Compute basic integrals
		S = self.compute_overlap_matrix()
		T = self.compute_kinetic_matrix()
		V = self.compute_nuclear_attraction_matrix()
		eri = self.compute_electron_repulsion_integrals()

		# Check if overlap matrix has any NaN or Inf values
		if np.any(np.isnan(S)) or np.any(np.isinf(S)):
			raise RuntimeError("Overlap matrix contains NaN or Inf values. Check basis set parameters.")

		# Print overlap matrix for debugging
		print("\nOverlap Matrix:")
		np.set_printoptions(precision=6, suppress=True)
		print(S)

		# Core Hamiltonian
		h_core = T + V

		# Compute S^(-1/2) for orthogonalization using canonical orthogonalization
		try:
			s_eigenvalues, s_eigenvectors = np.linalg.eigh(S)
		except np.linalg.LinAlgError:
			raise RuntimeError("Failed to diagonalize overlap matrix. Matrix may be singular.")

		# Print eigenvalues for debugging
		print("\nOverlap Matrix Eigenvalues:")
		print(s_eigenvalues)

		# Find valid eigenvalues and corresponding eigenvectors
		max_eigenvalue = np.max(s_eigenvalues)
		threshold = max_eigenvalue * 1e-10  # More conservative threshold

		# Filter out negative and very small eigenvalues
		valid_mask = s_eigenvalues > threshold
		valid_eigenvalues = s_eigenvalues[valid_mask]
		valid_eigenvectors = s_eigenvectors[:, valid_mask]

		if len(valid_eigenvalues) == 0:
			raise RuntimeError("No valid eigenvalues found in overlap matrix")

		n_removed = np.sum(~valid_mask)
		if n_removed > 0:
			print(f"\nRemoved {n_removed} small/negative eigenvalues below {threshold:.2e}")
			print(f"Remaining eigenvalues: {len(valid_eigenvalues)}")

			if self.n_occupied > len(valid_eigenvalues):
				raise RuntimeError(
					f"Too many electrons ({self.n_electrons}) for the reduced basis set "
					f"(remaining functions: {len(valid_eigenvalues)})"
				)

		# Compute condition number with valid eigenvalues
		min_eigenvalue = np.min(valid_eigenvalues)
		condition_number = max_eigenvalue / min_eigenvalue
		print(f"\nOverlap matrix condition number (after filtering): {condition_number:.2e}")

		try:
			# Compute S^(-1/2) using only valid eigenvalues/vectors
			sqrt_s_eigenvalues = np.sqrt(valid_eigenvalues)
			S_half_inv = valid_eigenvectors @ np.diag(1.0 / sqrt_s_eigenvalues) @ valid_eigenvectors.T

			# Verify the result
			if np.any(np.isnan(S_half_inv)) or np.any(np.isinf(S_half_inv)):
				raise RuntimeError("S^(-1/2) matrix contains NaN or Inf values")

			# Additional verification
			test_product = S_half_inv @ S @ S_half_inv
			if not np.allclose(test_product, test_product.T, rtol=1e-5, atol=1e-8):
				raise RuntimeError("S^(-1/2) verification failed: Result is not symmetric")

		except (RuntimeError, FloatingPointError) as e:
			print("\nDetailed overlap matrix analysis:")
			print(f"Shape of S: {S.shape}")
			print(f"Rank of S: {np.linalg.matrix_rank(S)}")
			print(f"Condition number of S: {np.linalg.cond(S)}")
			raise RuntimeError(f"Failed to compute S^(-1/2) matrix: {str(e)}")

		# Initial guess
		D = self.compute_initial_guess()

		# SCF procedure
		E_old = 0.0
		D_old = None
		converged = False

		# DIIS parameters
		diis_start = 3  # Start DIIS after this many iterations
		diis_dim = 6  # Maximum number of Fock matrices to use
		F_list = []  # List of Fock matrices
		e_list = []  # List of error matrices

		# Level shifting parameters
		level_shift = 0.0  # Initial level shift
		max_level_shift = 2.0  # Maximum level shift
		level_shift_factor = 1.2  # Factor to increase level shift

		# Damping parameters
		damping = 0.0  # Initial damping
		max_damping = 0.75  # Maximum damping
		damping_factor = 1.2  # Factor to increase damping

		# Energy history for oscillation detection
		energy_history = []

		for iteration in range(self.max_iterations):
			# Build Fock matrix
			F = self.compute_fock_matrix(D, h_core, eri)

			# Store current state
			if D_old is None:
				D_old = D.copy()

			# Check for energy oscillations
			if len(energy_history) >= 3:
				e1, e2, e3 = energy_history[-3:]
				if abs(e1 - e3) < abs(e1 - e2):  # Oscillating pattern
					damping = min(damping * damping_factor, max_damping)
					if damping == max_damping:  # If maximum damping reached, increase level shift
						level_shift = min(level_shift * level_shift_factor, max_level_shift)

			# Apply level shift to virtual orbitals
			F_shifted = F.copy()
			if level_shift > 0:
				# Add level shift to diagonal elements corresponding to virtual orbitals
				F_shifted += level_shift * S

			# DIIS extrapolation
			if iteration >= diis_start and len(F_list) > 0:
				# Compute DIIS error matrix
				FDS = F_shifted @ D @ S
				SDF = S @ D @ F_shifted
				error = FDS - SDF

				# Update DIIS matrices
				F_list.append(F_shifted.copy())
				e_list.append(error)

				if len(F_list) > diis_dim:
					F_list.pop(0)
					e_list.pop(0)

				# Build and solve DIIS equations
				n_diis = len(F_list)
				B = np.zeros((n_diis + 1, n_diis + 1))
				for i in range(n_diis):
					for j in range(n_diis):
						B[i, j] = np.trace(e_list[i] @ e_list[j].T)
				B[-1, :] = -1
				B[:, -1] = -1
				B[-1, -1] = 0

				# Right-hand side of DIIS equations
				resid = np.zeros(n_diis + 1)
				resid[-1] = -1

				try:
					c = np.linalg.solve(B, resid)
					F_shifted = sum(c[i] * F_list[i] for i in range(n_diis))
				except np.linalg.LinAlgError:
					print(f"DIIS extrapolation failed at iteration {iteration + 1}, continuing without it")

			# Transform Fock matrix
			F_prime = S_half_inv.T @ F_shifted @ S_half_inv

			# Solve eigenvalue problem
			try:
				orbital_energies, C_prime = np.linalg.eigh(F_prime)
			except np.linalg.LinAlgError:
				raise RuntimeError("Failed to diagonalize Fock matrix")

			# Transform coefficients back
			C = S_half_inv @ C_prime

			# Build new density matrix with damping
			D_new = np.zeros_like(D)
			for i_basis in range(self.n_basis):
				for j_basis in range(self.n_basis):
					for k_orbital in range(self.n_occupied):
						D_new[i_basis, j_basis] += 2.0 * C[i_basis, k_orbital] * C[j_basis, k_orbital]

			if damping > 0:
				D_new = (1 - damping) * D_new + damping * D

			# Compute energy
			E_electronic = 0.0
			for i_basis in range(self.n_basis):
				for j_basis in range(self.n_basis):
					E_electronic += 0.5 * D_new[i_basis, j_basis] * (h_core[i_basis, j_basis] + F[i_basis, j_basis])

			E_total = E_electronic + self.molecule.nuclear_repulsion_energy()

			# Update energy history
			energy_history.append(E_total)
			if len(energy_history) > 3:
				energy_history.pop(0)

			# Check convergence
			delta_E = abs(E_total - E_old)
			delta_D = np.max(np.abs(D_new - D))

			print(f"Iteration {iteration + 1:3d}: E = {E_total:15.10f} ΔE = {delta_E:10.3e} ΔD = {delta_D:10.3e}")
			if level_shift > 0 or damping > 0:
				print(f"           Level shift: {level_shift:.6f}  Damping: {damping:.6f}")

			if delta_E < self.convergence_threshold and delta_D < self.convergence_threshold:
				converged = True
				break

			# Store old values and update density
			E_old = E_total
			D_old = D.copy()
			D = D_new.copy()

		if not converged:
			raise RuntimeError(f"SCF failed to converge in {self.max_iterations} iterations")

		print("\nSCF Converged!")
		print(f"Final Energy: {E_total:15.10f} Hartree")
		print(f"Convergence: ΔE = {delta_E:10.3e}, ΔD = {delta_D:10.3e}")

		return E_total, orbital_energies, C
