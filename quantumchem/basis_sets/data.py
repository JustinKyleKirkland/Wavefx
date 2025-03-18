"""
Library of common basis sets.

Data format for each basis set:
{
    atomic_number: {
        orbital_type: {
            'exponents': [...],
            'coefficients': [...],
            'angular_momentum': (l, m, n)
        }
    }
}
"""

# STO-3G minimal basis set
# Reference: W. J. Hehre, R. F. Stewart, and J. A. Pople, J. Chem. Phys. 51, 2657 (1969)
STO3G = {
	1: {  # Hydrogen
		"1s": {
			"exponents": [3.425250914, 0.6239137298, 0.1688554040],
			"coefficients": [0.1543289673, 0.5353281423, 0.4446345422],
			"angular_momentum": (0, 0, 0),
		}
	},
	6: {  # Carbon
		"1s": {
			"exponents": [71.6168370, 13.0450960, 3.5305122],
			"coefficients": [0.1543289673, 0.5353281423, 0.4446345422],
			"angular_momentum": (0, 0, 0),
		},
		"2s": {
			"exponents": [2.9412494, 0.6834831, 0.2222899],
			"coefficients": [-0.0999672292, 0.3995128261, 0.7001154689],
			"angular_momentum": (0, 0, 0),
		},
		"2p": {
			"exponents": [2.9412494, 0.6834831, 0.2222899],
			"coefficients": [0.1559162750, 0.6076837186, 0.3919573931],
			"angular_momentum": (1, 0, 0),  # px
		},
	},
	7: {  # Nitrogen
		"1s": {
			"exponents": [99.1061690, 18.0523124, 4.8856602],
			"coefficients": [0.1543289673, 0.5353281423, 0.4446345422],
			"angular_momentum": (0, 0, 0),
		},
		"2s": {
			"exponents": [3.7804559, 0.8784966, 0.2857144],
			"coefficients": [-0.0999672292, 0.3995128261, 0.7001154689],
			"angular_momentum": (0, 0, 0),
		},
		"2p": {
			"exponents": [3.7804559, 0.8784966, 0.2857144],
			"coefficients": [0.1559162750, 0.6076837186, 0.3919573931],
			"angular_momentum": (1, 0, 0),  # px
		},
	},
	8: {  # Oxygen
		"1s": {
			"exponents": [130.7093214, 23.8088661, 6.4436083],
			"coefficients": [0.1543289673, 0.5353281423, 0.4446345422],
			"angular_momentum": (0, 0, 0),
		},
		"2s": {
			"exponents": [5.0331513, 1.1695961, 0.3803890],
			"coefficients": [-0.0999672292, 0.3995128261, 0.7001154689],
			"angular_momentum": (0, 0, 0),
		},
		"2p": {
			"exponents": [5.0331513, 1.1695961, 0.3803890],
			"coefficients": [0.1559162750, 0.6076837186, 0.3919573931],
			"angular_momentum": (1, 0, 0),  # px
		},
	},
}

# 3-21G split-valence basis set
# Reference: J. S. Binkley, J. A. Pople, and W. J. Hehre, J. Am. Chem. Soc. 102, 939 (1980)
BASIS_321G = {
	1: {  # Hydrogen
		"1s": {
			"exponents": [5.4471780, 0.8245472],
			"coefficients": [0.1562850, 0.9046910],
			"angular_momentum": (0, 0, 0),
		},
		"2s": {"exponents": [0.1831916], "coefficients": [1.0000000], "angular_momentum": (0, 0, 0)},
	},
	# Add more atoms as needed
}

# 6-31G basis set
# Reference: W. J. Hehre, R. Ditchfield, and J. A. Pople, J. Chem. Phys. 56, 2257 (1972)
BASIS_631G = {
	1: {  # Hydrogen
		"1s": {
			"exponents": [18.7311370, 2.8253937, 0.6401217],
			"coefficients": [0.0334946, 0.2347269, 0.8137573],
			"angular_momentum": (0, 0, 0),
		},
		"2s": {"exponents": [0.1612778], "coefficients": [1.0000000], "angular_momentum": (0, 0, 0)},
	},
	# Add more atoms as needed
}

# Dictionary mapping basis set names to their data
BASIS_SETS = {"STO-3G": STO3G, "3-21G": BASIS_321G, "6-31G": BASIS_631G}
