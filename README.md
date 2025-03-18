# QuantumChem

A quantum chemistry program for performing Hartree-Fock calculations.

## Features

- XYZ coordinate input
- Basis set handling
- Hartree-Fock energy calculation
- Orbital energy computation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from quantumchem import HartreeFock
from quantumchem.molecule import Molecule

# Create a molecule from XYZ coordinates
molecule = Molecule.from_xyz("molecule.xyz")

# Initialize HF calculation with STO-3G basis
hf = HartreeFock(molecule, basis="STO-3G")

# Run the calculation
energy = hf.compute()
print(f"HF Energy: {energy:.6f} Hartree")
```
