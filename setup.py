from setuptools import find_packages, setup

setup(
	name="quantumchem",
	version="0.1.0",
	packages=find_packages(),
	install_requires=[
		"numpy>=1.21.0",
		"scipy>=1.7.0",
	],
	extras_require={
		"dev": [
			"pytest>=7.0.0",
		],
	},
	author="Justin Kirkland",
	description="A quantum chemistry program for performing Hartree-Fock calculations",
	python_requires=">=3.7",
)
