# KINETIC MONTE CARLO FOR DIFFUSION AND DESORPTION

This is a Kinetic Monte Carlo script written in Python 3.10 that explicitly consider diffusion and desorption rates.

## Installation

To run the code is necessary to install the following packages:

- numpy
- pandas
- ase

## Script structure

The main script run_KMC.py take advantage of several files that should always be located in the same path as the one of run_KMC.py. The files are organized as follow:

- run_KMC.py --> Main code to be run
- KMC_functions.py --> Series of functions used by the main code
- NEW_FINAL_git.csv --> CSV file in which are stored the information about the diffusion and desorption barriers
- sulfur_positions.txt --> position of the adsorbate, used to compute the diffusion coefficient


## Associated publications

Please refer to the following publication to cite our work or retrieve the info:


## Acknowledgments

This project has received funding within the European Union’s Horizon 2020 research and innovation programme from the European Research Council (ERC) for the project, ”Quantum Chemistry on Interstellar Grains” (QUANTUMGRAIN), grant agreement No 865657.

