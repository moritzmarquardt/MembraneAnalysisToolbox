# Membrane Analysis Toolbox

The MembraneAnalysisToolbox is a python package than can be installed in a python environment via pip.

This toolbox contains methods for analysing molecular dynamics simulation data of Membranes.
It has been developed in the group of Kristyna Pluhackova at the University of Stuttgart for analysing nanoporous carbon materials and their properties. However due to the modular and object oriented structure, it may be used or extended for other projects and the analysis of all kinds of membranes in molecular dynamics simulations.

## Main Components:
- TransitionPathAnalysis class with the following key features:
    - Analyse passages through membrane and their **passage time distribution**
    - Calculate the **diffusion coefficient** for a solvent
- EffectivePoreSizeAnalysis class with the following key features:
    - Analyse the **effective pore radius** of a membrane pore
    - Analyse the density of molecules in the pore using kernel methods for a **density heatmap**


## Installation:
MembraneAnalysisToolbox is developed using python 3.12 and installing it requires a minimum of python 3.12.

1) Download or clone this repository
2) Run `pip install [path to package]` in your python environment

## Usage
- Look at `examples/` to see how the package can be used if you want to write your own code. 
- This python package was used for a publication on nanoporous carbon materials. Find the code and how the package was used for the paper here: [https://github.com/moritzmarquardt/carbon_paper_MAT](https://github.com/moritzmarquardt/carbon_paper_MAT). 

## Acknowledgement
This Toolbox uses work by Gotthold Fläschner.