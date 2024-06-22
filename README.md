# Membrane Analysis Toolbox

This toolbox contains several methods for analysing Carbon Materials and Membranes.
It has been developed in the group of Kristyna Pluhackova at the University of Stuttgart.
It also uses work by Gotthold Fläschner.

## Main Capabilities:
- TransitionPathAnalysis class with multiple features:
    - analyse membrane and trajectories using only the traj.xtc and the topol.tpr file
    - analyse membrane dimensions and location
    - plot membrane histograms
    - analyse passages through membrane and their **passage time distribution**
    - calculate the **diffusion coefficient** for a solvent
    - calculate the **path length** through a cubic structure membrane
- EffectivePoreSizeAnalysis
    - analyse the **effective radius** of membrane pores
    - analyse the density of molecules in the pore using kernels for a **density heatmap**


## Usage (how to install and use this python package):

1) Download or clone this repository
2) run ```pip install [path to package]``` in your python environment
3) look at examples/ to see how the classes are imported and used
