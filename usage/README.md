This usage/ folder contains python scripts that use the MembraneAnalysisToolbox to offer an interactive analysis for different problems. For example a interactive Diffusion Anaylsis that can be run by the user to analyse the Diffusion through a membrane.

## How to use the scripts?
- (if necessary) log into workstation: `ssh -X st123456@129.69.167.51` (-X is optional to forward plots to the local machine)
- `cd /bigpool/users/st166545/MembraneAnalysisToolbox` to get to the library
- `source .venv/bin/activate` to activate the virtual environment in which the library is installed
- (optional) `which python` to ensure that the "python" refers to the virtual environment python under "/bigpool/users/st166545/MembraneAnalysisToolbox/.venv/bin/python"
- `cd usage/` to go to the folder with the interactive anaylses. You can view the files here with `ll`
- `python interactive_DiffusionAnalysis.py` to run the interactive Diffusion Analysis for example.
