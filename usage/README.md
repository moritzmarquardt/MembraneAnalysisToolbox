This folder contains files that can be used by users to actually run analyses and use the Toolbox. I wanted to have them here antd not in their own repos so they are all in one place and no copies exists.


Use by running the following commands:
- (if necessary) log into workstation: `ssh -X st123456@129.69.167.51`
- `cd /bigpool/users/st166545/MembraneAnalysisToolbox` to get to the library
- `source .venv/bin/activate` to activate the virtual environment in which the library is installed
- (optional) `which python` to ensure that the "python" refers to the virtual environment python under "/bigpool/users/st166545/MembraneAnalysisToolbox/.venv/bin/python"
- `cd usage/` to go to the folder with the interactive anaylses. You can view the files here with `ll`
- `python interactive_DiffusionAnalysis.py` to run the interactive Diffusion Analysis for example.