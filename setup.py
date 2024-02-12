from setuptools import setup, find_packages

setup(
    name='MembraneAnalysisToolbox',
    version='1.0',    
    description='A python packages for analysing Membranes and Carbon Materials',
    url='https://github.com/moritzmarquardt/MembraneAnalysisToolbox',
    author='Moritz Marquardt',
    author_email='st166545@stud.uni-stuttgart.de',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy', 'MDAnalysis'],
)
