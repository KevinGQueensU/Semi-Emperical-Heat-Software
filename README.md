# Introduction
This package allows for simulation of particle beam irradiation heating in solid materials, with support for Gmsh custom grids. This was created for an undergraduate thesis at Queen's University in the Engineering Physics department. 

For information about the software and theory, see [Irradiation_Paleo_Thesis.pdf](/Irradiation_Paleo_Thesis.pdf) and the documentation

# Usage

Install required packages with `pip install -r requirements.txt`

To get mayavi working, try doing the following steps: \
1) Create a virtual environment in Python 3.11 (I used Conda) 
2) Run pip install numpy==1.26.4 vtk==9.3.1 
3) Run pip install PyQt5 
4) Run pip install mayavi --no-cache-dir --verbose  --no-build-isolation

This should hopefully successfully build MayAVI. Try VTK=9.3.1 and VTK = 9.3.2 if this doesn't work. Otherwise, good luck with it. (You can disable it with View=Disable in the 3D and GMSH sim functions)

**Beam Irradiation Heat Simulation Software | Undergraduate Thesis**
