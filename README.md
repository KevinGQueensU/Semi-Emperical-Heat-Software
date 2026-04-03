To get mayavi working, try doing the following steps:
    1) Create a virtual environment in Python 3.11 (I used Conda)
    2) Run pip install numpy==1.26.4 vtk==9.3.1
    3) Run pip install PyQt5
    4) Run pip install mayavi --no-cache-dir --verbose  --no-build-isolation
This should hopefully successfully build MayAVI. Try VTK=9.3.1 and VTK = 9.3.2 if this doesn't work.