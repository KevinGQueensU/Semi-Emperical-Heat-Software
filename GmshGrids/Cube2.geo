SetFactory("OpenCASCADE");
Mesh.Recombine3DAll = 1;
Mesh.MeshSizeFactor = 0.5;

//+
SetFactory("OpenCASCADE");
Box(1) = {0, -1, -1, 2, 2, 2};
//+
Physical Volume("Olivine", 25) = {1};
//+
Physical Surface("BBR", 26) = {6, 4, 5, 3};

Physical Surface("Fixed", 27) = {2};