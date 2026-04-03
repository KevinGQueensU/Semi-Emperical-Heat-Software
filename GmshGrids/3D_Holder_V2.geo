// Gmsh project created on Sun Feb 22 21:20:14 2026
SetFactory("OpenCASCADE");
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

//+
Box(1) = {0, -1, -1, 0.001, 2, 2};
//+
Box(2) = {0.001, -1, -1, 0.1, 2, 2};
//+
Box(3) = {-0.4, 0.6, -1, 2/5, 2/5, 2};
//+
Box(4) = {-0.4, -1, -1, 2/5, 2/5, 2};
//+
Physical Volume("Tantalum", 55) = {4, 3};

//+
BooleanFragments{ Volume{3}; Volume{1}; Volume{2}; Volume{4}; Delete; }{ }
Recursive Delete {
  Volume{1}; Volume{2}; 
}
//+
Line(43) = {8, 9};
//+
Line(44) = {9, 10};
//+
Line(45) = {10, 9};
//+
Line(46) = {10, 7};
//+
Line(47) = {7, 8};
//+
Extrude {0.001, 0, 0} {
  Curve{46}; Curve{7}; Curve{13}; Curve{43}; Layers {100}; Recombine;
}
//+//+
Extrude {0.01, 0, 0} {
  Curve{50}; Curve{52}; Curve{55}; Curve{54}; Layers {50}; Recombine;
}
//+
Curve Loop(32) = {46, -7, 43, 13};
//+
Plane Surface(32) = {32};
//+
Curve Loop(33) = {52, -50, -54, -55};
//+
Plane Surface(33) = {33};
//+
Surface Loop(1) = {33, 25, 27, 26, 24, 32};

Curve Loop(38) = {58, -60, 62, 63};
//+
Plane Surface(38) = {38};
//+
Surface Loop(2) = {38, 28, 31, 30, 29, 33};
//+
Volume(6) = {2};

//+
Surface Loop(3) = {26, 27, 25, 24, 33, 32};
//+
Volume(7) = {3};
//+
Surface Loop(4) = {2, 3, 5, 1, 4, 6};
