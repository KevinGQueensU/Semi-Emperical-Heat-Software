// Gmsh project created on Thu Mar 05 22:24:08 2026
SetFactory("OpenCASCADE");
//+
Point(1) = {0, -1, -1, 1.0};
//+
Point(2) = {0, -1, 1, 1.0};
//+
Point(3) = {0, 1, 1, 1.0};
//+
Point(4) = {0, 1, -1, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Extrude {1, 0, 0} {
  Curve{1}; Curve{2}; Curve{3}; Curve{4}; Layers {100}; Recombine;
}
//+
Extrude {1, 0, 0} {
  Curve{9}; Curve{11}; Curve{12}; Curve{7}; Layers {100}; Recombine;
}
//+
Curve Loop(10) = {9, 11, 12, 7};
//+
Plane Surface(10) = {10};
//+
Surface Loop(1) = {1, 2, 5, 4, 3, 10};
//+
Volume(1) = {1};
//+
Curve Loop(15) = {17, 19, 20, 15};
//+
Plane Surface(15) = {15};
//+
Surface Loop(2) = {6, 9, 8, 7, 15, 10};
//+
Volume(2) = {2};

//+
Box(3) = {-0.4, 0.6, -1, 2/5, 2/5, 2};
//+
Box(4) = {-0.4, -1, -1, 2/5, 2/5, 2};
//+
Physical Volume("Aluminum", 45) = {2};
//+
Physical Volume("Olivine", 46) = {1};
//+
Physical Volume("Tantalum", 47) = {3, 4};
//+
BooleanFragments{ Volume{1}; Delete; }{Volume{2}; Delete; }
//+
//+
Transfinite Surface {23} = {16, 4, 19, 15};
//+
Transfinite Curve {23, 32, 3, 31} = 100 Using Bump 1;
//+
Transfinite Curve {21, 22, 23, 24} = 100 Using Bump 1;
//+
Transfinite Curve {32, 22, 30, 2} = 100 Using Bump 1;
