// Gmsh project created on Sun Feb 22 21:20:14 2026
SetFactory("OpenCASCADE");
//+
Box(1) = {0, -1, -1, 0.001, 2, 2};
//+
Box(2) = {0.001, -1, -1, 0.01, 2, 2};
//+
Box(3) = {-0.4, 0.6, -1, 2/5, 2/5, 2};
//+
Box(4) = {-0.4, -1, -1, 2/5, 2/5, 2};
//+
Physical Surface("Fixed", 54) = {6, 23, 5, 22, 14};
//+
Physical Volume("Tantalum", 55) = {4, 3};
//+
Physical Volume("Olivine", 56) = {1};
//+
Physical Volume("Aluminum", 57) = {2};
//+
BooleanFragments{ Volume{3}; Volume{1}; Volume{2}; Volume{4}; Delete; }{ }
//+
Physical Surface("BBR", 52) = {8, 19, 21, 20, 4, 11, 16, 1, 3, 13, 18, 10, 15, 9, 17};
//+      
Transfinite Curve {1, 5, 3, 7} = 25 Using Bump 1;
//+
Transfinite Curve {9, 11, 8, 4, 2, 10, 6, 12} = 10 Using Bump 1;
//+
Transfinite Curve {35, 37, 15, 35, 13} = 25 Using Bump 1;
//++
Transfinite Curve {38, 39, 16, 41, 40, 36, 42, 14} = 10 Using Bump 1;//+

//+
Field[1] = Box;
//+
Field[1].VIn = 0.01;
//+
Field[1].XMax= 0.011;
Field[1].XMin = 0;
//+
Field[1].YMax = 1;
//+
Field[1].YMin = -1;
//+
Field[1].ZMax = 1;
//+
Field[1].ZMin = -1;

//+
Background Field = 1;
