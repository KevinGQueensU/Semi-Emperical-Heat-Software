SetFactory("OpenCASCADE");
Mesh.Recombine3DAll = 1;

// create a rectangle in yz
Point(1) = {0,-1,-1};
Point(2) = {0, 1,-1};
Point(3) = {0, 1, 1};
Point(4) = {0,-1, 1};
Line(1) = {1,2}; 
Line(2) = {2,3}; 
Line(3) = {3,4}; 
Line(4) = {4,1};
Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

// yz mesh resolution
Transfinite Curve{1,3} = 40; // y
Transfinite Curve{2,4} = 40; // z
Transfinite Surface{1};
Recombine Surface{1};

// Olivine slab
a[] = Extrude {0.0001, 0,0} { Surface{1}; Layers{5}; Recombine; };

// Aluminum slab
b[] = Extrude {0.001,0,0} { Surface{a[0]}; Layers{5}; Recombine; };

// force structured volumes (again: indices can vary; see note below)
Transfinite Volume{a[1]}; Recombine Volume{a[1]};
Transfinite Volume{b[1]}; Recombine Volume{b[1]};

//+
Physical Volume("Olivine", 46) = {1};
//+
Physical Volume("Aluminum", 47) = {2};

//+
Physical Surface("Fixed", 48) = {1, 11};
