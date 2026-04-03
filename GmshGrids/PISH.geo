SetFactory("OpenCASCADE");
Mesh.Recombine3DAll = 1;

// create a rectangle in yz
Point(1) = {0,-1.5,-1.5};
Point(2) = {0, 1.5,-1.5};
Point(3) = {0, 1.5, 1.5};
Point(4) = {0,-1.5, 1.5};
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
a[] = Extrude {0.005, 0,0} { Surface{1}; Layers{15}; Recombine; };

// Aluminum slab
b[] = Extrude {0.2,0,0} {
    Surface{a[0]};
    Layers{ {20, 5}, {0.06, 1.0} };
    Recombine;
};

// force structured volumes
Transfinite Volume{a[1]}; Recombine Volume{a[1]};
Transfinite Volume{b[1]}; Recombine Volume{b[1]};

//+
Box(3) = {-0.4, 1.1, -1.5, 2/5, 2/5, 3};
//+
Box(4) = {-0.4, -1.5, -1.5, 2/5, 2/5, 3};
//+
Physical Volume("Tantalum", 45) = {3, 4};
//+
Physical Volume("Olivine", 46) = {1};
//+
Physical Volume("Aluminum", 47) = {2};

//+
Physical Surface("BBR", 48) = {7, 8, 3, 2, 15, 12, 14, 1, 18, 20, 21, 4, 9, 10, 5};

//+
Physical Surface("Fixed", 49) = {17, 23, 16, 22};

//+
Physical Surface("Interface", 50) = {11};//+
Physical Surface("Internal", 51) = {6, 13, 19};
