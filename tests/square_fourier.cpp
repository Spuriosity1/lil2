#include "supercell.hpp"
#include "fourier.hpp"
#include <cmath>
#include <complex>
#include <iostream>

using namespace vector3;

struct Spin {
    ipos_t ipos;
    double Sz;

    Spin() : ipos(-1,-1,-1), Sz(0.0) {}
    Spin(const ipos_t& x) : ipos(x), Sz(0.0) {}
};
