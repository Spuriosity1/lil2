#include "supercell.hpp"
#include <iostream>

using namespace vector3;

struct S { ipos_t ipos; };

int main() {
    for (int L : {4, 8}) {
        UnitCellSpecifier<S> cell(imat33_t::from_cols({8,0,0},{0,8,0},{0,0,8}));
        cell.add<S>(S{{0,0,0}});

        auto Z_llx = imat33_t::from_cols({L,0,0},{0,L,0},{0,0,2*L});
        auto sc1 = build_supercell<S>(cell, Z_llx);
        auto D1 = sc1.lattice.size();
        std::cout << "Z=diag(" << L << "," << L << "," << 2*L << ") -> D=("
                  << D1[0] << "," << D1[1] << "," << D1[2] << ")\n";

        auto Z_xll = imat33_t::from_cols({2*L,0,0},{0,L,0},{0,0,L});
        auto sc2 = build_supercell<S>(cell, Z_xll);
        auto D2 = sc2.lattice.size();
        std::cout << "Z=diag(" << 2*L << "," << L << "," << L << ") -> D=("
                  << D2[0] << "," << D2[1] << "," << D2[2] << ")\n";
    }
    return 0;
}
