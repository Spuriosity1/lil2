#include "supercell.hpp"
#include <iostream>
#include <gtest/gtest.h>

using namespace vector3;

struct S { ipos_t ipos; };

TEST(SnfDims, DiagonalSizes) {
    for (int L : {4, 8}) {
        UnitCellSpecifier<S> cell(imat33_t::from_cols({8,0,0},{0,8,0},{0,0,8}));
        cell.add<S>(S{{0,0,0}});

        auto Z_llx = imat33_t::from_cols({L,0,0},{0,L,0},{0,0,2*L});
        auto sc1 = build_supercell<S>(cell, Z_llx);
        auto D1 = sc1.lattice.size();
        std::cout << "Z=diag(" << L << "," << L << "," << 2*L << ") -> D=("
                  << D1[0] << "," << D1[1] << "," << D1[2] << ")\n";
        EXPECT_EQ(D1[0]*D1[1]*D1[2], 2*L*L*L)
            << "Product of dims should equal det(Z) = 2*L^3 for L=" << L;

        auto Z_xll = imat33_t::from_cols({2*L,0,0},{0,L,0},{0,0,L});
        auto sc2 = build_supercell<S>(cell, Z_xll);
        auto D2 = sc2.lattice.size();
        std::cout << "Z=diag(" << 2*L << "," << L << "," << L << ") -> D=("
                  << D2[0] << "," << D2[1] << "," << D2[2] << ")\n";
        EXPECT_EQ(D2[0]*D2[1]*D2[2], 2*L*L*L)
            << "Product of dims should equal det(Z) = 2*L^3 for L=" << L;

        // Both supercells have the same number of sites
        EXPECT_EQ(sc1.lattice.num_primitive_cells(), sc2.lattice.num_primitive_cells());
    }
}
