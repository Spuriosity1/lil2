#include "supercell.hpp"
#include "unitcellspec.hpp"
#include "fourier.hpp"
#include <fstream>
#include <random>
#include <set>
#include <stack>
#include "pyro_data.hpp"
#include <gtest/gtest.h>

struct Spin {
    ipos_t ipos;
    std::vector<Spin*> neighbours;

    Spin() : ipos(0,0,0) {}
    Spin(const ipos_t& x) : ipos(x) {}
};

inline ipos_t floordiv(const ipos_t& x, int base) {
    return ipos_t(x[0]/base, x[1]/base, x[2]/base);
}

using MyCell = UnitCellSpecifier<Spin>;
using SuperLat = Supercell<Spin>;

inline auto initialise_lattice(const imat33_t& Z) {
    using namespace pyrochlore;

    MyCell cell(imat33_t::from_cols({0,4,4},{4,0,4},{4,4,0}));
    for (int mu=0; mu<4; mu++)
        cell.add<Spin>(Spin(pyro[mu]));

    Supercell sc = build_supercell<Spin>(cell, Z);

    std::ofstream log("pyro_construct.log");
    for (const auto& [I, cell_] : sc.enumerate_cells())
        for (auto [spin_sl, spin] : cell_.enumerate_objects<Spin>())
            log << "Spin at " << spin->ipos << "\n";

    return sc;
}

static void link_spins(Supercell<Spin>& sc) {
    for (const auto& [I, cell_] : sc.enumerate_cells())
        for (auto [spin_sl, spin] : cell_.enumerate_objects<Spin>())
            for (const auto& dx : pyrochlore::nn_vectors[spin_sl]) {
                auto r_nn = spin->ipos + dx;
                auto nn = sc.get_object_at<Spin>(r_nn);
                spin->neighbours.push_back(nn);
            }
}

static int test_lookup_consistency(Supercell<Spin>& sc) {
    int failed = 0;
    for (const auto& [I, cell_] : sc.enumerate_cells()) {
        for (auto [spin_sl, spin] : cell_.enumerate_objects<Spin>()) {
            auto R_tmp = spin->ipos;
            auto R_tmp_2 = spin->ipos;
            sc.lattice.get_supercell_IDX(R_tmp);
            sc.lattice.wrap_primitive(R_tmp_2);
            assert(R_tmp == R_tmp_2);

            try {
                auto s = sc.get_object_at<Spin>(spin->ipos);
                if (s->ipos != spin->ipos) {
                    std::cerr << "Lookup broken at " << spin->ipos
                              << " (got " << s->ipos << ")" << std::endl;
                    failed++;
                }
            } catch (ObjectLookupError& e) {
                std::cerr << "Lookup broken at " << spin->ipos
                          << ": " << e.what() << std::endl;
                failed++;
            }
        }
    }
    link_spins(sc);
    return failed;
}

TEST(PyroConstruct, LookupConsistency) {
    int L = 10;
    auto Z = imat33_t::from_cols({L,0,0},{0,L,0},{0,0,L});
    auto sc = initialise_lattice(Z);
    EXPECT_EQ(test_lookup_consistency(sc), 0);
}
