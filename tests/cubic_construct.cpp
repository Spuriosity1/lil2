#include "supercell.hpp"
#include "unitcellspec.hpp"
#include <fstream>
#include <gtest/gtest.h>

struct Spin {
    ipos_t ipos;
    std::vector<Spin*> neighbours;

    Spin() : ipos(-1,-1,-1) {}
    Spin(const ipos_t& x) : ipos(x) {}
};

using MyCell = UnitCellSpecifier<Spin>;
using SuperLat = Supercell<Spin>;

inline auto initialise_lattice(const imat33_t& Z) {
    MyCell cell(imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8}));
    cell.add<Spin>(Spin({0,0,0}));

    Supercell sc = build_supercell<Spin>(cell, Z);

    std::ofstream log("cubic_construct.log");
    for (const auto& [I, cell_] : sc.enumerate_cells())
        for (auto [spin_sl, spin] : cell_.enumerate_objects<Spin>())
            log << "Spin at " << spin->ipos << "\n";

    return sc;
}

static std::vector<ipos_t> nn_vectors = { {8,0,0}, {0,8,0}, {0,0,8} };

static void link_spins(Supercell<Spin>& sc) {
    for (const auto& [I, cell_] : sc.enumerate_cells())
        for (auto [spin_sl, spin] : cell_.enumerate_objects<Spin>())
            for (const auto& dx : nn_vectors) {
                auto r_nn = spin->ipos + dx;
                auto nn = sc.get_object_at<Spin>(r_nn);
                spin->neighbours.push_back(nn);
            }
}

static int test_lookup_consistency(Supercell<Spin>& sc) {
    int failed = 0;
    for (const auto& [I, cell_] : sc.enumerate_cells())
        for (auto [spin_sl, spin] : cell_.enumerate_objects<Spin>()) {
            auto s = sc.get_object_at<Spin>(spin->ipos);
            if (s->ipos != spin->ipos) {
                std::cerr << "Lookup broken at " << s->ipos << std::endl;
                failed++;
            }
        }
    link_spins(sc);
    return failed;
}

TEST(CubicConstruct, LookupConsistency) {
    int L = 10;
    auto Z = imat33_t::from_cols({-L, L, L}, {L, -L, L}, {L, L, -L});
    auto sc = initialise_lattice(Z);
    EXPECT_EQ(test_lookup_consistency(sc), 0);
}
