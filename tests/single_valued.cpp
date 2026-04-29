#include "supercell.hpp"
#include "common.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <gtest/gtest.h>

static void test_backfold_multiplicity(const imat33_t& Z, const imat33_t& W) {
    auto cell1 = build_pryo_primitive();
    const auto& A = pyro_primitive_cell;

    auto sc1 = build_supercell<Spin>(cell1, W * Z);

    auto sc_tmp = build_supercell<Spin>(cell1, W);
    MyCell cell2(A * W);
    for (const auto& s : sc_tmp.get_objects<Spin>())
        cell2.add<Spin>(Spin(s.ipos));

    auto sc2 = build_supercell<Spin>(cell2, Z);

    for (auto& s1 : sc1.get_objects<Spin>()) s1.Sz = 0;
    for (auto& s2 : sc2.get_objects<Spin>()) s2.Sz = 0;

    for (auto& s1 : sc1.get_objects<Spin>())
        sc2.get_object_at<Spin>(s1.ipos)->Sz += 1;
    for (auto& s2 : sc2.get_objects<Spin>())
        sc1.get_object_at<Spin>(s2.ipos)->Sz += 1;

    bool ok = true;
    for (auto& s1 : sc1.get_objects<Spin>()) {
        if (s1.Sz != 1) {
            std::cout << "Bad counting at " << s1.ipos << ": count=" << s1.Sz << std::endl;
            ok = false;
        }
    }
    for (auto& s2 : sc2.get_objects<Spin>()) {
        if (s2.Sz != 1) {
            std::cout << "Bad counting at " << s2.ipos << ": count=" << s2.Sz << std::endl;
            ok = false;
        }
    }
    EXPECT_TRUE(ok) << "Sublattice covering is even";
}

// Z1 = diag(L,L,L)
TEST(SingleValued, Z1_W0) { auto Z = imat33_t::from_cols({11,0,0},{0,11,0},{0,0,11}); test_backfold_multiplicity(Z, W_tests[0]); }
TEST(SingleValued, Z1_W1) { auto Z = imat33_t::from_cols({11,0,0},{0,11,0},{0,0,11}); test_backfold_multiplicity(Z, W_tests[1]); }
TEST(SingleValued, Z1_W2) { auto Z = imat33_t::from_cols({11,0,0},{0,11,0},{0,0,11}); test_backfold_multiplicity(Z, W_tests[2]); }
TEST(SingleValued, Z1_W3) { auto Z = imat33_t::from_cols({11,0,0},{0,11,0},{0,0,11}); test_backfold_multiplicity(Z, W_tests[3]); }
TEST(SingleValued, Z1_W4) { auto Z = imat33_t::from_cols({11,0,0},{0,11,0},{0,0,11}); test_backfold_multiplicity(Z, W_tests[4]); }

// Z2 = shear
TEST(SingleValued, Z2_W0) { auto Z = imat33_t::from_cols({11,1,0},{0,11,0},{0,0,11}); test_backfold_multiplicity(Z, W_tests[0]); }
TEST(SingleValued, Z2_W1) { auto Z = imat33_t::from_cols({11,1,0},{0,11,0},{0,0,11}); test_backfold_multiplicity(Z, W_tests[1]); }
TEST(SingleValued, Z2_W2) { auto Z = imat33_t::from_cols({11,1,0},{0,11,0},{0,0,11}); test_backfold_multiplicity(Z, W_tests[2]); }
TEST(SingleValued, Z2_W3) { auto Z = imat33_t::from_cols({11,1,0},{0,11,0},{0,0,11}); test_backfold_multiplicity(Z, W_tests[3]); }
TEST(SingleValued, Z2_W4) { auto Z = imat33_t::from_cols({11,1,0},{0,11,0},{0,0,11}); test_backfold_multiplicity(Z, W_tests[4]); }

// Z3 = FCC-like
TEST(SingleValued, Z3_W0) { auto Z = imat33_t::from_cols({-11,11,11},{11,-11,11},{11,11,-11}); test_backfold_multiplicity(Z, W_tests[0]); }
TEST(SingleValued, Z3_W1) { auto Z = imat33_t::from_cols({-11,11,11},{11,-11,11},{11,11,-11}); test_backfold_multiplicity(Z, W_tests[1]); }
TEST(SingleValued, Z3_W2) { auto Z = imat33_t::from_cols({-11,11,11},{11,-11,11},{11,11,-11}); test_backfold_multiplicity(Z, W_tests[2]); }
TEST(SingleValued, Z3_W3) { auto Z = imat33_t::from_cols({-11,11,11},{11,-11,11},{11,11,-11}); test_backfold_multiplicity(Z, W_tests[3]); }
TEST(SingleValued, Z3_W4) { auto Z = imat33_t::from_cols({-11,11,11},{11,-11,11},{11,11,-11}); test_backfold_multiplicity(Z, W_tests[4]); }
