#include "supercell.hpp"
#include "common.hpp"
#include "fourier.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <gtest/gtest.h>

static const int L = 8;

// Uniform field: only k=0 should be nonzero, with value N = L³
void test_uniform_simple(int l) {
    auto sc = build_simple_cubic(l);
    for (auto& s : sc.get_objects<Spin>()) s.Sz = 1.0;

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();

    auto& buf = ft.get_buffer();
    const int N = l * l * l;
    const double tol = 1e-9 * N;

    EXPECT_TRUE(std::abs(buf[0][0] - std::complex<double>(N, 0)) < tol)
        << "uniform: k=0 should equal N=" << N << " but got " << buf[0][0];

    bool all_zero = true;
    for (int i = 1; i < N; ++i) {
        if (std::abs(buf[0][i]) > tol) { all_zero = false; break; }
    }
    EXPECT_TRUE(all_zero) << "uniform: all k!=0 should be zero";
}

// Uniform field: only k=0 should be nonzero; allows non-cubic primitive cells
void test_uniform_general(const imat33_t& Z, const imat33_t& A) {
    auto sc = build_cubic(Z, A);
    for (auto& s : sc.get_objects<Spin>()) s.Sz = 1.0;

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();

    auto& buf = ft.get_buffer();
    const int N = det(Z);
    const double tol = 1e-9 * N;

    EXPECT_TRUE(std::abs(buf[0][0] - std::complex<double>(N, 0)) < tol)
        << "uniform: k=0 should equal N=" << N << " but got " << buf[0][0];

    bool all_zero = true;
    for (int i = 1; i < N; ++i) {
        if (std::abs(buf[0][i]) > tol) { all_zero = false; break; }
    }
    EXPECT_TRUE(all_zero) << "general uniform Z=" << to_string(Z)
                          << " A=" << to_string(A) << ": all k!=0 should be zero";
}

// Cosine wave at Q: peaks at k=+Q and k=-Q with magnitude N/2, zero elsewhere
void test_plane_wave(int l, ivec3_t Q) {
    auto sc = build_simple_cubic(l);
    set_cosine_wave(sc, Q);

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();

    auto& buf = ft.get_buffer();
    auto D = sc.lattice.size();
    const int N = l * l * l;
    const double tol = 1e-9 * N;
    const double expected = N / 2.0;

    auto kidx = [&](ivec3_t K) -> int {
        for (int a = 0; a < 3; ++a)
            K[a] = ((K[a] % D[a]) + D[a]) % D[a];
        return static_cast<int>((K[0] * D[1] + K[1]) * D[2] + K[2]);
    };

    int k_pos = kidx(Q);
    int k_neg = kidx({-Q[0], -Q[1], -Q[2]});

    bool aliased = (k_pos == k_neg);
    double peak_expected = aliased ? static_cast<double>(N) : expected;

    EXPECT_TRUE(std::abs(std::abs(buf[0][k_pos]) - peak_expected) < tol)
        << "plane wave: peak at +Q wrong";
    if (!aliased){
        EXPECT_TRUE(std::abs(std::abs(buf[0][k_neg]) - peak_expected) < tol)
            << "plane wave: peak at -Q wrong";
    }

    bool all_zero = true;
    for (int i = 0; i < N; ++i) {
        if (i == k_pos || i == k_neg) continue;
        if (std::abs(buf[0][i]) > tol) { all_zero = false; break; }
    }
    EXPECT_TRUE(all_zero) << "plane wave: should be zero elsewhere";
}

// Two-sublattice smoke test: uniform Sz=1 → k=0 peaks in each sublattice
void test_two_sublattice(int l) {
    int axis = 0;
    auto a = imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8});
    a(axis, axis) *= 2;
    MyCell cell(a);
    cell.add<Spin>(Spin({0,0,0}));
    ivec3_t x1 = {0,0,0};
    x1(axis) += 8;
    cell.add<Spin>(Spin(x1));

    auto Z = imat33_t::from_cols({l,0,0}, {0,l,0}, {0,0,l});
    auto sc = build_supercell<Spin>(cell, Z);
    for (auto& s : sc.get_objects<Spin>()) s.Sz = 1.0;

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();

    auto& buf = ft.get_buffer();
    const int N = sc.lattice.num_primitive_cells();
    const double tol = 1e-9 * N;

    EXPECT_TRUE(std::abs(buf[0][0] - std::complex<double>(N, 0)) < tol)
        << "2-sl: sublattice 0 k=0 should equal N";
    EXPECT_TRUE(std::abs(buf[1][0] - std::complex<double>(N, 0)) < tol)
        << "2-sl: sublattice 1 k=0 should equal N";
}

void test_backfolding(imat33_t Z, imat33_t W) {
    auto A = imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8});
    MyCell cell1(A);
    cell1.add<Spin>(Spin({0,0,0}));

    auto sc1 = build_supercell<Spin>(cell1, W * Z);
    const int np1 = sc1.lattice.num_primitive_cells();
    auto& spins1 = sc1.get_objects<Spin>();

    auto sc_tmp = build_supercell<Spin>(cell1, W);
    MyCell cell2(A * W);
    for (const auto& s : sc_tmp.get_objects<Spin>())
        cell2.add<Spin>(Spin(s.ipos));

    auto sc2 = build_supercell<Spin>(cell2, Z);
    const int np2 = sc2.lattice.num_primitive_cells();
    const auto D2 = sc2.lattice.size();
    auto& spins2 = sc2.get_objects<Spin>();
    const int num_sl = static_cast<int>(
        std::get<SlPos<Spin>>(sc2.sl_positions).size());

    if (spins1.size() != spins2.size())
        throw std::runtime_error("spin count mismatch between sc1 and sc2");

    for (int flat = 0; flat < np1; ++flat)
        spins1[flat].Sz = std::sin(2.0 * M_PI * flat / np1);

    for (int mu = 0; mu < num_sl; ++mu)
        for (int flat2 = 0; flat2 < np2; ++flat2) {
            const ipos_t pos = spins2[mu * np2 + flat2].ipos;
            spins2[mu * np2 + flat2].Sz = sc1.get_object_at<Spin>(pos)->Sz;
        }

    auto ft1 = make_fourier_transform<Spin, &Spin::Sz>(sc1);
    ft1.transform();
    auto ft2 = make_fourier_transform<Spin, &Spin::Sz>(sc2);
    ft2.transform();

    const auto& buf1 = ft1.get_buffer();
    const auto& buf2 = ft2.get_buffer();
    const auto& sl_pos2 = std::get<SlPos<Spin>>(sc2.sl_positions);

    const auto B1  = sc1.lattice.get_reciprocal_lattice_vectors();
    const auto M2d = dmat33_t::from_other(sc2.lattice.get_lattice_vectors());
    const double tol = 1e-9 * np1;
    bool ok = true;

    for (int f1 = 0; f1 < np1 && ok; ++f1) {
        const auto K  = sc1.lattice.idx3_from_flat(f1);
        const auto q  = B1 * vec3<double>(K);
        const auto K2f = (1.0 / (2.0 * M_PI)) * (M2d.tr() * q);
        ivec3_t Kf;
        for (int a = 0; a < 3; ++a) {
            auto k2a = static_cast<int64_t>(std::llround(K2f[a]));
            Kf[a] = ((k2a % D2[a]) + D2[a]) % D2[a];
        }
        const int f2 = sc2.lattice.flat_from_idx3(Kf);

        std::complex<double> expected = 0;
        for (int mu = 0; mu < num_sl; ++mu) {
            const double arg = -dot(q, vec3<double>(sl_pos2[mu]));
            expected += std::exp(std::complex<double>(0, arg)) * buf2[mu][f2];
        }

        if (std::abs(buf1[0][f1] - expected) > tol) {
            std::cerr << "  backfolding mismatch at K=(" << K[0] << ","
                      << K[1] << "," << K[2] << "): Y1=" << buf1[0][f1]
                      << " expected=" << expected << "\n";
            ok = false;
        }
    }
    EXPECT_TRUE(ok) << "backfolding: Y1[K] == sum_mu exp(-i q_K·r_mu) * Y2[mu, K mod D2]\n"
                    << "sc1.lattice_vectors = " << to_string(sc1.lattice.get_lattice_vectors())
                    << "\nsc2.lattice_vectors = " << to_string(sc2.lattice.get_lattice_vectors());
}

void test_structure_factor(imat33_t Z, imat33_t W) {
    auto A = imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8});
    MyCell cell1(A);
    cell1.add<Spin>(Spin({0,0,0}));

    auto sc1 = build_supercell<Spin>(cell1, W * Z);
    const int np1 = sc1.lattice.num_primitive_cells();
    auto& spins1 = sc1.get_objects<Spin>();

    auto sc_tmp = build_supercell<Spin>(cell1, W);
    MyCell cell2(A * W);
    for (const auto& s : sc_tmp.get_objects<Spin>())
        cell2.add<Spin>(Spin(s.ipos));

    auto sc2 = build_supercell<Spin>(cell2, Z);
    const int np2 = sc2.lattice.num_primitive_cells();
    auto& spins2 = sc2.get_objects<Spin>();
    const int num_sl = static_cast<int>(
        std::get<SlPos<Spin>>(sc2.sl_positions).size());

    if (spins1.size() != spins2.size())
        throw std::runtime_error("spin count mismatch between sc1 and sc2");

    for (int flat = 0; flat < np1; ++flat)
        spins1[flat].Sz = std::sin(2.0 * M_PI * flat / np1) +
                          0.3 * std::cos(2.0 * M_PI * 3 * flat / np1);

    for (int mu = 0; mu < num_sl; ++mu)
        for (int flat2 = 0; flat2 < np2; ++flat2) {
            const ipos_t pos = spins2[mu * np2 + flat2].ipos;
            spins2[mu * np2 + flat2].Sz = sc1.get_object_at<Spin>(pos)->Sz;
        }

    auto ft1 = make_fourier_transform<Spin, &Spin::Sz>(sc1);
    ft1.transform();
    auto ft2 = make_fourier_transform<Spin, &Spin::Sz>(sc2);
    ft2.transform();

    const auto& buf1 = ft1.get_buffer();
    const auto& buf2 = ft2.get_buffer();

    const auto& sl_pos1 = std::get<SlPos<Spin>>(sc1.sl_positions);
    const auto& sl_pos2 = std::get<SlPos<Spin>>(sc2.sl_positions);

    auto ph1 = SublatWeightMatrix::phase_factors(sc1.lattice,
                   std::vector<ipos_t>(sl_pos1.begin(), sl_pos1.end()));
    auto ph2 = SublatWeightMatrix::phase_factors(sc2.lattice,
                   std::vector<ipos_t>(sl_pos2.begin(), sl_pos2.end()));

    auto S1v = ph1.contract(correlate<Spin>(buf1, buf1));
    auto S2v = ph2.contract(correlate<Spin>(buf2, buf2));

    const double tol = 1e-6 * np1 * np1;


    auto M1 = sc1.lattice.get_lattice_vectors();
    auto M2 = sc2.lattice.get_lattice_vectors();
    auto M2_inv_tr = unnormed_inverse(M2).tr();
    auto det_M2 = det(M2);

    bool ok_cross = true;
    bool ok_self  = true;

    for (int f2 = 0; f2 < np2 && (ok_cross || ok_self); ++f2) {
        const auto K2 = sc2.lattice.idx3_from_flat(f2);
        // Map the physical (centred) wavevector of K2 into sc1's index space.
        // Using raw K2 would be wrong: the same raw index in two lattices with
        // different D can correspond to different physical wavevectors once the
        // BZ-centring shift is applied.
        idx3_t K2c = K2;
        const auto D2 = sc2.lattice.size();
        for (int a = 0; a < 3; ++a)
            if (K2c[a] > static_cast<int64_t>(D2[a]) / 2)
                K2c[a] -= static_cast<int64_t>(D2[a]);
        
        auto K1 = M1.tr() * M2_inv_tr * K2c;
        for (int i = 0; i < 3; ++i) {
            assert(K1[i] % det_M2 == 0);
            K1[i] /= det_M2;
            K1[i] = mod<int64_t>(K1[i], sc1.lattice.size(i));
        }

        const int f1 = sc1.lattice.flat_from_idx3_wrapped(K1);

        if (std::abs(S1v[f1] - S2v[f2]) > tol && !sc2.lattice.is_Nyquist_aliased(f2)) {
            const auto K1=sc1.lattice.idx3_from_flat(f1);
            std::cerr << "  structure factor mismatch at K1="<<K1<<" K2="<<K2<<": S1="
                      << S1v[f1] << " S2=" << S2v[f2] << "\n";
            std::cerr<<"  D1="<<sc1.lattice.size()<<" D2="<<sc2.lattice.size()<<"\n";
            ok_cross = false;
        }

        const double mag2 = std::norm(buf1[0][f1]);
        if (std::abs(S1v[f1] - mag2) > tol) {
            std::cerr << "  self-product mismatch at K=("
                      << K2[0] << "," << K2[1] << "," << K2[2] << "): S1="
                      << S1v[f1] << " |A|²=" << mag2 << "\n";
            ok_self = false;
        }
    }

    EXPECT_TRUE(ok_cross) << "structure factor: S1(K) == S2(K) over entire backfolded BZ";
    EXPECT_TRUE(ok_self)  << "structure factor: primitive S(K) == |A(K)|²";
}

// ---------------------------------------------------------------------------

TEST(CubicFourier, UniformSimple)         { test_uniform_simple(L); }

TEST(CubicFourier, PlaneWave_1_0_0)       { test_plane_wave(L, {1, 0, 0}); }
TEST(CubicFourier, PlaneWave_0_2_0)       { test_plane_wave(L, {0, 2, 0}); }
TEST(CubicFourier, PlaneWave_1_1_1)       { test_plane_wave(L, {1, 1, 1}); }
TEST(CubicFourier, PlaneWave_Nyquist)     { test_plane_wave(L, {L / 2, 0, 0}); }

TEST(CubicFourier, TwoSublattice)         { test_two_sublattice(L); }

TEST(CubicFourier, UniformGeneral_ShearA) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    test_uniform_general(Z, imat33_t::from_cols({1,-1,0}, {0,1,0}, {0,0,1}));
}
TEST(CubicFourier, UniformGeneral_ShearB) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    test_uniform_general(Z, imat33_t::from_cols({1,0,0}, {-1,1,0}, {0,0,1}));
}

TEST(CubicFourier, Backfolding_2x1x1) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    test_backfolding(Z, imat33_t::from_cols({2,0,0}, {0,1,0}, {0,0,1}));
}
TEST(CubicFourier, Backfolding_2x2x1) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    test_backfolding(Z, imat33_t::from_cols({2,0,0}, {0,2,0}, {0,0,1}));
}
TEST(CubicFourier, Backfolding_ShearRow) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    test_backfolding(Z, imat33_t::from_cols({1,-1,0}, {0,1,0}, {0,0,1}));
}
TEST(CubicFourier, Backfolding_ShearCol) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    test_backfolding(Z, imat33_t::from_cols({1,0,0}, {-1,1,0}, {0,0,1}));
}

TEST(CubicFourier, StructureFactor_2x1x1) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    test_structure_factor(Z, imat33_t::from_cols({2,0,0}, {0,1,0}, {0,0,1}));
}
TEST(CubicFourier, StructureFactor_2x2x1) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    test_structure_factor(Z, imat33_t::from_cols({2,0,0}, {0,2,0}, {0,0,1}));
}
