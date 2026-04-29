#include "supercell.hpp"
#include "common.hpp"
#include "fourier.hpp"
#include "modulus.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <gtest/gtest.h>

static const int L = 8;

inline imat33_t cube(int l) {
    return imat33_t::from_cols({l,0,0},{0,l,0},{0,0,l});
}

// Simple L×L×L FCC lattice, four spins per cell
inline auto build_simple_FCC(int l) {
    return build_supercell<Spin>(build_pryo_primitive(), cube(l));
}

// Uniform field: only k=0 should be nonzero, with value N = L³
void test_uniform_simple(int l) {
    auto sc = build_simple_FCC(l);
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

// Cosine wave at Q: peaks at k=+Q and k=-Q with magnitude N/2, zero elsewhere
void test_plane_wave(const imat33_t& Z, const ivec3_t& Q) {
    auto sc = build_supercell<Spin>(build_pryo_primitive(), Z);
    set_cosine_wave(sc, Q);

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();

    auto& buf = ft.get_buffer();
    const int N = sc.lattice.num_primitive_cells();
    const double tol = 1e-9 * N;
    const double expected = N / 2.0;

    int k_pos = sc.lattice.flat_from_idx3_wrapped(Q);
    int k_neg = sc.lattice.flat_from_idx3_wrapped(-Q);

    bool aliased = (k_pos == k_neg);
    double peak_expected = aliased ? static_cast<double>(N) : expected;

    EXPECT_TRUE(std::abs(std::abs(buf[0][k_pos]) - peak_expected) < tol)
        << "plane wave: peak at +Q wrong";
    if (!aliased)
        EXPECT_TRUE(std::abs(std::abs(buf[0][k_neg]) - peak_expected) < tol)
            << "plane wave: peak at -Q wrong";

    bool all_zero = true;
    for (int i = 0; i < N; ++i) {
        if (i == k_pos || i == k_neg) continue;
        if (std::abs(buf[0][i]) > tol) { all_zero = false; break; }
    }
    EXPECT_TRUE(all_zero) << "plane wave: should be zero elsewhere";
}

// Checks that the <SzSz> correlator matches expected analytic expression
void test_delta_fn(const imat33_t& Z, const imat33_t& W, const ivec3_t& I_p) {
    auto cell0 = build_pryo_primitive();
    auto sc0 = build_supercell<Spin>(cell0, W);
    MyCell cell1(pyro_primitive_cell * W);
    for (const auto& s : sc0.get_objects<Spin>())
        cell1.add(Spin{s.ipos});

    auto sc = build_supercell<Spin>(cell1, Z);

    double atol = 1e-10;

    const int n_sublat = sc.num_sl<Spin>();
    std::vector<double> a = {0.2, 0.7, -3, 9};
    for (int i=4; i<n_sublat; i++) a.push_back(0);

    for (const auto& [I, c] : sc.enumerate_cells()){
        for (auto [mu, s] : c.enumerate_objects<Spin>()) {
            if (I == I_p)
                s->Sz = a[mu];
            else
                s->Sz = 0;
        }
    }

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();

    const auto& Sz_raw = ft.get_buffer();
    auto tI_p = vec3<double>(sc.lattice.translation_of(I_p));

    unsigned failed_lin = 0;
    for (const auto& Q : sc.lattice.enumerate_cell_index()) {
        auto flat = sc.lattice.flat_from_idx3(Q);
        auto q = sc.lattice.wavevector_from_idx3(Q);

        for (int mu=0; mu<n_sublat; mu++){
            auto got_ft = Sz_raw[mu][flat];
            auto expected = a[mu] * std::polar<double>(1.0, -dot(q, tI_p));
            double err = std::abs(got_ft - expected);
            if (err > atol) {
                failed_lin++;
                std::cerr << "[ft] Disagreement (got "<<got_ft<<" expect "<< expected
                    <<")at mu = "<<mu<<", Q="<<Q<<std::endl;
            }
        }
    }
    EXPECT_EQ(failed_lin, 0u) << "delta fn transform as expected";

    auto Sraw_Sraw_corr = correlate(Sz_raw, Sz_raw);

    unsigned failed_corr = 0;
    for (int mu=0; mu<4; mu++){
        for (int nu=0; nu<4; nu++){
            const auto& Smn = Sraw_Sraw_corr(mu, nu);
            for (auto Q : sc.lattice.enumerate_cell_index()) {
                auto flat = sc.lattice.flat_from_idx3(Q);
                auto obtained = Smn[flat];
                auto expected = a[mu]*a[nu];

                double err = std::abs(obtained - expected);
                if (err > atol) {
                    failed_corr++;
                    std::cerr << "[corr] Disagreement (got "<<obtained<<" expect "<< expected
                        <<")at mu,nu = "<<mu<<", "<<nu<<" Q="<<Q<<std::endl;
                }
            }
        }
    }
    EXPECT_EQ(failed_corr, 0u) << "delta fn raw corr as expected";

    auto cw = SublatWeightMatrix::phase_factors(sc.lattice,
         std::get<SlPos<Spin>>(sc.sl_positions));

    auto SzSz = cw.contract(Sraw_Sraw_corr);
    const auto r_sl = std::get<SlPos<Spin>>(sc.sl_positions);

    unsigned failed_contracted = 0;
    for (auto Q : sc.lattice.enumerate_cell_index()) {
        auto flat = sc.lattice.flat_from_idx3(Q);
        auto q = sc.lattice.wavevector_from_idx3(Q);

        auto obtained = SzSz[flat];
        std::complex<double> w = 0;
        for (auto mu=0; mu<n_sublat; mu++)
            w += a[mu] * std::polar(1.0, -dot<double>(q, r_sl[mu]));

        auto expected = w * std::conj(w);
        double err = std::abs(obtained - expected);
        if (err > atol) {
            failed_contracted++;
            std::cerr << "[SzSz] Disagreement (got "<<obtained<<" expect "<< expected
                <<")at Q="<<Q<<std::endl;
        }
    }
    EXPECT_EQ(failed_contracted, 0u) << "delta fn contracted corr as expected";
}

void test_backfolding(imat33_t Z, imat33_t W) {
    auto cell1 = build_pryo_primitive();
    const auto& A = pyro_primitive_cell;

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

    const auto B1 = sc1.lattice.get_reciprocal_lattice_vectors();
    const auto M1 = sc1.lattice.get_lattice_vectors();
    const auto M2 = sc2.lattice.get_lattice_vectors();
    const auto M2d = dmat33_t::from_other(M2);
    const auto M1_inv_tr = unnormed_inverse(M1).tr();

    const double tol = 1e-9 * np1;
    bool ok = true;

    for (int f1 = 0; f1 < np1; ++f1) {
        const auto K1  = sc1.lattice.idx3_from_flat(f1);
        const auto q  = B1 * vec3<double>(K1);
        const auto K2f = (1.0 / (2.0 * M_PI)) * (M2d.tr() * q);
        ivec3_t K2;
        for (int a = 0; a < 3; ++a) {
            auto k2a = static_cast<int64_t>(std::llround(K2f[a]));
            K2[a] = mod(k2a, D2[a]);
        }

        auto K2_other = M2.tr() * M1_inv_tr * K1;
        for (int i = 0; i < 3; ++i) {
            assert(K2_other[i] % det(M1) == 0);
            K2_other[i] /= det(M1);
            K2_other[i] = mod<int64_t>(K2_other[i], sc2.lattice.size(i));
        }
        assert(K2 == K2_other);

        const int f2 = sc2.lattice.flat_from_idx3(K2);

        std::complex<double> expected = 0;
        for (int mu = 0; mu < num_sl; ++mu) {
            const double arg = -dot(q, vec3<double>(sl_pos2[mu]));
            expected += std::exp(std::complex<double>(0, arg)) * buf2[mu][f2];
        }

        if (std::abs(buf1[0][f1] - expected) > tol) {
            std::cerr << "  backfolding mismatch at K=(" << K1[0] << ","
                      << K1[1] << "," << K1[2] << "): Y1=" << buf1[0][f1]
                      << " expected=" << expected << "\n";
            ok = false;
        }
    }
    EXPECT_TRUE(ok)
        << "backfolding: Y1[K] == sum_mu exp(-i q_K·r_mu) * Y2[mu, K mod D2]\n"
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

    auto M1 = sc1.lattice.get_lattice_vectors();
    auto M2 = sc2.lattice.get_lattice_vectors();

    const double tol = 1e-6 * np1 * np1;
    bool ok_cross = true;

    auto M2_inv_tr = unnormed_inverse(M2).tr();
    auto det_M2 = det(M2);

    for (int f2 = 0; f2 < np2; ++f2) {
        const auto K2 = sc2.lattice.idx3_from_flat(f2);
        auto K1 = M1.tr() * M2_inv_tr * K2;
        for (int i = 0; i < 3; ++i) {
            assert(K1[i] % det_M2 == 0);
            K1[i] /= det_M2;
            K1[i] = mod<int64_t>(K1[i], sc1.lattice.size(i));
        }
        const int f1 = sc1.lattice.flat_from_idx3(K1);

        if (std::abs(S1v[f1] - S2v[f2]) > tol) {
            std::cerr << "  structure factor mismatch at K2=" << K2
                      << ", K1=" << K1
                      << ": S1=" << S1v[f1] << " S2=" << S2v[f2] << "\n";
            ok_cross = false;
        }
    }
    EXPECT_TRUE(ok_cross) << "structure factor: S1(K) == S2(K) over backfolded BZ";
}

// ---------------------------------------------------------------------------

TEST(PyroFourier, UniformSimple) { test_uniform_simple(L); }

TEST(PyroFourier, PlaneWave_1_0_0) { test_plane_wave(cube(L), {1, 0, 0}); }
TEST(PyroFourier, PlaneWave_0_2_0) { test_plane_wave(cube(L), {0, 2, 0}); }
TEST(PyroFourier, PlaneWave_1_1_1) { test_plane_wave(cube(L), {1, 1, 1}); }
TEST(PyroFourier, PlaneWave_Nyquist) { test_plane_wave(cube(L), {L / 2, 0, 0}); }

// W_tests[0]: diag(2,1,1)
TEST(PyroFourier, W0_PlaneWaves) {
    auto Z = cube(L); auto W = W_tests[0];
    test_plane_wave(W*Z, {1, 0, 0});
    test_plane_wave(W*Z, {0, 2, 0});
    test_plane_wave(W*Z, {1, 1, 1});
    test_plane_wave(W*Z, {1, -1, -1});
    test_plane_wave(W*Z, {L / 2, 0, 0});
}
TEST(PyroFourier, W0_Backfolding)     { test_backfolding(cube(L), W_tests[0]); }
TEST(PyroFourier, W0_StructureFactor) { test_structure_factor(cube(L), W_tests[0]); }
TEST(PyroFourier, W0_DeltaFn) {
    auto Z = cube(L); auto W = W_tests[0];
    test_delta_fn(Z, W, {0,0,0});
    test_delta_fn(Z, W, {1,0,0});
    test_delta_fn(Z, W, {2,0,0});
    test_delta_fn(Z, W, {1,1,0});
}

// W_tests[1]: diag(2,2,1)
TEST(PyroFourier, W1_PlaneWaves) {
    auto Z = cube(L); auto W = W_tests[1];
    test_plane_wave(W*Z, {1, 0, 0});
    test_plane_wave(W*Z, {0, 2, 0});
    test_plane_wave(W*Z, {1, 1, 1});
    test_plane_wave(W*Z, {1, -1, -1});
    test_plane_wave(W*Z, {L / 2, 0, 0});
}
TEST(PyroFourier, W1_Backfolding)     { test_backfolding(cube(L), W_tests[1]); }
TEST(PyroFourier, W1_StructureFactor) { test_structure_factor(cube(L), W_tests[1]); }
TEST(PyroFourier, W1_DeltaFn) {
    auto Z = cube(L); auto W = W_tests[1];
    test_delta_fn(Z, W, {0,0,0});
    test_delta_fn(Z, W, {1,0,0});
    test_delta_fn(Z, W, {2,0,0});
    test_delta_fn(Z, W, {1,1,0});
}

// W_tests[2]: shear row
TEST(PyroFourier, W2_PlaneWaves) {
    auto Z = cube(L); auto W = W_tests[2];
    test_plane_wave(W*Z, {1, 0, 0});
    test_plane_wave(W*Z, {0, 2, 0});
    test_plane_wave(W*Z, {1, 1, 1});
    test_plane_wave(W*Z, {1, -1, -1});
    test_plane_wave(W*Z, {L / 2, 0, 0});
}
TEST(PyroFourier, W2_Backfolding)     { test_backfolding(cube(L), W_tests[2]); }
TEST(PyroFourier, W2_StructureFactor) { test_structure_factor(cube(L), W_tests[2]); }
TEST(PyroFourier, W2_DeltaFn) {
    auto Z = cube(L); auto W = W_tests[2];
    test_delta_fn(Z, W, {0,0,0});
    test_delta_fn(Z, W, {1,0,0});
    test_delta_fn(Z, W, {2,0,0});
    test_delta_fn(Z, W, {1,1,0});
}

// W_tests[3]: shear col
TEST(PyroFourier, W3_PlaneWaves) {
    auto Z = cube(L); auto W = W_tests[3];
    test_plane_wave(W*Z, {1, 0, 0});
    test_plane_wave(W*Z, {0, 2, 0});
    test_plane_wave(W*Z, {1, 1, 1});
    test_plane_wave(W*Z, {1, -1, -1});
    test_plane_wave(W*Z, {L / 2, 0, 0});
}
TEST(PyroFourier, W3_Backfolding)     { test_backfolding(cube(L), W_tests[3]); }
TEST(PyroFourier, W3_StructureFactor) { test_structure_factor(cube(L), W_tests[3]); }
TEST(PyroFourier, W3_DeltaFn) {
    auto Z = cube(L); auto W = W_tests[3];
    test_delta_fn(Z, W, {0,0,0});
    test_delta_fn(Z, W, {1,0,0});
    test_delta_fn(Z, W, {2,0,0});
    test_delta_fn(Z, W, {1,1,0});
}

// W_tests[4]: FCC-like
TEST(PyroFourier, W4_PlaneWaves) {
    auto Z = cube(L); auto W = W_tests[4];
    test_plane_wave(W*Z, {1, 0, 0});
    test_plane_wave(W*Z, {0, 2, 0});
    test_plane_wave(W*Z, {1, 1, 1});
    test_plane_wave(W*Z, {1, -1, -1});
    test_plane_wave(W*Z, {L / 2, 0, 0});
}
TEST(PyroFourier, W4_Backfolding)     { test_backfolding(cube(L), W_tests[4]); }
TEST(PyroFourier, W4_StructureFactor) { test_structure_factor(cube(L), W_tests[4]); }
TEST(PyroFourier, W4_DeltaFn) {
    auto Z = cube(L); auto W = W_tests[4];
    test_delta_fn(Z, W, {0,0,0});
    test_delta_fn(Z, W, {1,0,0});
    test_delta_fn(Z, W, {2,0,0});
    test_delta_fn(Z, W, {1,1,0});
}
