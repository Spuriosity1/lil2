#include "supercell.hpp"
#include "common.hpp"
#include "fourier.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <gtest/gtest.h>

// Verify correlate(buf, buf)(mu, nu)[k] == conj(buf[mu][k]) * buf[nu][k]
// for every mu, nu, k.
void test_correlate_elementwise() {
    auto Z = imat33_t::from_cols({4,0,0}, {0,4,0}, {0,0,4});
    auto sc = build_supercell<Spin>(build_pyro_cell(), Z);

    int flat = 0;
    for (auto& s : sc.get_objects<Spin>())
        s.Sz = std::sin(2.0 * M_PI * (flat++) / sc.get_objects<Spin>().size());

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();
    const auto& buf = ft.get_buffer();

    auto corr = correlate<Spin>(buf, buf);
    const int num_sl = buf.num_sublattices;
    const int k_size = buf.k_dims[0] * buf.k_dims[1] * buf.k_dims[2];
    const double tol = 1e-12;
    bool ok = true;

    for (int mu = 0; mu < num_sl && ok; ++mu)
        for (int nu = 0; nu < num_sl && ok; ++nu)
            for (int k = 0; k < k_size && ok; ++k) {
                std::complex<double> expected = std::conj(buf[mu][k]) * buf[nu][k];
                if (std::abs(corr(mu, nu)[k] - expected) > tol) {
                    std::cerr << "  correlate mismatch at mu=" << mu
                              << " nu=" << nu << " k=" << k << "\n";
                    ok = false;
                }
            }
    EXPECT_TRUE(ok) << "correlate: result(mu,nu)[k] == conj(buf[mu][k]) * buf[nu][k]";
}

// Verify that phase_factors weights satisfy Hermitian symmetry w(mu,nu)[k] == conj(w(nu,mu)[k]).
void test_phase_factors_selfadjoint() {
    auto Z = imat33_t::from_cols({4,0,0}, {0,4,0}, {0,0,4});
    auto sc = build_supercell<Spin>(build_pyro_cell(), Z);
    const auto& sl_pos = std::get<SlPos<Spin>>(sc.sl_positions);
    std::vector<ipos_t> sl_vec(sl_pos.begin(), sl_pos.end());

    auto pw = SublatWeightMatrix::phase_factors(sc.lattice, sl_vec);
    const int num_sl = pw.num_sublattices;
    const int k_size = pw.k_dims[0] * pw.k_dims[1] * pw.k_dims[2];
    const double tol = 1e-12;
    bool ok = true;

    for (int mu = 0; mu < num_sl && ok; ++mu)
        for (int nu = 0; nu < num_sl && ok; ++nu)
            for (int k = 0; k < k_size && ok; ++k) {
                if (std::abs(pw(mu, nu)[k] - std::conj(pw(nu, mu)[k])) > tol) {
                    std::cerr << "  Hermitian violation at mu=" << mu
                              << " nu=" << nu << " k=" << k << "\n";
                    ok = false;
                }
            }
    EXPECT_TRUE(ok) << "phase_factors: w(mu,nu)[k] == conj(w(nu,mu)[k])";
}

void test_structure_factor(imat33_t Z, imat33_t W) {
    std::cerr<<"test_structure_factor Z="<<Z<<" W="<<W<<std::endl;
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
        throw std::runtime_error("spin count mismatch");

    for (int i = 0; i < np1; ++i)
        spins1[i].Sz = std::sin(2.0 * M_PI * i / np1)
                     + 0.3 * std::cos(2.0 * M_PI * 3 * i / np1);

    for (int mu = 0; mu < num_sl; ++mu)
        for (int i = 0; i < np2; ++i) {
            const ipos_t pos = spins2[mu * np2 + i].ipos;
            spins2[mu * np2 + i].Sz = sc1.get_object_at<Spin>(pos)->Sz;
            assert(sc2.lattice.is_same_pos(spins2[mu*np2 + i].ipos, pos));
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

    auto S1 = ph1.contract(correlate<Spin>(buf1, buf1));
    auto S2 = ph2.contract(correlate<Spin>(buf2, buf2));

    const auto M1d = dmat33_t::from_other(sc1.lattice.get_lattice_vectors());
    const auto B2  = sc2.lattice.get_reciprocal_lattice_vectors();
    const auto T12 = (1.0 / (2.0 * M_PI)) * M1d.tr() * B2;
    const auto D1  = sc1.lattice.size();

    const double tol = 1e-6 * np1 * np1;
    bool ok_cross = true;
    bool ok_self  = true;

    const auto D2  = sc2.lattice.size();
    for (int f2 = 0; f2 < np2; ++f2) {
        const auto K2  = sc2.lattice.idx3_from_flat(f2);
        // Centre K2 in sc2's BZ before mapping so that T12 maps the
        // physical (centred) wavevector into sc1's index space, matching
        // the convention used by phase_factors on both sides.
        idx3_t K2c = K2;
        for (int j = 0; j < 3; ++j)
            if (K2c[j] > static_cast<int64_t>(D2[j]) / 2)
                K2c[j] -= static_cast<int64_t>(D2[j]);
        const auto K1_d = T12 * vec3<double>(K2c);
        ivec3_t K1;
        for (int j = 0; j < 3; ++j)
            K1[j] = ((int64_t)std::llround(K1_d[j]) % D1[j] + D1[j]) % D1[j];
        const int f1 = sc1.lattice.flat_from_idx3(K1);

        if (std::abs(S1[f1] - S2[f2]) > tol) {
            std::cerr << "  structure factor mismatch at K=("
                      << K2[0] << "," << K2[1] << "," << K2[2]
                      << "): S1=" << S1[f1] << " S2=" << S2[f2] << "\n";
            ok_cross = false;
        }

        const double mag2 = std::norm(buf1[0][f1]);
        if (std::abs(S1[f1] - mag2) > tol) {
            std::cerr << "  self-product mismatch at K=("
                      << K2[0] << "," << K2[1] << "," << K2[2]
                      << "): S1=" << S1[f1] << " |A|²=" << mag2 << "\n";
            ok_self = false;
        }
    }

    EXPECT_TRUE(ok_cross) << "new API: S1(K) == S2(K) over sc2 BZ";
    EXPECT_TRUE(ok_self)  << "new API: 1-sl S(K) == |A(K)|²";
}

// Verify SublatWeightMatrix::constant() with uniform unit weights matches Σ|Ã_mu|² directly.
void test_constant_weights_uniform() {
    auto Z = imat33_t::from_cols({4,0,0}, {0,4,0}, {0,0,4});
    auto sc = build_supercell<Spin>(build_pyro_cell(), Z);
    const int num_sl = static_cast<int>(
        std::get<SlPos<Spin>>(sc.sl_positions).size());
    const int np = sc.lattice.num_primitive_cells();

    int flat = 0;
    for (auto& s : sc.get_objects<Spin>())
        s.Sz = std::sin(2.0 * M_PI * (flat++) / (num_sl * np));

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();
    const auto& buf = ft.get_buffer();

    std::vector<std::vector<double>> w_unit(num_sl, std::vector<double>(num_sl, 1.0));
    auto cw = SublatWeightMatrix::constant(num_sl, buf.k_dims, w_unit);
    auto result = cw.contract(correlate<Spin>(buf, buf));

    const double tol = 1e-10;
    bool ok = true;
    for (int k = 0; k < np && ok; ++k) {
        std::complex<double> expected = 0;
        for (int mu = 0; mu < num_sl; ++mu)
            for (int nu = 0; nu < num_sl; ++nu)
                expected += std::conj(buf[mu][k]) * buf[nu][k];
        if (std::abs(result[k] - expected) > tol) {
            std::cerr << "  constant weight mismatch at k=" << k << "\n";
            ok = false;
        }
    }
    EXPECT_TRUE(ok) << "constant weights: unit matrix contracts to Σ_{μν} conj(Ã_μ)·Ã_ν";
}

static std::complex<double> form_factor(
        const LatticeIndexing& lat,
        const std::vector<ipos_t>& sl_pos,
        ivec3_t K)
{
    const auto B = lat.get_reciprocal_lattice_vectors();
    const auto q = B * vec3<double>(K);
    std::complex<double> F = 0;
    for (const auto& r : sl_pos)
        F += std::exp(std::complex<double>(0, dot(q, vec3<double>(r))));
    return F;
}

void test_structure_factor_real_and_inversion(int L) {
    auto sc = build_supercell<Spin>(build_pyro_cell(),
        imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L}));
    const int num_sl = (int)std::get<SlPos<Spin>>(sc.sl_positions).size();
    const int np     = sc.lattice.num_primitive_cells();
    auto& spins      = sc.get_objects<Spin>();

    for (int sl = 0; sl < num_sl; ++sl)
        for (int flat = 0; flat < np; ++flat) {
            auto I = sc.lattice.idx3_from_flat(flat);
            spins[sl * np + flat].Sz =
                std::sin(2.0*M_PI*(3*I[0] + I[1] + 2*I[2]) / L)
              + 0.4*std::cos(2.0*M_PI*(I[0] + 5*I[2]) / L);
        }

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();
    const auto& buf = ft.get_buffer();

    const auto& sl_pos = std::get<SlPos<Spin>>(sc.sl_positions);
    auto ph = SublatWeightMatrix::phase_factors(sc.lattice,
                  std::vector<ipos_t>(sl_pos.begin(), sl_pos.end()));
    auto Sv = ph.contract(correlate<Spin>(buf, buf));

    const double tol = 1e-8 * np * np;
    unsigned ok_real = np;
    unsigned ok_inv = np;

    for (int f = 0; f < np; ++f) {
        const auto K = sc.lattice.idx3_from_flat(f);
        if (std::abs(Sv[f].imag()) > tol) {
            std::cerr << "  S(k) not real at K=" << K
                      << ": Im=" << Sv[f].imag() << "\n";
            ok_real--;
        }
        const int fm = sc.lattice.flat_from_idx3_wrapped(K);
        if (std::abs(Sv[f] - Sv[fm]) > tol) {
            std::cerr << "  S(k)!=S(-k) at K=" << K
                      << ": S=" << Sv[f] << " S(-k)=" << Sv[fm] << "\n";
            ok_inv--;
        }
    }
    EXPECT_EQ(ok_real, (unsigned)np) << "S(k) real: Im(S(k)) == 0 for all k";
    EXPECT_EQ(ok_inv,  (unsigned)np) << "S(k) inversion: S(k) == S(-k) for all k";
}

void test_plane_wave_pyrochlore(int L, ivec3_t Q) {
    auto sc = build_supercell<Spin>(build_pyro_cell(),
        imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L}));
    const int num_sl = (int)std::get<SlPos<Spin>>(sc.sl_positions).size();
    const int np     = sc.lattice.num_primitive_cells();
    auto& spins      = sc.get_objects<Spin>();
    const auto D     = sc.lattice.size();

    for (int sl = 0; sl < num_sl; ++sl)
        for (int flat = 0; flat < np; ++flat) {
            auto I = sc.lattice.idx3_from_flat(flat);
            double ph = 2.0*M_PI*(
                static_cast<double>(Q[0])*I[0]/D[0] +
                static_cast<double>(Q[1])*I[1]/D[1] +
                static_cast<double>(Q[2])*I[2]/D[2]);
            spins[sl * np + flat].Sz = std::cos(ph);
        }

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();
    const auto& buf = ft.get_buffer();

    auto kidx = [&](ivec3_t K) -> int {
        for (int a = 0; a < 3; ++a)
            K[a] = ((K[a] % (int)D[a]) + (int)D[a]) % (int)D[a];
        return sc.lattice.flat_from_idx3(K);
    };
    const int k_pos  = kidx(Q);
    const int k_neg  = kidx({-Q[0], -Q[1], -Q[2]});
    const bool alias = (k_pos == k_neg);
    const double peak = alias ? (double)np : np / 2.0;
    const double tol_buf = 1e-9 * np;

    bool ok_peaks = true;
    for (int sl = 0; sl < num_sl && ok_peaks; ++sl) {
        if (std::abs(std::abs(buf[sl][k_pos]) - peak) > tol_buf) {
            std::cerr << "  DFT peak at +Q wrong: sl=" << sl
                      << " |buf|=" << std::abs(buf[sl][k_pos]) << " expected=" << peak << "\n";
            ok_peaks = false;
        }
        if (!alias && std::abs(std::abs(buf[sl][k_neg]) - peak) > tol_buf) {
            std::cerr << "  DFT peak at -Q wrong: sl=" << sl << "\n";
            ok_peaks = false;
        }
        for (int k = 0; k < np && ok_peaks; ++k) {
            if (k == k_pos || k == k_neg) continue;
            if (std::abs(buf[sl][k]) > tol_buf) {
                std::cerr << "  buf nonzero away from ±Q: sl=" << sl << " k=" << k << "\n";
                ok_peaks = false;
            }
        }
    }
    EXPECT_TRUE(ok_peaks) << "plane wave pyrochlore: DFT peaks at ±Q, zero elsewhere";

    const auto& sl_pos = std::get<SlPos<Spin>>(sc.sl_positions);
    std::vector<ipos_t> sl_vec(sl_pos.begin(), sl_pos.end());
    auto phw = SublatWeightMatrix::phase_factors(sc.lattice, sl_vec);
    auto Sv  = phw.contract(correlate<Spin>(buf, buf));

    const double tol_sf  = 1e-6 * np * np;
    const auto K_pos_3   = sc.lattice.idx3_from_flat(k_pos);
    const double S_expect = peak * peak * std::norm(form_factor(sc.lattice, sl_vec, K_pos_3));

    bool ok_val = std::abs(Sv[k_pos].real() - S_expect) < tol_sf;
    if (!ok_val)
        std::cerr << "  S(Q) got=" << Sv[k_pos].real() << " expected=" << S_expect << "\n";
    EXPECT_TRUE(ok_val) << "plane wave pyrochlore: S(Q) == peak^2 * |F(Q)|^2";

    bool ok_inv = std::abs(Sv[k_pos] - Sv[k_neg]) < tol_sf;
    if (!ok_inv)
        std::cerr << "  S(Q)=" << Sv[k_pos] << " S(-Q)=" << Sv[k_neg] << "\n";
    EXPECT_TRUE(ok_inv) << "plane wave pyrochlore: S(Q) == S(-Q)";
}

void test_sublattice_amplitude_plane_wave(int L, ivec3_t Q, std::array<double,4> A) {
    auto sc = build_supercell<Spin>(build_pyro_cell(),
        imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L}));
    const int num_sl = (int)std::get<SlPos<Spin>>(sc.sl_positions).size();
    const int np     = sc.lattice.num_primitive_cells();
    auto& spins      = sc.get_objects<Spin>();
    const auto D     = sc.lattice.size();

    for (int sl = 0; sl < num_sl; ++sl)
        for (int flat = 0; flat < np; ++flat) {
            auto I = sc.lattice.idx3_from_flat(flat);
            double ph = 2.0*M_PI*(
                static_cast<double>(Q[0])*I[0]/D[0] +
                static_cast<double>(Q[1])*I[1]/D[1] +
                static_cast<double>(Q[2])*I[2]/D[2]);
            spins[sl * np + flat].Sz = A[sl] * std::cos(ph);
        }

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();
    const auto& buf = ft.get_buffer();

    auto kidx = [&](ivec3_t K) -> int {
        for (int a = 0; a < 3; ++a)
            K[a] = ((K[a] % (int)D[a]) + (int)D[a]) % (int)D[a];
        return sc.lattice.flat_from_idx3(K);
    };
    const int k_pos   = kidx(Q);
    const bool alias  = (k_pos == kidx({-Q[0], -Q[1], -Q[2]}));
    const double hN   = alias ? (double)np : np / 2.0;

    const auto& sl_pos = std::get<SlPos<Spin>>(sc.sl_positions);
    std::vector<ipos_t> sl_vec(sl_pos.begin(), sl_pos.end());

    const auto B   = sc.lattice.get_reciprocal_lattice_vectors();
    const auto Kv  = sc.lattice.idx3_from_flat(k_pos);
    const auto q   = B * vec3<double>(Kv);
    std::complex<double> wF = 0;
    for (int mu = 0; mu < num_sl; ++mu)
        wF += A[mu] * std::exp(std::complex<double>(0,
                  dot(q, vec3<double>(sl_vec[mu]))));
    const double S_expect = hN * hN * std::norm(wF);

    auto phw = SublatWeightMatrix::phase_factors(sc.lattice, sl_vec);
    auto Sv  = phw.contract(correlate<Spin>(buf, buf));

    const double tol = 1e-6 * np * np;
    bool ok = std::abs(Sv[k_pos].real() - S_expect) < tol;
    if (!ok)
        std::cerr << "  sublat amp: S(Q)=" << Sv[k_pos].real()
                  << " expected=" << S_expect
                  << " A=" << A[0] << "," << A[1] << "," << A[2] << "," << A[3] << "\n";
    EXPECT_TRUE(ok) << "sublat amplitude plane wave: S(Q) == hN^2 * |Σ A_mu exp(+i q·r_mu)|^2";
}

void test_random_superposition_inversion(int L) {
    auto sc = build_supercell<Spin>(build_pyro_cell(),
        imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L}));
    const int num_sl = (int)std::get<SlPos<Spin>>(sc.sl_positions).size();
    const int np     = sc.lattice.num_primitive_cells();
    auto& spins      = sc.get_objects<Spin>();
    const auto D     = sc.lattice.size();

    const std::vector<ivec3_t> Qs = {{1,0,0}, {1,1,0}, {0,1,2}, {2,1,1}};
    const std::vector<std::array<double,4>> As = {
        { 1.0, -0.5,  0.7, -0.3},
        {-0.4,  1.2, -0.8,  0.6},
        { 0.9, -0.9,  0.4, -0.1},
        { 0.3,  0.7,  0.5, -0.9}
    };

    for (int sl = 0; sl < num_sl; ++sl)
        for (int flat = 0; flat < np; ++flat)
            spins[sl * np + flat].Sz = 0.0;

    for (int wi = 0; wi < (int)Qs.size(); ++wi)
        for (int sl = 0; sl < num_sl; ++sl)
            for (int flat = 0; flat < np; ++flat) {
                auto I = sc.lattice.idx3_from_flat(flat);
                double ph = 2.0*M_PI*(
                    static_cast<double>(Qs[wi][0])*I[0]/D[0] +
                    static_cast<double>(Qs[wi][1])*I[1]/D[1] +
                    static_cast<double>(Qs[wi][2])*I[2]/D[2]);
                spins[sl * np + flat].Sz += As[wi][sl] * std::cos(ph);
            }

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();
    const auto& buf = ft.get_buffer();

    const auto& sl_pos = std::get<SlPos<Spin>>(sc.sl_positions);
    auto phw = SublatWeightMatrix::phase_factors(sc.lattice,
                   std::vector<ipos_t>(sl_pos.begin(), sl_pos.end()));
    auto Sv  = phw.contract(correlate<Spin>(buf, buf));

    const double tol = 1e-8 * np * np;
    unsigned ok_inv = np;
    unsigned ok_pos = np;

    for (int f = 0; f < np; ++f) {
        const auto K  = sc.lattice.idx3_from_flat(f);
        const int  fm = sc.lattice.flat_from_idx3_wrapped(K);

        if (std::abs(Sv[f] - Sv[fm]) > tol) {
            std::cerr << "  superposition S(k)!=S(-k) at K=" << K
                      << ": " << Sv[f] << " vs " << Sv[fm] << "\n";
            ok_inv--;
        }
        if (Sv[f].real() < -tol) {
            std::cerr << "  superposition S(k)<0 at flat=" << f
                      << ": " << Sv[f].real() << "\n";
            ok_pos--;
        }
    }
    EXPECT_EQ(ok_inv, (unsigned)np) << "superposition: S(k) == S(-k) for all k";
    EXPECT_EQ(ok_pos, (unsigned)np) << "superposition: S(k) >= 0 for all k";
}

// ---------------------------------------------------------------------------

TEST(SublatCorrelator, CorrelateElementwise)     { test_correlate_elementwise(); }
TEST(SublatCorrelator, PhaseFactorsSelfAdjoint)  { test_phase_factors_selfadjoint(); }

TEST(SublatCorrelator, StructureFactor_diag_Z_W0) {
    auto Z = imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8});
    test_structure_factor(Z, imat33_t::from_cols({2,0,0}, {0,1,0}, {0,0,1}));
}
TEST(SublatCorrelator, StructureFactor_diag_Z_W1) {
    auto Z = imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8});
    test_structure_factor(Z, imat33_t::from_cols({2,0,0}, {0,2,0}, {0,0,1}));
}
TEST(SublatCorrelator, StructureFactor_weird_subcell_A) {
    auto Z = imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8});
    test_structure_factor(Z, imat33_t::from_cols({2,0,0}, {-1,1,-1}, {0,2,1}));
}
TEST(SublatCorrelator, StructureFactor_weird_subcell_B) {
    auto Z = imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8});
    test_structure_factor(Z, imat33_t::from_cols({-1,1,1}, {1,-1,1}, {1,1,-1}));
}
TEST(SublatCorrelator, StructureFactor_weird_supercell_A) {
    auto Z = imat33_t::from_cols({-2,-2,3}, {2,-2,0}, {1,2,3});
    test_structure_factor(Z, imat33_t::from_cols({2,0,0}, {0,1,0}, {0,0,1}));
}
TEST(SublatCorrelator, StructureFactor_weird_supercell_B) {
    auto Z = imat33_t::from_cols({-2,-2,3}, {2,-2,0}, {1,2,3});
    test_structure_factor(Z, imat33_t::from_cols({2,0,0}, {0,2,0}, {0,0,1}));
}
TEST(SublatCorrelator, ConstantWeightsUniform)        { test_constant_weights_uniform(); }
TEST(SublatCorrelator, StructureFactorRealAndInversion) { test_structure_factor_real_and_inversion(6); }

TEST(SublatCorrelator, PlaneWavePyrochlore_1_0_0) { test_plane_wave_pyrochlore(8, {1, 0, 0}); }
TEST(SublatCorrelator, PlaneWavePyrochlore_1_1_0) { test_plane_wave_pyrochlore(8, {1, 1, 0}); }
TEST(SublatCorrelator, PlaneWavePyrochlore_1_1_1) { test_plane_wave_pyrochlore(8, {1, 1, 1}); }
TEST(SublatCorrelator, PlaneWavePyrochlore_2_1_0) { test_plane_wave_pyrochlore(8, {2, 1, 0}); }

TEST(SublatCorrelator, SublatAmplitudePlaneWave_uniform) {
    test_sublattice_amplitude_plane_wave(8, {1, 0, 0}, {1.0,  1.0,  1.0,  1.0});
}
TEST(SublatCorrelator, SublatAmplitudePlaneWave_mixed) {
    test_sublattice_amplitude_plane_wave(8, {1, 0, 0}, {1.0, -1.0,  0.5, -0.5});
}
TEST(SublatCorrelator, SublatAmplitudePlaneWave_mixed_diag) {
    test_sublattice_amplitude_plane_wave(8, {1, 1, 1}, {1.0, -1.0,  0.5, -0.5});
}
TEST(SublatCorrelator, RandomSuperpositionInversion) { test_random_superposition_inversion(6); }
