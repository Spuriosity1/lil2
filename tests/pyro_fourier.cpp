#include "supercell.hpp"
#include "common.hpp"
#include "fourier.hpp"
#include <cmath>
#include <complex>
#include <iostream>



// Simple L×L×L FCC lattice, four spins per cell
inline auto build_simple_FCC(int L) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    auto cell = build_pryo_primitive();
    return build_supercell<Spin>(cell, Z);
}


// ---- minimal test harness --------------------------------------------------
static int g_failed = 0;

static void check(bool cond, const std::string& test_name, const std::string& message_on_failure="") {
    if (!cond) {
        std::cerr << "FAIL: " << test_name << "\n";
        std::cerr << message_on_failure << "\n";
        ++g_failed;
    } else {
        std::cout << "PASS: " << test_name << "\n";
    }
}

// ---- tests -----------------------------------------------------------------

// Uniform field: only k=0 should be nonzero, with value N = L³
void test_uniform_simple(int L) {
    auto sc = build_simple_FCC(L);
    for (auto& s : sc.get_objects<Spin>()) s.Sz = 1.0;

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();

    auto& buf = ft.get_buffer();
    const int N = L * L * L;
    const double tol = 1e-9 * N;

    check(std::abs(buf[0][0] - std::complex<double>(N, 0)) < tol,
          "uniform: k=0 == N");

    bool all_zero = true;
    for (int i = 1; i < N; ++i) {
        if (std::abs(buf[0][i]) > tol) { all_zero = false; break; }
    }
    check(all_zero, "uniform: all k!=0 are zero");
}

// Cosine wave at Q: peaks at k=+Q and k=-Q with magnitude N/2, zero elsewhere
void test_plane_wave(int L, ivec3_t Q) {
    auto sc = build_simple_FCC(L);
    set_cosine_wave(sc, Q);

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();

    auto& buf = ft.get_buffer();
    auto D = sc.lattice.size();
    const int N = L * L * L;
    const double tol = 1e-9 * N;
    const double expected = N / 2.0;

    // Flat index for a given K, with periodic wrap
    auto kidx = [&](ivec3_t K) -> int {
        for (int a = 0; a < 3; ++a)
            K[a] = ((K[a] % D[a]) + D[a]) % D[a];
        return static_cast<int>((K[0] * D[1] + K[1]) * D[2] + K[2]);
    };

    int k_pos = kidx(Q);
    int k_neg = kidx({-Q[0], -Q[1], -Q[2]});

    // At Nyquist (+Q ≡ −Q mod D), the full power N lands at one point;
    // otherwise it splits equally between +Q and −Q.
    bool aliased = (k_pos == k_neg);
    double peak_expected = aliased ? static_cast<double>(N) : expected;

    check(std::abs(std::abs(buf[0][k_pos]) - peak_expected) < tol, "plane wave: peak at +Q");
    if (!aliased)
        check(std::abs(std::abs(buf[0][k_neg]) - peak_expected) < tol, "plane wave: peak at -Q");

    bool all_zero = true;
    for (int i = 0; i < N; ++i) {
        if (i == k_pos || i == k_neg) continue;
        if (std::abs(buf[0][i]) > tol) {
            all_zero = false;
            break;
        }
    }
    check(all_zero, "plane wave: zero elsewhere");
}


// Generalised backfolding test.
//
// Two equivalent representations of the same physical system:
//
//   sc1: 1-sublattice, primitive cell A, supercell Z·W
//        → det(Z·W) sites total
//
//   sc2: det(W)-sublattice, primitive cell A·W (the W-fold of A),
//        supercell Z
//        → det(W) sublattices × det(Z) cells = det(Z·W) sites total
//
// The backfolding identity is:
//
//   Y₁[K] = Σ_μ  exp(-i q_K · r_μ)  ·  Y₂[μ, K mod D₂]
//
// where:
//   q_K = B₁ · K   (physical k-vector; B₁ from get_reciprocal_lattice_vectors)
//   r_μ            (physical sublattice position stored in sc2.sl_positions)
//   K mod D₂       (component-wise; valid for diagonal Z, W with trivial SNF)
//
// Position-based copy via get_object_at makes the spin-transfer step
// independent of coordinate-system details.
void test_backfolding(imat33_t Z, imat33_t W) {
    auto cell1 = build_pryo_primitive();
    const auto& A = pyro_primitive_cell;

    // sc1: 1-sublattice, full supercell W * Z
    auto sc1 = build_supercell<Spin>(cell1, W * Z);
    const int np1 = sc1.lattice.num_primitive_cells();
    auto& spins1 = sc1.get_objects<Spin>();

    // Build the W-fold unit cell: the det(W) sites of the W-supercell of
    // cell1 become the basis of cell2.  Primitive matrix of cell2 is A·W.
    auto sc_tmp = build_supercell<Spin>(cell1, W);
    MyCell cell2(A * W);
    for (const auto& s : sc_tmp.get_objects<Spin>())
        cell2.add<Spin>(Spin(s.ipos));

    // sc2: det(W)-sublattice, outer supercell Z
    auto sc2 = build_supercell<Spin>(cell2, Z);
    const int np2 = sc2.lattice.num_primitive_cells();
    const auto D2 = sc2.lattice.size();
    auto& spins2 = sc2.get_objects<Spin>();
    const int num_sl = static_cast<int>(
        std::get<SlPos<Spin>>(sc2.sl_positions).size());

    if (spins1.size() != spins2.size())
        throw std::runtime_error("spin count mismatch between sc1 and sc2");

    // Set a non-trivial pattern on sc1
    for (int flat = 0; flat < np1; ++flat)
        spins1[flat].Sz = std::sin(2.0 * M_PI * flat / np1);

    // Copy to sc2 by physical position: each sc2 spin's ipos uniquely
    // identifies the same physical site in sc1.
    for (int mu = 0; mu < num_sl; ++mu)
        for (int flat2 = 0; flat2 < np2; ++flat2) {
            const ipos_t pos = spins2[mu * np2 + flat2].ipos;
            spins2[mu * np2 + flat2].Sz = sc1.get_object_at<Spin>(pos)->Sz;
        }

    // Compute FFTs
    auto ft1 = make_fourier_transform<Spin, &Spin::Sz>(sc1);
    ft1.transform();
    auto ft2 = make_fourier_transform<Spin, &Spin::Sz>(sc2);
    ft2.transform();

    const auto& buf1 = ft1.get_buffer();
    const auto& buf2 = ft2.get_buffer();
    const auto& sl_pos2 = std::get<SlPos<Spin>>(sc2.sl_positions);

    // B₁ · K gives the physical k-vector for FFT index K.
    // K₂ such that q₂(K₂) = q₁(K) is given by K₂ = M₂ᵀ · q / (2π),
    // where M₂ is sc2's index-cell-vectors.  This reduces to K mod D₂ only
    // when M₁ = M₂ (i.e. the SNF of the two supercells picks the same R).
    const auto B1  = sc1.lattice.get_reciprocal_lattice_vectors();
    const auto M2d = dmat33_t::from_other(sc2.lattice.get_lattice_vectors());
    const double tol = 1e-9 * np1;
    bool ok = true;

    for (int f1 = 0; f1 < np1 && ok; ++f1) {
        const auto K  = sc1.lattice.idx3_from_flat(f1);
        const auto q  = B1 * vec3<double>(K);
        // K₂ = M₂ᵀ · q / (2π) — matches physical wavevectors regardless of SNF choice
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
    check(ok,
          "backfolding: Y1[K] == sum_mu exp(-i q_K·r_mu) * Y2[mu, K mod D2]",
          std::string("sc1.lattice_vectors = ") +
              to_string(sc1.lattice.get_lattice_vectors()) +
              std::string("\nsc2.lattice_vectors = ") +
              to_string(sc2.lattice.get_lattice_vectors()));
}

// Structure factor test.
//
// Verifies that the `inner` function gives the same result regardless of how
// the lattice is tiled.  Uses the same two-representation setup as
// test_backfolding:
//
//   sc1: 1-sublattice, supercell W * Z   (det(W * Z) sites)
//   sc2: det(W)-sublattice, supercell Z (same sites, differently packaged)
//
// Both are loaded with the same non-trivial spin pattern via position-based
// lookup.  We then compute
//
//   S1(K) = inner(buf1, buf1, sc1.lattice, sl_pos1)[0][K]
//   S2(K) = inner(buf2, buf2, sc2.lattice, sl_pos2)[0][K]
//
// and assert S1(K) == S2(K) for every K in the smaller (sc2) BZ.
// Additionally we verify the single-sublattice sanity identity:
//   S1(K) == |buf1[0][K]|²
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

    // Non-trivial pattern on sc1
    for (int flat = 0; flat < np1; ++flat)
        spins1[flat].Sz = std::sin(2.0 * M_PI * flat / np1) +
                          0.3 * std::cos(2.0 * M_PI * 3 * flat / np1);

    // Copy to sc2 by physical position
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
    bool ok_cross = true;
    bool ok_self  = true;

    for (int f2 = 0; f2 < np2 && (ok_cross || ok_self); ++f2) {
        const auto K2 = sc2.lattice.idx3_from_flat(f2);
        // B1 = B2 (index cells are equal),
        // so q1(K) = q2(K) at the same integer index. The sc2 BZ is a subset of sc1's.
        const int f1 = sc1.lattice.flat_from_idx3(K2);

        // S1 vs S2 agreement
        if (std::abs(S1v[f1] - S2v[f2]) > tol) {
            std::cerr << "  structure factor mismatch at K=("
                      << K2[0] << "," << K2[1] << "," << K2[2] << "): S1="
                      << S1v[f1] << " S2=" << S2v[f2] << "\n";
            ok_cross = false;
        }

        // Single-sublattice sanity: S1(K) == |Ã(K)|² (phase trivially 1 for 1-sl)
        const double mag2 = std::norm(buf1[0][f1]);
        if (std::abs(S1v[f1] - mag2) > tol) {
            std::cerr << "  self-product mismatch at K=("
                      << K2[0] << "," << K2[1] << "," << K2[2] << "): S1="
                      << S1v[f1] << " |A|²=" << mag2 << "\n";
            ok_self = false;
        }
    }

    check(ok_cross, "structure factor: S1(K) == S2(K) over backfolded BZ");
    check(ok_self,  "structure factor: primitive S(K) == |A(K)|²");
}

// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    int L = 8;
    if (argc >= 2) L = atoi(argv[1]);

    test_uniform_simple(L);
    test_plane_wave(L, {1, 0, 0});
    test_plane_wave(L, {0, 2, 0});
    test_plane_wave(L, {1, 1, 1});
    test_plane_wave(L, {L / 2, 0, 0});

    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    // 2-cell: double along axis 0
    test_backfolding(Z, imat33_t::from_cols({2,0,0}, {0,1,0}, {0,0,1}));
    // 4-cell: double along axes 0 and 1
    test_backfolding(Z, imat33_t::from_cols({2,0,0}, {0,2,0}, {0,0,1}));
    // 2-cell non-orthogonal
    test_backfolding(Z, imat33_t::from_cols({1,-1,0}, {0,1,0}, {0,0,1}));

    // Structure factor tests
    test_structure_factor(Z, imat33_t::from_cols({2,0,0}, {0,1,0}, {0,0,1}));
    test_structure_factor(Z, imat33_t::from_cols({2,0,0}, {0,2,0}, {0,0,1}));

    if (g_failed == 0)
        std::cout << "All tests passed.\n";
    else
        std::cerr << g_failed << " test(s) failed.\n";

    return g_failed > 0 ? 1 : 0;
}
