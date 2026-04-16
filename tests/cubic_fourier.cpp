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

using MyCell = UnitCellSpecifier<Spin>;
using SuperLat = Supercell<Spin>;

// Simple L×L×L cubic lattice, one spin per cell
inline auto build_simple_cubic(int L) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    MyCell cell(imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8}));
    cell.add<Spin>(Spin({0,0,0}));
    return build_supercell<Spin>(cell, Z);
}

// Set Sz[I] = cos(2π * dot(Q, I/D)) for each cell index I
inline void set_cosine_wave(SuperLat& lat, ivec3_t Q) {
    auto D = lat.lattice.size();
    int num_prim = lat.lattice.num_primitive_cells();
    auto& spins = lat.get_objects<Spin>();
    for (int flat = 0; flat < num_prim; ++flat) {
        auto I = lat.lattice.idx3_from_flat(flat);
        double phase = 2.0 * M_PI * (
            static_cast<double>(Q[0]) * I[0] / D[0] +
            static_cast<double>(Q[1]) * I[1] / D[1] +
            static_cast<double>(Q[2]) * I[2] / D[2]);
        spins[flat].Sz = std::cos(phase);
    }
}

// ---- minimal test harness --------------------------------------------------
static int g_failed = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        ++g_failed;
    } else {
        std::cout << "PASS: " << msg << "\n";
    }
}

// ---- tests -----------------------------------------------------------------

// Uniform field: only k=0 should be nonzero, with value N = L³
void test_uniform(int L) {
    auto sc = build_simple_cubic(L);
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
    auto sc = build_simple_cubic(L);
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
        if (std::abs(buf[0][i]) > tol) { all_zero = false; break; }
    }
    check(all_zero, "plane wave: zero elsewhere");
}

// Two-sublattice smoke test: uniform Sz=1 in each sublattice → k=0 peaks
void test_two_sublattice(int L) {
    int axis = 0;
    auto a = imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8});
    a(axis, axis) *= 2;
    MyCell cell(a);
    cell.add<Spin>(Spin({0,0,0}));
    ivec3_t x1 = {0,0,0};
    x1(axis) += 8;
    cell.add<Spin>(Spin(x1));

    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    auto sc = build_supercell<Spin>(cell, Z);
    for (auto& s : sc.get_objects<Spin>()) s.Sz = 1.0;

    auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft.transform();

    auto& buf = ft.get_buffer();
    const int N = sc.lattice.num_primitive_cells();
    const double tol = 1e-9 * N;

    check(std::abs(buf[0][0] - std::complex<double>(N, 0)) < tol,
          "2-sl: sublattice 0 k=0 == N");
    check(std::abs(buf[1][0] - std::complex<double>(N, 0)) < tol,
          "2-sl: sublattice 1 k=0 == N");
}

// Backfolding test: a 2-sublattice FFT is equivalent to a 1-sublattice FFT
// on the unfolded (doubled) lattice, related by:
//
//   Y1[K0, K1, K2] = Y2[sl0, K0%L, K1, K2]
//                  + exp(-iπ K0/L) · Y2[sl1, K0%L, K1, K2]
//
// for K0 ∈ [0, 2L).  The phase factor flips sign in the second half of the BZ,
// which is the hallmark of the zone-folding / band-backfolding identity.
//
// Layout used here (axis 0 is doubled):
//   1-sl: primitive cell diag(8,8,8), Z = diag(2L,L,L) → D = (2L,L,L)
//         cell (I0,I1,I2) sits at physical position (8·I0, 8·I1, 8·I2)
//   2-sl: primitive cell diag(16,8,8), Z = L·I → D = (L,L,L)
//         sl0 at (0,0,0), sl1 at (8,0,0) within the cell
//         cell (n0,n1,n2) → sl0 at (16·n0, 8·n1, 8·n2)
//                           sl1 at (16·n0+8, 8·n1, 8·n2)
//
// Mapping: 1-sl (2n0, n1, n2) ↔ 2-sl sl0 at (n0,n1,n2)
//          1-sl (2n0+1, n1, n2) ↔ 2-sl sl1 at (n0,n1,n2)
void test_backfolding(int L) {
    // --- one-sublattice: D = (2L, L, L) ---
    MyCell cell1(imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8}));
    cell1.add<Spin>(Spin({0,0,0}));
    auto Z1 = imat33_t::from_cols({2*L,0,0}, {0,L,0}, {0,0,L});
    auto sc1 = build_supercell<Spin>(cell1, Z1);
    const int np1 = sc1.lattice.num_primitive_cells();
    auto& spins1 = sc1.get_objects<Spin>();

    // --- two-sublattice: D = (L, L, L), cell doubled along axis 0 ---
    MyCell cell2(imat33_t::from_cols({16,0,0}, {0,8,0}, {0,0,8}));
    cell2.add<Spin>(Spin({0,0,0}));
    cell2.add<Spin>(Spin({8,0,0}));
    auto Z2 = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    auto sc2 = build_supercell<Spin>(cell2, Z2);
    const int np2 = sc2.lattice.num_primitive_cells();
    auto& spins2 = sc2.get_objects<Spin>();

    // Assign an arbitrary non-trivial pattern to the 1-sl lattice,
    // then copy the same physical values into the 2-sl lattice.
    for (int flat = 0; flat < np1; ++flat) {
        auto I = sc1.lattice.idx3_from_flat(flat);
        spins1[flat].Sz = std::sin(2.0*M_PI*(I[0]*1.1/(2*L) + I[1]*0.7/L + I[2]*0.4/L));
    }
    for (int flat2 = 0; flat2 < np2; ++flat2) {
        auto n = sc2.lattice.idx3_from_flat(flat2);
        // 1-sl cell (2·n0, n1, n2) → sublattice 0
        int f0 = sc1.lattice.flat_from_idx3({2*n[0],   n[1], n[2]});
        // 1-sl cell (2·n0+1, n1, n2) → sublattice 1
        int f1 = sc1.lattice.flat_from_idx3({2*n[0]+1, n[1], n[2]});
        spins2[0 * np2 + flat2].Sz = spins1[f0].Sz;
        spins2[1 * np2 + flat2].Sz = spins1[f1].Sz;
    }

    // Compute FFTs
    auto ft1 = make_fourier_transform<Spin, &Spin::Sz>(sc1);
    ft1.transform();
    auto ft2 = make_fourier_transform<Spin, &Spin::Sz>(sc2);
    ft2.transform();

    const auto& buf1 = ft1.get_buffer();
    const auto& buf2 = ft2.get_buffer();

    // Check: Y1[K0,K1,K2] == Y2[sl0, K0%L, K1, K2] + exp(-iπK0/L)*Y2[sl1, K0%L, K1, K2]
    // K0 runs over [0, 2L); K1, K2 over [0, L).
    const double tol = 1e-9 * np1;
    bool ok = true;
    for (int K0 = 0; K0 < 2*L && ok; ++K0) {
        const int k0f = K0 % L;
        const std::complex<double> phase{0.0, -M_PI * K0 / L};
        for (int K1 = 0; K1 < L && ok; ++K1) {
            for (int K2 = 0; K2 < L && ok; ++K2) {
                int f1 = sc1.lattice.flat_from_idx3({K0,  K1, K2});
                int f2 = sc2.lattice.flat_from_idx3({k0f, K1, K2});
                std::complex<double> expected = buf2[0][f2] + std::exp(phase) * buf2[1][f2];
                if (std::abs(buf1[0][f1] - expected) > tol) {
                    std::cerr << "  backfolding mismatch at K=(" << K0 << ","
                              << K1 << "," << K2 << "): Y1=" << buf1[0][f1]
                              << " expected=" << expected << "\n";
                    ok = false;
                }
            }
        }
    }
    check(ok, "backfolding: Y1[K0] == Y2[sl0,K0%L] + exp(-iπK0/L)*Y2[sl1,K0%L]");
}

// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    int L = 8;
    if (argc >= 2) L = atoi(argv[1]);

    test_uniform(L);
    test_plane_wave(L, {1, 0, 0});
    test_plane_wave(L, {0, 2, 0});
    test_plane_wave(L, {1, 1, 1});
    test_plane_wave(L, {L / 2, 0, 0});
    test_two_sublattice(L);
    test_backfolding(L);

    if (g_failed == 0)
        std::cout << "All tests passed.\n";
    else
        std::cerr << g_failed << " test(s) failed.\n";

    return g_failed > 0 ? 1 : 0;
}
