#include "supercell.hpp"
#include "common.hpp"
#include "fourier.hpp"
//#include <array>
#include <cmath>
//#include <complex>
#include <iostream>

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

// Verify the planar transform against the full 3D FFT.
//
// For each sublattice and every k-point on the plane K = n1*e1 + n2*e2,
// the planar buffer must equal the corresponding entry of the 3D buffer.
// Also checks make_phase_weights() consistency with SublatWeightMatrix::phase_factors()
// on the same plane.
void test_planar_vs_3d(const imat33_t& Z, ivec3_t e1, ivec3_t e2) {
    auto sc = build_supercell<Spin>(build_pyro_cell(), Z);
    const int num_sl = static_cast<int>(
        std::get<SlPos<Spin>>(sc.sl_positions).size());
    const int np = sc.lattice.num_primitive_cells();

    // Non-trivial pattern
    int flat = 0;
    for (auto& s : sc.get_objects<Spin>())
        s.Sz = std::sin(2.0 * M_PI * (flat++) / (num_sl * np));

    // Full 3D transform
    auto ft3d = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft3d.transform();
    const auto& buf3d = ft3d.get_buffer();

    // Planar transform
    KPlaneSpec spec(e1, e2, sc.lattice.size());
    auto ft2d = make_planar_fourier_transform<Spin, &Spin::Sz>(sc, spec);
    ft2d.transform();
    const auto& buf2d = ft2d.get_buffer();

    const int N1 = spec.N1, N2 = spec.N2;
    const double tol = 1e-8 * np;
    bool ok_vals = true;

    // Check every (n1,n2) on the plane
    for (int n1 = 0; n1 < N1 && ok_vals; ++n1) {
        for (int n2 = 0; n2 < N2 && ok_vals; ++n2) {
            const int k2d = n1 * N2 + n2;
            idx3_t K3{e1[0]*n1 + e2[0]*n2, e1[1]*n1 + e2[1]*n2, e1[2]*n1 + e2[2]*n2};
            // Wrap into the 3D BZ
            const auto D = sc.lattice.size();
            for (int j = 0; j < 3; ++j)
                K3[j] = ((K3[j] % static_cast<int>(D[j])) + static_cast<int>(D[j]))
                         % static_cast<int>(D[j]);
            const int k3d = sc.lattice.flat_from_idx3(K3);

            for (int sl = 0; sl < num_sl && ok_vals; ++sl) {
                if (std::abs(buf2d[sl][k2d] - buf3d[sl][k3d]) > tol) {
                    std::cerr << "  planar/3d mismatch at sl=" << sl
                              << " n1=" << n1 << " n2=" << n2
                              << ": 2d=" << buf2d[sl][k2d]
                              << " 3d=" << buf3d[sl][k3d] << "\n";
                    ok_vals = false;
                }
            }
        }
    }

    check(ok_vals, "planar FFT: buf2d[sl][n1,n2] == buf3d[sl][K(n1,n2)]");

    // Check that make_phase_weights() matches SublatWeightMatrix::phase_factors()
    // on the plane points
    const auto& sl_pos = std::get<SlPos<Spin>>(sc.sl_positions);
    std::vector<ipos_t> sl_vec(sl_pos.begin(), sl_pos.end());

    auto pw2d = ft2d.make_phase_weights(sl_vec);
    auto pw3d_full = SublatWeightMatrix::phase_factors(sc.lattice, sl_vec);

    bool ok_phase = true;
    for (int n1 = 0; n1 < N1 && ok_phase; ++n1) {
        for (int n2 = 0; n2 < N2 && ok_phase; ++n2) {
            const int k2d = n1 * N2 + n2;
            idx3_t K3{e1[0]*n1 + e2[0]*n2, e1[1]*n1 + e2[1]*n2, e1[2]*n1 + e2[2]*n2};
            const auto D = sc.lattice.size();
            for (int j = 0; j < 3; ++j)
                K3[j] = ((K3[j] % static_cast<int>(D[j])) + static_cast<int>(D[j]))
                         % static_cast<int>(D[j]);
            const int k3d = sc.lattice.flat_from_idx3(K3);

            for (int mu = 0; mu < num_sl && ok_phase; ++mu)
            for (int nu = 0; nu < num_sl && ok_phase; ++nu) {
                if (std::abs(pw2d(mu,nu)[k2d] - pw3d_full(mu,nu)[k3d]) > 1e-12) {
                    std::cerr << "  phase weight mismatch at mu=" << mu
                              << " nu=" << nu << " n1=" << n1 << " n2=" << n2 << "\n";
                    ok_phase = false;
                }
            }
        }
    }
    check(ok_phase, "make_phase_weights: matches SublatWeightMatrix::phase_factors on plane");
}


int main() {
    // Planar FFT tests
    auto Z4 = imat33_t::from_cols({4,0,0}, {0,4,0}, {0,0,4});
    test_planar_vs_3d(Z4, {1,1,0}, {0,0,1});  // h,h,l
    test_planar_vs_3d(Z4, {1,0,0}, {0,1,0});  // h,k,0
    test_planar_vs_3d(Z4, {1,0,0}, {0,0,1});  // h,0,l
}
