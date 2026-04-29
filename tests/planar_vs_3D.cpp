#include "supercell.hpp"
#include "common.hpp"
#include "fourier.hpp"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

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

    int flat = 0;
    for (auto& s : sc.get_objects<Spin>())
        s.Sz = std::sin(2.0 * M_PI * (flat++) / (num_sl * np));

    auto ft3d = make_fourier_transform<Spin, &Spin::Sz>(sc);
    ft3d.transform();
    const auto& buf3d = ft3d.get_buffer();

    KPlaneSpec spec(e1, e2, sc.lattice.size());
    auto ft2d = make_planar_fourier_transform<Spin, &Spin::Sz>(sc, spec);
    ft2d.transform();
    const auto& buf2d = ft2d.get_buffer();

    const int N1 = spec.N1, N2 = spec.N2;
    const double tol = 1e-8 * np;

    unsigned ok_vals = 0;

    for (int n1 = 0; n1 < N1; ++n1) {
        for (int n2 = 0; n2 < N2; ++n2) {
            const int k2d = n1 * N2 + n2;
            idx3_t K3{e1[0]*n1 + e2[0]*n2, e1[1]*n1 + e2[1]*n2, e1[2]*n1 + e2[2]*n2};
            const auto D = sc.lattice.size();
            for (int j = 0; j < 3; ++j)
                K3[j] = ((K3[j] % static_cast<int>(D[j])) + static_cast<int>(D[j]))
                         % static_cast<int>(D[j]);
            const int k3d = sc.lattice.flat_from_idx3(K3);

            for (int sl = 0; sl < num_sl; ++sl) {
                if (std::abs(buf2d[sl][k2d] - buf3d[sl][k3d]) > tol) {
                    std::cerr << "  planar/3d mismatch at sl=" << sl
                              << " n1=" << n1 << " n2=" << n2
                              << ": 2d=" << buf2d[sl][k2d]
                              << " 3d=" << buf3d[sl][k3d] << "\n";
                } else ok_vals++;
            }
        }
    }
    EXPECT_EQ(ok_vals, N1*N2*num_sl) << "planar FFT: buf2d[sl][n1,n2] == buf3d[sl][K(n1,n2)]";

    const auto& sl_pos = std::get<SlPos<Spin>>(sc.sl_positions);
    std::vector<ipos_t> sl_vec(sl_pos.begin(), sl_pos.end());

    auto pw2d = ft2d.make_phase_weights(sl_vec);
    auto pw3d_full = SublatWeightMatrix::phase_factors(sc.lattice, sl_vec);

    unsigned ok_phase = 0;
    for (int n1 = 0; n1 < N1; ++n1) {
        for (int n2 = 0; n2 < N2; ++n2) {
            const int k2d = n1 * N2 + n2;
            idx3_t K3{e1[0]*n1 + e2[0]*n2, e1[1]*n1 + e2[1]*n2, e1[2]*n1 + e2[2]*n2};
            const int k3d = sc.lattice.flat_from_idx3_wrapped(K3);

            for (int mu = 0; mu < num_sl; ++mu)
            for (int nu = 0; nu < num_sl; ++nu) {
                if (std::abs(pw2d(mu,nu)[k2d] - pw3d_full(mu,nu)[k3d]) > 1e-12) {
                    std::cerr << "  phase weight mismatch at mu=" << mu
                              << " nu=" << nu << " n1=" << n1 << " n2=" << n2 << "\n";
                } else {
                    ok_phase++;
                }
            }
        }
    }
    EXPECT_EQ(ok_phase, num_sl*num_sl*N1*N2) << "make_phase_weights: matches SublatWeightMatrix::phase_factors on plane";
}

static const auto Z4 = imat33_t::from_cols({4,0,0}, {0,4,0}, {0,0,4});
static const auto Z8 = imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8});
static const auto Z9 = imat33_t::from_cols({9,0,0}, {0,9,0}, {0,0,9});

TEST(PlanarVs3D, HHL) {
    test_planar_vs_3d(Z4, {1,1,0}, {0,0,1});
    test_planar_vs_3d(Z8, {1,1,0}, {0,0,1});
    test_planar_vs_3d(Z9, {1,1,0}, {0,0,1});
}
TEST(PlanarVs3D, HK0) {
    test_planar_vs_3d(Z4, {1,0,0}, {0,1,0});
    test_planar_vs_3d(Z8, {1,0,0}, {0,1,0});
    test_planar_vs_3d(Z9, {1,0,0}, {0,1,0});
}
TEST(PlanarVs3D, H0L) {
    test_planar_vs_3d(Z4, {1,0,0}, {0,0,1});
    test_planar_vs_3d(Z8, {1,0,0}, {0,0,1});
    test_planar_vs_3d(Z9, {1,0,0}, {0,0,1});
}
