#include "common.hpp"
#include <gtest/gtest.h>

void test_recip_latvec(int Lx, int Ly, int Lz, double tol = 1e-10) {
    auto lat = build_cubic(imat33_t::from_cols(
                {Lx, 0, 0}, {0, Ly, 0}, {0,0,Lz}));

    auto B = lat.lattice.get_reciprocal_lattice_vectors();
    auto D = lat.lattice.size();

    bool ok = true;

    for (auto Q : lat.lattice.enumerate_cell_index()){

        vec3d q1 = 2.0 * M_PI * vec3d{
            static_cast<double>(Q[0])/8 / D[0],
            static_cast<double>(Q[1])/8 / D[1],
            static_cast<double>(Q[2])/8 / D[2]
        };

        vec3d q2 =  B * Q;
        if( dot(q1 - q2, q1-q2) > tol){
            std::cerr<<"Mismatch: q1="<<q1<<" q2="<<q2<<std::endl;
            ok=false;
        }
    }

    EXPECT_TRUE(ok) << "Reciprocal lattice vector mismatch for Lx=" << Lx
                    << " Ly=" << Ly << " Lz=" << Lz;
}

TEST(BasicFourier, Orthogonal_8_8_8) { test_recip_latvec(8, 8, 8); }
TEST(BasicFourier, Orthogonal_2_3_4) { test_recip_latvec(2, 3, 4); }
