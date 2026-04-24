#include "common.hpp"


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

void test_recip_latvec(int Lx, int Ly, int Lz, double tol = 1e-10) {

    auto lat = build_cubic(imat33_t::from_cols(
                {Lx, 0, 0}, {0, Ly, 0}, {0,0,Lz}));

    auto B = lat.lattice.get_reciprocal_lattice_vectors();
    auto D = lat.lattice.size();

    bool ok = true;

    for (auto Q : lat.enumerate_cell_index()){

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

    check(ok, "Orthongonal supercell of cubic");
}

int main() {
    test_recip_latvec(8,8,8);
    test_recip_latvec(2,3,4);
    if (g_failed == 0)
        std::cout << "All tests passed.\n";
    else
        std::cerr << g_failed << " test(s) failed.\n";

    return g_failed > 0 ? 1 : 0;

}
