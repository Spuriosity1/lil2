#include "supercell.hpp"
#include "common.hpp"
#include <cmath>
#include <complex>
#include <iostream>


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


static void test_backfold_multiplicity(const imat33_t& Z, const imat33_t& W){
    
    auto cell1 = build_pryo_primitive();
    const auto& A = pyro_primitive_cell;

    auto sc1 = build_supercell<Spin>(cell1, W * Z);

    // Build the W-fold unit cell: the det(W) sites of the W-supercell of
    // cell1 become the basis of cell2.  Primitive matrix of cell2 is A·W.
    auto sc_tmp = build_supercell<Spin>(cell1, W);
    MyCell cell2(A * W);
    for (const auto& s : sc_tmp.get_objects<Spin>())
        cell2.add<Spin>(Spin(s.ipos));

    // sc2: det(W)-sublattice, outer supercell Z
    auto sc2 = build_supercell<Spin>(cell2, Z);

    for (auto& s1 : sc1.get_objects<Spin>()){
        s1.Sz = 0;
    }
    for (auto& s2 : sc2.get_objects<Spin>()){
        s2.Sz = 0;
    }
    for (auto& s1 : sc1.get_objects<Spin>()){
        sc2.get_object_at<Spin>(s1.ipos)->Sz += 1;
    }
    for (auto& s2 : sc2.get_objects<Spin>()){
        sc1.get_object_at<Spin>(s2.ipos)->Sz += 1;
    }
    bool ok = true;

    for (auto& s1 : sc1.get_objects<Spin>()){
        if (s1.Sz != 1) {
            std::cout<<"Bad counting at "<<s1.ipos<<": count="<<
                s1.Sz<<std::endl;
            ok=false;
        }
    }
    for (auto& s2 : sc2.get_objects<Spin>()){
        if (s2.Sz != 1) {
            std::cout<<"Bad counting at "<<s2.ipos<<": count="<<
                s2.Sz<<std::endl;
            ok = false;
        }
    }
    check(ok, "Subllattice covering is even");
}

int main() {

    int L = 11;

    auto Z1 = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    auto Z2 = imat33_t::from_cols({L,1,0}, {0,L,0}, {0,0,L});
    auto Z3 = imat33_t::from_cols({-L,L,L}, {L,-L,L}, {L,L,-L});

    for (const auto& W : W_tests) {
        test_backfold_multiplicity(Z1, W);
        test_backfold_multiplicity(Z2, W);
        test_backfold_multiplicity(Z3, W);
    }
    
}
