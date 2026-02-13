
#include "supercell.hpp"
#include "unitcellspec.hpp"
#include "fourier.hpp"
#include <fstream>
#include <random>
#include <set>
#include <stack>
#include "pyro_data.hpp"



// forward declarations
struct Spin;

struct Spin {
    ipos_t ipos;
    std::vector<Spin*> neighbours;

    Spin() : ipos(0,0,0){ }
    Spin(const ipos_t& x) : ipos(x) {}
};


inline ipos_t floordiv(const ipos_t& x, int base){
    return ipos_t(x[0]/base, x[1]/base, x[2]/base);
}

void assert_position(void *ptr, ipos_t R) {
    if (ptr == nullptr) {
        std::cerr << "Could not resolve object at position " << R << "\n";
        throw std::runtime_error("Bad position");
    }
}

using MyCell = UnitCellSpecifier<Spin>;
using SuperLat = Supercell<Spin>;

// Generates a supercell specified by L
inline auto initialise_lattice(const imat33_t& Z )
{
    using namespace pyrochlore;

    MyCell cell(imat33_t::from_cols({0,4,4},{4,0,4},{4,4,0}));

    for (int mu=0; mu<4; mu++){
        cell.add<Spin>(Spin(pyro[mu]));
    }
    
    Supercell sc = build_supercell<Spin>(cell, Z);
    

    std::ofstream log("pyro_construct.log");
    for (const auto& [I, cell] : sc.enumerate_cells() ) {
        for (auto [spin_sl, spin] : cell.enumerate_objects<Spin>()){
            log << "Spin at "<<spin->ipos<<"\n";
        }
    }
    log.close();


    return sc;
}


void link_spins(Supercell<Spin>& sc){
    for (const auto& [I, cell] : sc.enumerate_cells() ) {
        for (auto [spin_sl, spin] : cell.enumerate_objects<Spin>()){
            // link up first neighbours
            for (const auto& dx : pyrochlore::nn_vectors[spin_sl]){
                std::cout << "Spin at "<<spin->ipos<<"\t| dx="<<dx<<"\n";
                auto r_nn = spin->ipos + dx;
                auto nn = sc.get_object_at<Spin>(r_nn);
                spin->neighbours.push_back(nn);
            }
        }
    }
}

int test_lookup_consistency(Supercell<Spin>& sc){

    int failed = 0;

    for (const auto& [I, cell] : sc.enumerate_cells() ) {
        for (auto [spin_sl, spin] : cell.enumerate_objects<Spin>()){
            std::cout << "Spin at "<<spin->ipos<<"\n";

            auto R_tmp = spin->ipos;
            auto R_tmp_2 = spin->ipos;
            sc.lattice.get_supercell_IDX(R_tmp);
            sc.lattice.wrap_primitive(R_tmp_2);
            assert(R_tmp == R_tmp_2);

            try {
            auto s = sc.get_object_at<Spin>(spin->ipos);
            if(s->ipos != spin->ipos){
                std::cerr<< "Lookup broken at "<<spin->ipos<<" (got "<<s->ipos<<")"<<std::endl;
                failed++;
            }
            } catch (ObjectLookupError& e){

                std::cerr<< "Lookup broken at "<<spin->ipos<<": "<<
                    e.what() <<std::endl;
                failed++;
            }
        }
    }

    link_spins(sc);

    return failed;
}

using namespace std;

int main (int argc, char *argv[]) {
    if (argc < 3){
        cout << "Usage: "<<argv[0]<<" L P"<<std::endl;
        return 1;
    }
    int L = atoi(argv[1]);
//    auto Z = imat33_t::from_cols({-L, L, L}, {L, -L, L}, {L, L, -L});
    auto Z = imat33_t::from_cols({L,0,0}, {0, L, 0}, {0, 0, L});

    SuperLat sc = initialise_lattice(Z);

    int failed = test_lookup_consistency(sc);

    link_spins(sc);

    return failed;
}

