
#include "supercell.hpp"
#include "unitcellspec.hpp"
#include <fstream>




// forward declarations
struct Spin;

struct Spin {
    ipos_t ipos;
    std::vector<Spin*> neighbours;

    Spin() : ipos(-1,-1,-1){ }
    Spin(const ipos_t& x) : ipos(x) {}
};



using MyCell = UnitCellSpecifier<Spin>;
using SuperLat = Supercell<Spin>;

// Generates a supercell specified by L
inline auto initialise_lattice(const imat33_t& Z )
{

    MyCell cell(imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8}));

    cell.add<Spin>(Spin({0,0,0}));
    
    
    Supercell sc = build_supercell<Spin>(cell, Z);
    
    std::ofstream log("cubic_construct.log");
    for (const auto& [I, cell] : sc.enumerate_cells() ) {
        for (auto [spin_sl, spin] : cell.enumerate_objects<Spin>()){
            log << "Spin at "<<spin->ipos<<"\n";
        }
    }
    log.close();


    return sc;
}


std::vector<ipos_t> nn_vectors = { {8,0,0}, {0,8,0}, {0,0,8}};

void link_spins(Supercell<Spin>& sc){
    for (const auto& [I, cell] : sc.enumerate_cells() ) {
        for (auto [spin_sl, spin] : cell.enumerate_objects<Spin>()){
            // link up first neighbours
            for (const auto& dx : nn_vectors){
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
            auto s = sc.get_object_at<Spin>(spin->ipos);
            if(s->ipos != spin->ipos){
                std::cerr<< "Lookup broken at "<<s->ipos<<std::endl;
            }
        }
    }

    link_spins(sc);

    return failed;
}

using namespace std;

int main (int argc, char *argv[]) {
    if (argc < 2){
        cout << "Usage: "<<argv[0]<<" L"<<std::endl;
        return 1;
    }
    int L = atoi(argv[1]);
    auto Z = imat33_t::from_cols({-L, L, L}, {L, -L, L}, {L, L, -L});
//    auto Z = imat33_t::from_cols({L,0,0}, {0, L, 0}, {0, 0, L});

    SuperLat sc = initialise_lattice(Z);

    int failed = test_lookup_consistency(sc);

    link_spins(sc);

    return failed;
}

