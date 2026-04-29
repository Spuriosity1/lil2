#include "supercell.hpp"

using namespace vector3;

struct Spin {
    ipos_t ipos;
    double Sz;

    Spin() : ipos(-1,-1,-1), Sz(0.0) {}
    Spin(const ipos_t& x) : ipos(x), Sz(0.0) {}
};

using MyCell = UnitCellSpecifier<Spin>;
using SuperLat = Supercell<Spin>;

static const imat33_t pyro_primitive_cell = imat33_t::from_cols({0,4,4}, {4,0,4}, {4,4,0});

inline auto build_pryo_primitive(){
    MyCell cell(pyro_primitive_cell);
    cell.add<Spin>(Spin({0,0,0}));
    cell.add<Spin>(Spin({0,2,2}));
    cell.add<Spin>(Spin({2,0,2}));
    cell.add<Spin>(Spin({2,2,0}));
    return cell;
}


//inline auto build_pyro_cubic(){
//    MyCell cell(imat33_t::from_cols({8,0,0},{0,8,0},{0,0,8}));
//    cell.add<Spin>(Spin({0,0,0}));
//    cell.add<Spin>(Spin({0,2,2}));
//    cell.add<Spin>(Spin({2,0,2}));
//    cell.add<Spin>(Spin({2,2,0}));
//    return cell;
//}


static const imat33_t pyro_A = imat33_t::from_cols({0,4,4}, {4,0,4}, {4,4,0});

inline auto build_pyro_cell() {
    MyCell cell(pyro_A);
    cell.add<Spin>(Spin({0,0,0}));
    cell.add<Spin>(Spin({0,2,2}));
    cell.add<Spin>(Spin({2,0,2}));
    cell.add<Spin>(Spin({2,2,0}));
    return cell;
}


inline auto build_cubic(const imat33_t& Z, 
        const imat33_t& A =imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8}) ) {
    MyCell cell(A);
    cell.add<Spin>(Spin({0,0,0}));
    return build_supercell<Spin>(cell, Z);
}


inline auto build_pyro(const imat33_t& Z) {
    MyCell cell = build_pyro_cell();
    return build_supercell<Spin>(cell, Z);
}


// Simple L×L×L cubic lattice, one spin per cell
inline auto build_simple_cubic(int L) {
    auto Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});
    return build_cubic(Z);
}

inline void set_cosine_wave(SuperLat& lat, ivec3_t Q) {
    auto B = lat.lattice.get_reciprocal_lattice_vectors();
    vec3<double> q = B * Q;
    for (auto& s : lat.get_objects<Spin>()){
        s.Sz = std::cos( dot<double>(q, s.ipos));
    }
}

// Set Sz[I] = cos(2π * dot(Q, I/D)) for each cell index I
inline void set_cosine_wave_on_sl(SuperLat& lat, ivec3_t Q, int sl) {
    auto B = lat.lattice.get_reciprocal_lattice_vectors();
    vec3<double> q = B * Q;
    for (const auto& [I, c] : lat.enumerate_cells() ){
        double phase = dot<double>(q, lat.lattice.translation_of(I));
        for (const auto [mu, s] : c.enumerate_objects<Spin>()){
            if (mu == sl){
                s->Sz = (1<<16)*std::cos(phase);
            }
        }

    }
}


static const std::vector<imat33_t> W_tests = {
    imat33_t::from_cols({2,0,0}, {0,1,0}, {0,0,1}),
    imat33_t::from_cols({2,0,0}, {0,2,0}, {0,0,1}),
    imat33_t::from_cols({1,-1,0}, {0,1,0}, {0,0,1}),
    imat33_t::from_cols({1,0,0}, {-1,1,0}, {0,0,1}),
    imat33_t::from_cols({-1,1,1}, {1,-1,1}, {1,1,-1})
};
