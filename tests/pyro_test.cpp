#include "supercell.hpp"
#include "unitcellspec.hpp"
#include "fourier.hpp"


struct Spin {
    ipos_t ipos;
    double S[3];
    std::array<std::vector<Spin*>, 2> neighbour;

    Spin(ipos_t ipos_) : ipos(ipos_), S{0,0,0} {}
};

int main (int argc, char *argv[]) {

    UnitCellSpecifier<Spin> cell(imat33_t::from_cols({0,4,4}, {4,0,4}, {4,4,0}));
    cell.add<Spin>(Spin({0,0,0}));
    cell.add<Spin>(Spin({0,2,2}));
    cell.add<Spin>(Spin({2,0,2}));
    cell.add<Spin>(Spin({2,2,0}));

    // The size of the supercell
    imat33_t Z = imat33_t::from_cols({1,2,3},{4,5,6}, {7,8,9});

    Supercell sc = build_supercell<Spin>(cell, Z);

    const std::vector<std::vector<ipos_t>> contact_map_1nn = {
        {{0,2,2},{2,0,2},{2,2,0},{0,-2,-2},{-2,0,-2},{-2,-2,0}},
        {{2, -2, 0}, {2, 0, -2}, {0, -2, -2}, {-2, 2, 0}, {-2, 0, 2}, {0, 2, 
  2}},
        {{0, 2, -2}, {-2, 0, -2}, {-2, 2, 0}, {0, -2, 2}, {2, 0, 2}, {2, -2, 
  0}},
        {{-2, -2, 0}, {-2, 0, 2}, {0, -2, 2}, {2, 2, 0}, {2, 0, -2}, {0, 
  2, -2}}
    };
    

    // preparation: linking up spins
    for (const auto& [I, cell] : sc.enumerate_cells() ) {
        for (auto [spin_sl, spin] : cell.enumerate_objects<Spin>()){
            // link up first neighbours
            for (const auto& dx : contact_map_1nn[spin_sl]){
                spin->neighbour[0].push_back(
                        sc.get_object_at<Spin>(spin->ipos + dx)
                );
            }
            // link up second neighbours
            // ... TODO
        }
    }

    // initialise for MC: set the spins to all point in Z direction
    // This loop needs to be very fast
    for (auto& s : sc.get_objects<Spin>()){
        s.S[0] = 0;
        s.S[1] = 0;
        s.S[2] = 0;
    }

    // ... some MC code
    auto Sx_tf = make_fourier_transform<Spin, &Spin::S[0]>(sc);
    auto Sy_tf = make_fourier_transform<Spin, &Spin::S[1]>(sc);
    auto Sz_tf = make_fourier_transform<Spin, &Spin::S[2]>(sc);

    // Execute transforms
    Sx_tf.transform();
    Sy_tf.transform();
    Sz_tf.transform();

    // Compute structure factor S(k) = sum_mu S_mu(k) * S_mu(-k)
    auto SdotS = empty_FT_buffer_like(Sx_tf.get_buffer());
    for (int mu = 0; mu < 4; mu++) {
        SdotS += inner(Sx_tf.get_buffer()[mu], Sx_tf.get_buffer()[mu]);
        SdotS += inner(Sy_tf.get_buffer()[mu], Sy_tf.get_buffer()[mu]);
        SdotS += inner(Sz_tf.get_buffer()[mu], Sz_tf.get_buffer()[mu]);
    }






    return 0;
}

