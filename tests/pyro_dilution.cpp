
#include "supercell.hpp"
#include "unitcellspec.hpp"
#include "fourier.hpp"
#include <random>
#include <set>
#include <stack>



template<typename T>
concept UFElement = requires(T t) {
    { t.ipos } -> std::convertible_to<const ipos_t>;
    { t.neighbours } -> std::same_as<std::vector<T*>&>;
    { t.deleted } -> std::convertible_to<bool>;
    { t.dx } -> std::convertible_to<const ipos_t>;
    { t.parent } -> std::convertible_to<T*>;
};


// forward declarations
struct Spin;
struct Plaq;
struct Bond;

struct Spin {
    ipos_t ipos;
    std::vector<Spin*> neighbours;
    std::vector<Plaq*> plaqs_containing_me;
    std::vector<Bond*> bonds_containing_me;
    bool deleted=false;

    // auxiliary
    ipos_t dx;
    Spin* parent;

    Spin() : ipos(0,0,0), parent(this){ }
    Spin(const ipos_t& x) : ipos(x), parent(this) {}
};

struct Bond {
    ipos_t ipos;
    std::array<Spin*, 2> spin_members = {nullptr, nullptr};
    bool deleted = false;

    Bond() : ipos(0,0,0) {}
    Bond(const ipos_t& x) : ipos(x) {}
};

struct Plaq { 
    ipos_t ipos;
    std::vector<Spin*> spin_members;
    bool deleted = false;

    Plaq() : ipos(0,0,0){}
    Plaq(const ipos_t& x) : ipos(x) {}
};


namespace pyrochlore {
// nn_vectors[sl] -> neighbour vectors by spin SL
const std::vector<std::vector<ipos_t>> nn_vectors = {
    {{0,2,2},{2,0,2},{2,2,0},{0,-2,-2},{-2,0,-2},{-2,-2,0}},
    {{2, -2, 0}, {2, 0, -2}, {0, -2, -2}, {-2, 2, 0}, {-2, 0, 2}, {0, 2, 
2}},
    {{0, 2, -2}, {-2, 0, -2}, {-2, 2, 0}, {0, -2, 2}, {2, 0, 2}, {2, -2, 
0}},
    {{-2, -2, 0}, {-2, 0, 2}, {0, -2, 2}, {2, 2, 0}, {2, 0, -2}, {0, 
2, -2}}
};

const std::vector<ipos_t> pyro {
    {-1,-1,-1},
    {-1,1,1},
    {1,-1,1},
    {1,1,-1}
}; // vectos from cell origin to the pyro position



const std::vector<ipos_t> plaq_boundaries[4] = {
    {
        {0, -2, 2},    {2, -2, 0},
        {2, 0, -2},    {0, 2, -2}, 
        {-2, 2, 0},    {-2, 0, 2}
    },
    {
        { 0, 2,-2},    { 2, 2, 0},
        { 2, 0, 2},    { 0,-2, 2},
        {-2,-2, 0},    {-2, 0,-2}
    },
    {
        { 0,-2,-2},	{-2,-2, 0},
        {-2, 0, 2},	{ 0, 2, 2},
        { 2, 2, 0},	{ 2, 0,-2}
    },
    {
        { 0, 2, 2},	{-2, 2, 0},
        {-2, 0,-2},	{ 0,-2,-2},
        { 2,-2, 0},	{ 2, 0, 2}
    }
};


}; // end namespace


inline ipos_t floordiv(const ipos_t& x, int base){
    return ipos_t(x[0]/base, x[1]/base, x[2]/base);
}

void assert_position(void *ptr, ipos_t R) {
    if (ptr == nullptr) {
        std::cerr << "Could not resolve object at position " << R << "\n";
        throw std::runtime_error("Bad position");
    }
}

using MyCell = UnitCellSpecifier<Spin, Bond, Plaq>;
using SuperLat = Supercell<Spin, Bond, Plaq>;

// Generates a supercell of cubic dimension L
inline auto initialise_lattice(int L)
{
    using namespace pyrochlore;

    MyCell cell(imat33_t::from_cols({0,4,4},{4,0,4},{4,4,0}));

    for (int mu=0; mu<4; mu++){
        cell.add<Spin>(Spin(pyro[mu]));
        cell.add<Plaq>(Plaq(ipos_t{2,2,2} - pyro[mu]));
    }
    
    std::vector<std::pair<int, int>> munu_map;
    
    for (int mu=0; mu<4; mu++){
        for (int nu=mu+1; nu<4; nu++){
            cell.add<Bond>(Bond(floordiv(pyro[mu] + pyro[nu], 2)));
            munu_map.push_back(std::make_pair(mu, nu));
        }
    }
    
    // The size of the supercell
    imat33_t Z = imat33_t::from_cols({-L,L,L},{L,-L,L},{L,L,-L});

    Supercell sc = build_supercell<Spin, Bond, Plaq>(cell, Z);



    for (const auto& [I, cell] : sc.enumerate_cells() ) {
        for (auto [spin_sl, spin] : cell.enumerate_objects<Spin>()){
            // link up first neighbours
            for (const auto& dx : nn_vectors[spin_sl]){
                auto r_nn = spin->ipos + dx;
                auto nn = sc.get_object_at<Spin>(r_nn);
                std::cout << "Spin at "<<spin->ipos<<"\t| dx="<<dx<<"\n";
                assert_position(nn, r_nn);
                spin->neighbours.push_back(nn);
            }
        }
    }

//        for (auto [plaq_sl, plaq] : cell.enumerate_objects<Plaq>()){
//            // link up the spin neighbours of the plaquettes
//            for (const auto& dp : plaq_boundaries[plaq_sl]){
//                ipos_t R = plaq->ipos + dp;
//                Spin* s0 = sc.get_object_at<Spin>(R);
//                assert_position(s0, R);
//                plaq->spin_members.push_back(s0);
//                s0->plaqs_containing_me.push_back(plaq);
//            }
//        }
//
//        for (auto [bond_sl, bond] : cell.enumerate_objects<Bond>()){
//            auto [mu, nu] = munu_map[bond_sl];
//            auto delta = floordiv(pyro[mu]-pyro[nu], 2);
//
//            auto r0 = bond->ipos + delta;
//            auto r1 = bond->ipos - delta;
//
//            Spin* s0 = sc.get_object_at<Spin>(r0);
//            Spin* s1 = sc.get_object_at<Spin>(r1);
//
//            assert_position(s0, r0);
//            assert_position(s1, r1);
//
//            bond->spin_members[0] = s0;
//            bond->spin_members[1] = s1;
//
//            s0->bonds_containing_me.push_back(bond);
//            s1->bonds_containing_me.push_back(bond);
//            
//        }
//    }

    return sc;
}


// Marks all neighbours of spin 's' as deleted if any of them have a hole.
void set_spin_deleted(Spin& s, bool s_deleted){
    s.deleted = s_deleted;
    for (auto b : s.bonds_containing_me){
        b->deleted = b->spin_members[0]->deleted || b->spin_members[1]->deleted;
    }
    for (auto p : s.plaqs_containing_me){
        p->deleted = false;
        for (auto s2 : p->spin_members){
            p->deleted |= s2->deleted;
        }
    }
}

// Quicker version assuming all previous states were correct.
void set_spin_deleted_fast(Spin& s, bool s_deleted){
    if (s_deleted == s.deleted) return;
    if(s_deleted == true){
        // viral spread
        for (auto b: s.bonds_containing_me) b->deleted = true;
        for (auto p: s.plaqs_containing_me) p->deleted = true;
    } else {
        // spin restoration, full check necessary
        set_spin_deleted(s, s_deleted);
    }
}

// Sweeps through all spins, and deletes with probability p
// i.e. p = 0 means 100% clean
void spin_sweep(SuperLat& sc, double p, std::mt19937& rng){
    static auto rand01 = std::uniform_real_distribution();

    for (auto& s : sc.get_objects<Spin>()){
        set_spin_deleted(s, rand01(rng) < p);
    }
}


// finds the label for the cluster of element, 'elem'
template <UFElement T> 
T* find(T* elem){
    if (elem->parent == elem) return elem;

    T* p = elem->parent;

    auto root = find(p);
    elem->dx += p->dx;
    elem->parent = root;
    return root;
}

// Joins e1 to e2. 
// returns true if the resulting join created a winding path
template <UFElement T>
bool join_nodes(T* e1, T* e2, const LatticeIndexing& lat){
    T* root1 = find(e1);
    T* root2 = find(e2);

    auto Delta_x =  e2->ipos - e1->ipos;
    lat.wrap_super_delta(Delta_x);

    bool percolates = false;

    if (root1 == root2){
        // check for winding!
        auto loop_dx = Delta_x + (e1->dx - e2->dx);
        if (loop_dx != ipos_t{0,0,0}){
            percolates = true;
        }
    } else {
        // merge the two
        root1->parent = root2;
        root1->dx = root2->dx - root1->dx - Delta_x;
    }
    
    return percolates;
}



// modifies the "root" parts of the elements; union-find with path compression
template<UFElement T> 
bool initialise_tree(std::vector<T>& elements, const LatticeIndexing& lat
        ){
    // reset all spins
    for (auto& el : elements){
        el.parent = &el;
        el.dx = {0,0,0};
    }

    bool percolating = false;

    for (auto& el : elements){
        if (el.deleted) continue;

        // check neighbours
        for (auto s : el.neighbours){
            if (s->deleted) continue;
            percolating |= join_nodes(s, &el, lat);
        }
    }

    return percolating;
}


using namespace std;

int main (int argc, char *argv[]) {
    if (argc < 3){
        cout << "Usage: "<<argv[0]<<"L P"<<std::endl;
        return 1;
    }
    int L = atoi(argv[1]);
    double p = atof(argv[2]); // site deletion probability

    SuperLat sc = initialise_lattice(L);

    std::vector<int> links_percolate; // stores 1=percolates, 
                                  // 0=does not percolate

    // Delete about p*100% of the spins
    // (Bernoulli sample)
    std::mt19937 rng;
    spin_sweep(sc, p, rng);


    bool per = initialise_tree(sc.get_objects<Spin>(), sc.lattice);

    cout<<"Percolates: "<<per<<endl;
    



    return 0;
}

