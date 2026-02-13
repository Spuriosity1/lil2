#pragma once
#include "unitcellspec.hpp"
#include "smith_adapter.hpp"
#include <stdexcept>
#include "type_pretty.hpp"

class ObjectLookupError : public std::runtime_error {
    public:
  explicit ObjectLookupError(const std::string &msg)
      : std::runtime_error(msg) {}
};

using idx3_t = ivec3_t;

// type-tagged vector alias
template<class T>
struct SlPos : public std::vector<ipos_t> {
    using std::vector<ipos_t>::vector;
};

// the indexing engine
class LatticeIndexing {
    SNF_decomp LDW; // L D W
    imat33_t index_cell_vectors; // prim * L^-1 D
    imat33_t primitive_cell_vectors; // the reshaped primitive cell
    int num_primitive;

    CellWrapper wrap_to_index_cell;
    CellWrapper wrap_to_supercell;
    CellWrapper wrap_to_primitive;

public:
    LatticeIndexing( 
            const imat33_t& specified_cell_vectors, const imat33_t& supercell) :
                	// Smith decopose the supercell spec to find a primitive cell that aligns 
	// nicely with the supercell
	LDW(ComputeSmithNormalForm( to_snfmat(supercell))), // L supercell W = D
	// Cell vectors only used for indexing
	index_cell_vectors(specified_cell_vectors
			* supercell * LDW.R), // equivalent to L^-1 D
    primitive_cell_vectors(specified_cell_vectors * LDW.Linv),
    num_primitive(LDW.D[0]*LDW.D[1]*LDW.D[2]),
    wrap_to_index_cell(index_cell_vectors),
    wrap_to_supercell(specified_cell_vectors * supercell),
    wrap_to_primitive(primitive_cell_vectors)
	{
        // primitive_spec <- prim L^-1
        // A <- prim L^-1 D
        // A <- primitive_spec * D

        // sanity checks
        auto D = LDW.L * supercell * LDW.R;
        for (int i=0; i<3; i++) {
            assert(D(i,i) == LDW.D[i]);
            if (D(i,i) <= 0) {
                std::cerr << "D = " << LDW.D << std::endl;
                throw std::out_of_range("Supercell specification is singular.");
            }

        }
//        assert(index_cell_vectors == specified_primitive.cell_vectors * D);
	}

    ivec3_t size() const { return LDW.D; }
    int size(int i) const { return LDW.D[i]; }

    inline idx_t flat_from_idx3(const idx3_t& I) const {
		return (I[0]*LDW.D[1] + I[1])*LDW.D[2] + I[2];
	}

    inline idx3_t idx3_from_flat(idx_t i) const {
        idx3_t I;
        I[2] = i % LDW.D[2];
        i /= LDW.D[2];
        I[1]  = i % LDW.D[1];
        I[0] = i /LDW.D[1];
        return I;
    }

    auto num_primitive_cells(){ return num_primitive; }

/*
 * Wraps r to primitive cell, and returns the primitive-cell index
 *
 * Given a 3D unit cell, seek an index I in [0,D_0) x [0, D_1) x [0, D_2)
 * such that R = b * (I + D N) + r
 * for some N in Z3
 * where b is primitive_spec.lattice_vectors
 * mutating R to now contain the remainder r
*/
    inline idx3_t get_supercell_IDX(ipos_t& R) const;

    inline idx_t get_supercell_idx(ipos_t& R) const {
        return flat_from_idx3(get_supercell_IDX(R));
    }

    void wrap_index(ipos_t& R) const {
        wrap_to_index_cell.wrap(R);
    }

    void wrap_super(ipos_t& R) const {
        wrap_to_supercell.wrap(R);
    }

    void wrap_primitive(ipos_t& R) const {
        wrap_to_primitive.wrap(R);
    }

    // guarantted to return a vector in lattice coords [-0.5,0.5)^3
    void wrap_super_delta(ipos_t& R) const {
        R *= 2;
        auto x0 = index_cell_vectors*ipos_t{1,1,1};
        x0 /= 2;
        R += x0;
        wrap_super(R);
        R -= x0;
    }

    inline ipos_t translation_of(const idx3_t& I){
        return primitive_cell_vectors * I;
    }

};



inline idx3_t LatticeIndexing::get_supercell_IDX(ipos_t& R) const {
	// b^-1 R  = I + D N + b^-1 r
	ipos_t x = wrap_to_primitive.inv_latvecs_times_det * R;
	idx3_t I;
	for (int n=0; n<3; n++){
		auto res = moddiv(x[n], wrap_to_primitive.abs_det_latvecs);
		x[n] = res.rem;
		I[n] = res.quot;
		I[n] = mod(I[n], LDW.D[n]);
	}
	R = this->wrap_to_primitive.latvecs * x;
	for (int n=0; n<3; n++){
		R[n] /= wrap_to_primitive.abs_det_latvecs;
	}
	return I;
}


// forward declaration
template <GeometricObject... Ts>
struct Supercell;

// Helper struct representing a cell with its objects
template <GeometricObject... Ts>
struct Cell {
    idx3_t I;  // Cell index
    Supercell<Ts...>* sc;  // Pointer back to supercell
    idx_t flat_idx;  // Flattened cell index
    
    Cell(idx3_t idx, Supercell<Ts...>* supercell, idx_t flat)
        : I(idx), sc(supercell), flat_idx(flat) {}
    
    // Enumerate objects of type T in this cell
    template<typename T>
    auto enumerate_objects() const {

        static_assert((std::same_as<T, Ts> || ...),
                "Type T not stored in this Supercell");

        struct ObjectIterator {
            int num_sl;
            idx_t cell_flat;
            idx_t num_prim;
            std::vector<T>* obj_vec;

            struct Item {
                int sublattice;
                T* object;
            };

            struct Iterator {
                int current_sl;
                int max_sl;
                idx_t cell_idx;
                idx_t num_prim;
                std::vector<T>* vec;

                Item operator*() const {
                    idx_t idx = current_sl * num_prim + cell_idx;
                    return { current_sl, &(*vec)[idx] };
                }

                Iterator& operator++() {
                    ++current_sl;
                    return *this;
                }

                bool operator!=(const Iterator& other) const {
                    return current_sl != other.current_sl;
                }
            };

            Iterator begin() {
                return { 0, num_sl, cell_flat, num_prim, obj_vec };
            }

            Iterator end() {
                return { num_sl, num_sl, cell_flat, num_prim, obj_vec };
            }
        };

        auto& sl_vec = std::get<SlPos<T>>(sc->sl_positions);

        int num_sl = static_cast<int>(sl_vec.size());

        // Empty range: return empty iterator
        if (num_sl == 0) {
            return ObjectIterator{
                0,
                    flat_idx,
                    sc->lattice.num_primitive_cells(),
                    nullptr
            };
        }

        auto& obj_vec = std::get<std::vector<T>>(sc->objects);

        return ObjectIterator{
            num_sl,
                flat_idx,
                sc->lattice.num_primitive_cells(),
                &obj_vec
        };
    }

};

template <GeometricObject... Ts>
struct Supercell {
    // Indexing engine
    LatticeIndexing lattice;

    
    // vectors containing different object types
    // index convention: each sublattice is contiguous:
    // objects<T>[ sl * lattice.num_primitive_cells() + j ]
    std::tuple<std::vector<Ts>...> objects;

    // For each type T: positions of its sublattices in the reference cell
    // typeid(T) -> vector of ipos_t (one per sublattice)
    std::tuple<SlPos<Ts>...> sl_positions;

    // Cell -> object pointers (optional, for enumerate_cells)
    std::vector<std::vector<void*>> objects_by_cell;

    Supercell(LatticeIndexing lat,
              std::tuple<std::vector<Ts>...> objs,
              std::tuple<SlPos<Ts>...> sl_pos,
              std::vector<std::vector<void*>> obc)
        : lattice(std::move(lat)),
          objects(std::move(objs)),
          sl_positions(std::move(sl_pos)),
          objects_by_cell(std::move(obc)) {}

    // Enumerate all objects of type T
    template<typename T>
    auto& get_objects() {
        return std::get<std::vector<T>>(objects);
    }

    // Lookup by position (integer coordinates)
    // note deliberate ipos copy
    template<typename T>
    T* get_object_at(ipos_t pos) {
        ipos_t pos_copy = pos;
        // Wrap to supercell and get flattened index; pos becomes ref-cell coord
        idx_t super_i = lattice.get_supercell_idx(pos);

        auto& sl_pos_vec = std::get<SlPos<T>>(sl_positions);
        
        const int num_sl = static_cast<int>(sl_pos_vec.size());

        // Find which sublattice this position belongs to
        int sl = -1;
        for (int s = 0; s < num_sl; ++s) {
            if (sl_pos_vec[s] == pos) {
                sl = s;
                break;
            }
        }
        if (sl < 0) throw ObjectLookupError("No '"+type_name<T>()+"' found at "+to_string(pos_copy) + 
                " (resolved to SL pos "+to_string(pos)+")");

        auto& vec = std::get<std::vector<T>>(objects);
        const idx_t idx = sl * lattice.num_primitive_cells() + super_i;
        if (idx < 0 || idx >= static_cast<idx_t>(vec.size())) 
            throw std::runtime_error("Serious failure in get_object_at -- something is very wrong!");

        return &vec[idx];
    }

    // Enumerate cells
     auto enumerate_cells() {
        struct CellIterator {
            Supercell<Ts...>* sc;
            
            struct Item {
                idx3_t I;
                Cell<Ts...> cell;
            };
            
            struct Iterator {
                idx_t current;
                idx_t total;
                Supercell<Ts...>* supercell;
                
                Item operator*() const {
                    idx3_t I = supercell->lattice.idx3_from_flat(current);
                    return {I, Cell<Ts...>(I, supercell, current)};
                }
                
                Iterator& operator++() {
                    ++current;
                    return *this;
                }
                
                bool operator!=(const Iterator& other) const {
                    return current != other.current;
                }
            };
            
            Iterator begin() {
                return {0, sc->lattice.num_primitive_cells(), sc};
            }
            
            Iterator end() {
                idx_t total = sc->lattice.num_primitive_cells();
                return {total, total, sc};
            }
        };
        
        return CellIterator{this};
     }
};



template<GeometricObject... Ts>
Supercell<Ts...>
build_supercell(const UnitCellSpecifier<Ts...>& cell, const imat33_t& Z)
{
    LatticeIndexing lattice(cell.primitive_cell, Z);
    const int Np = lattice.num_primitive_cells();

    std::tuple<std::vector<Ts>...> objs;
    std::tuple<SlPos<Ts>...> sl_positions;

    // Discover sublattice positions for each type
    ([&]{
     auto& sl = std::get<SlPos<Ts>>(sl_positions);

     for (const auto& obj :
             std::get<std::vector<Ts>>(cell.basis_objects))
     {
         auto R = obj.ipos;
         lattice.wrap_primitive(R);
         sl.push_back(R);
     }
     }(), ...);
    
    // Allocate
    ([&]{
        auto& vec = std::get<std::vector<Ts>>(objs);
        int num_sl = std::get<SlPos<Ts>>(sl_positions).size();
        vec.resize(num_sl * Np);
    }(), ...);
    
    // Expand into supercell
    for (int flat = 0; flat < Np; ++flat) {
        idx3_t I = lattice.idx3_from_flat(flat);
        ipos_t R = lattice.translation_of(I);
        
        ([&]{
            const auto& basis = std::get<std::vector<Ts>>(cell.basis_objects);
            auto& vec = std::get<std::vector<Ts>>(objs);
            auto& sl_vec = std::get<SlPos<Ts>>(sl_positions);
            
            for (const auto& obj : basis) {
                Ts placed = obj;
                lattice.wrap_primitive(placed.ipos);
            
                int sl = std::find(sl_vec.begin(), sl_vec.end(), placed.ipos) - sl_vec.begin();
                assert(static_cast<size_t>(sl) <sl_vec.size());
                placed.ipos += R;
                vec[sl * Np + flat] = placed;
            }
        }(), ...);
    }
    
    // Build result and objects_by_cell as before...
    Supercell<Ts...> result(
        std::move(lattice),
        std::move(objs),
        std::move(sl_positions),
        std::vector<std::vector<void*>>()
    );
    
    // Build objects_by_cell...
    // (same as your code)
    
    return result;
}
