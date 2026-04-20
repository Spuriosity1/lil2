# Latlib

Any condensed matter physicist will be familiar with the pain of setting up
and using a good indexing scheme for points on a periodic lattice. It's tough
to get boundary conditions right, and even tougher to consider *disorder*
(i.e. deletions) in a way that doesn't feel, well, hacky.

This becomes even more complicated when one wishes to also consider
higher-dimensional geometrical objects than simple points, such as links,
elementary plaquettes, and elementary cells.

This project aims to provide a **single, universal** solution to this problem,
defining a flexible, templated factory function `build_supercell` which allows
the user to quickly and easily define arbitrary periodic structures in 3D (or
lower), and to compute Fourier transforms of fields defined on them.


## Installing

```bash
git clone https://github.com/Spuriosity1/lil2.git && cd lil2
meson setup build
ninja -C build install
```

If you can't write to `/usr/local`:
```bash
meson setup build -Dprefix="/your/install/prefix"
```

Link against the installed library with `-llatlib` (or via its pkgconfig entry
`latlib`).  The Fourier module additionally requires `-lfftw3`.


## Core concepts

### Coordinate convention

All positions are stored as `ipos_t` (`vec3<int64_t>`).  The library never
converts to floating-point internally, so wrapping arithmetic is always exact.
The physical Cartesian position of a site is `A * ipos` where `A` is the
real-space primitive cell matrix (which the user supplies and the library never
stores).  Reciprocal vectors and physical wavevectors are computed in double
precision on demand.

### The primitive cell matrix

`imat33_t` stores a 3×3 integer matrix in **column-major** convention:
`from_cols(a0, a1, a2)` creates a matrix whose columns are `a0`, `a1`, `a2`.
Matrix–vector multiplication `A * v` is standard (column vector on the right).
The supercell matrix `Z` passed to `build_supercell` is likewise column-based:
column *j* of `Z` gives the supercell basis vector as a linear combination of
primitive vectors.  The supercell lattice matrix is therefore `A * Z`.


## Quick start

### 1. Define a geometric object type

Any struct with a public `ipos_t ipos` member satisfies the `GeometricObject`
concept and can be stored in a supercell:

```cpp
#include "supercell.hpp"

struct Spin {
    ipos_t ipos;
    double Sx, Sy, Sz;

    Spin() : ipos{0,0,0}, Sx(0), Sy(0), Sz(0) {}
    Spin(ipos_t p) : ipos(p), Sx(0), Sy(0), Sz(0) {}
};
```

Multiple object types can coexist in the same supercell via the variadic
template parameter pack.

### 2. Build a unit cell

```cpp
// FCC primitive cell (lattice constant a=8 in integer units)
imat33_t A = imat33_t::from_cols({0,4,4}, {4,0,4}, {4,4,0});
UnitCellSpecifier<Spin> cell(A);

// Add basis sites (integer positions inside the primitive cell)
cell.add<Spin>(Spin({0,0,0}));   // site 0
cell.add<Spin>(Spin({0,2,2}));   // site 1  (pyrochlore sublattices)
cell.add<Spin>(Spin({2,0,2}));   // site 2
cell.add<Spin>(Spin({2,2,0}));   // site 3
```

### 3. Tile into a supercell

```cpp
// L×L×L repetitions of the primitive cell
int L = 8;
imat33_t Z = imat33_t::from_cols({L,0,0}, {0,L,0}, {0,0,L});

auto sc = build_supercell<Spin>(cell, Z);
// sc.lattice.num_primitive_cells() == L*L*L
// sc.get_objects<Spin>().size()    == 4 * L*L*L  (4 sublattices)
```

The Smith Normal Form decomposition is computed once at construction and
reused for all indexing.  For diagonal `Z` the decomposition is trivial (no
index permutation).

### 4. Access and iterate over objects

Objects of type `T` are laid out contiguously with the convention:
```
index = sublattice * num_primitive_cells + flat_cell_index
```

```cpp
// Flat iteration over all spins
for (auto& s : sc.get_objects<Spin>())
    s.Sz = 1.0;

// Lookup by physical position (wraps to supercell automatically)
Spin* s = sc.get_object_at<Spin>(ipos_t{0, 2, 2});

// Cell-by-cell iteration, with per-cell object enumeration
for (auto [I, cell] : sc.enumerate_cells()) {
    for (auto [sublattice, spin] : cell.enumerate_objects<Spin>()) {
        // I is the idx3_t cell index; sublattice is the integer sublattice label
        auto neighbour = sc.get_object_at<Spin>(spin->ipos + ipos_t{8,0,0});
    }
}
```

### 5. Sublattice positions

```cpp
auto& sl_pos = std::get<SlPos<Spin>>(sc.sl_positions);
// sl_pos[mu] is the ipos_t of sublattice mu in the reference primitive cell
```


## Fourier transforms

`fourier.hpp` provides an FFTW3-backed class for computing the discrete Fourier
transform of any scalar field defined on a supercell, on a per-sublattice basis.

### Setting up a transform

Use the factory function, supplying the type and a pointer-to-member (or a
free function) that extracts the scalar field:

```cpp
#include "fourier.hpp"

// One transformer per field component
auto ft_z = make_fourier_transform<Spin, &Spin::Sz>(sc);
```

The constructor allocates FFTW buffers and creates plans for every sublattice.
Plans are destroyed in the destructor.

### Executing the transform

```cpp
// Set spin pattern, then:
ft_z.transform();

// k-space data: FourierBuffer<Spin>
const auto& buf = ft_z.get_buffer();

// buf[mu][flat_k] is the complex DFT coefficient for sublattice mu at k-point flat_k
// flat_k = (K[0]*D[1] + K[1])*D[2] + K[2]  (row-major, same as FFTW)
```

The DFT convention is the standard forward transform:
```
Ã_μ(K) = Σ_I  s_μ(I)  exp(-2πi K·I / D)
```
where `I` is the integer cell index, `D = (D₀, D₁, D₂)` are the supercell
dimensions, and `μ` labels the sublattice.

### Physical wavevectors

```cpp
dmat33_t B = sc.lattice.get_reciprocal_lattice_vectors();
// B = 2π (A·Z)^{-T}
// Physical k-vector for DFT index K: q = B * K  (q in units of 1/[ipos units])
```

For a cosine wave `cos(q·r)` loaded into the spin field, set
`q = B * K_target` before evaluating; do **not** divide by `D`.

### Structure factor

`inner(a, b, lat, sl_positions)` computes the unnormalised cross-spectrum:

```
result[K] = Σ_{μ,ν}  Ã_μ*(K) · Ã_ν(K) · exp(+i q_K · (r_μ − r_ν))
```

For `a = b` this equals the physical structure factor `|Ã_full(K)|²` (the
squared magnitude of the full Fourier transform after folding sublattice phases
back in).  Divide by the total site count `N` to get the normalised `S(q)`.

```cpp
// Transform all three spin components
auto ft_x = make_fourier_transform<Spin, &Spin::Sx>(sc);
auto ft_y = make_fourier_transform<Spin, &Spin::Sy>(sc);
auto ft_z = make_fourier_transform<Spin, &Spin::Sz>(sc);
ft_x.transform(); ft_y.transform(); ft_z.transform();

// Collect sublattice positions
const auto& sl_pos = std::get<SlPos<Spin>>(sc.sl_positions);
std::vector<ipos_t> sl_pos_vec(sl_pos.begin(), sl_pos.end());

const int N = sc.lattice.num_primitive_cells() * (int)sl_pos.size();

// S(q) = (1/N) Σ_α inner(α, α)
auto Sq = inner<Spin>(ft_x.get_buffer(), ft_x.get_buffer(), sc.lattice, sl_pos_vec);
Sq     += inner<Spin>(ft_y.get_buffer(), ft_y.get_buffer(), sc.lattice, sl_pos_vec);
Sq     += inner<Spin>(ft_z.get_buffer(), ft_z.get_buffer(), sc.lattice, sl_pos_vec);

// Sq.data[0][flat_k] / N  is S(q_K)
```

### Backfolding identity

When the same physical system is described as a 1-sublattice supercell `Z·W`
or as a `det(W)`-sublattice supercell `Z`, the DFT coefficients satisfy:

```
Ã₁(K) = Σ_μ  exp(-i q_K · r_μ)  ·  Ã₂(μ, K mod D₂)
```

where `q_K = B₁ · K`.  This means `inner` returns identical values in either
representation — a useful consistency check in practice.


## Non-trivial supercell matrices

`build_supercell` accepts any invertible integer matrix `Z`, not just diagonal
ones.  For example, an FCC conventional cell expressed in terms of its own
primitive vectors:

```cpp
// Body-centred tetragonal supercell
imat33_t Z = imat33_t::from_cols({-L,L,L}, {L,-L,L}, {L,L,-L});
auto sc = build_supercell<Spin>(cell, Z);
```

The Smith Normal Form decomposition `L·Z·R = D` is computed internally.
`D[0]*D[1]*D[2]` always equals `|det(Z)|`, and the FFTW grid dimensions are
`D[0] × D[1] × D[2]`.
