#pragma once

#include "supercell.hpp"
#include <complex>
#include <numeric>
#include <type_traits>
#include <fftw3.h>


// Accessor: extracts a double from an object at compile-time
template<typename T, auto MemberPtr>
struct FieldAccessor {
    static constexpr auto ptr = MemberPtr;

    static double get(const T& obj) {
        if constexpr (std::is_member_object_pointer_v<decltype(MemberPtr)>) {
            return obj.*MemberPtr;
        } else {
            return MemberPtr(obj);
        }
    }

    static void set(T& obj, double val) {
        if constexpr (std::is_member_object_pointer_v<decltype(MemberPtr)>) {
            obj.*MemberPtr = val;
        }
    }
};

// K-space buffer: stores transformed data per sublattice
template<typename T>
struct FourierBuffer {
    int num_sublattices;
    ivec3_t k_dims;  // dimensions in k-space
    std::vector<std::vector<std::complex<double>>> data;  // [sublattice][k_index]

    FourierBuffer(int num_sl, ivec3_t dims)
        : num_sublattices(num_sl), k_dims(dims) {
        int k_size = dims[0] * dims[1] * dims[2];
        data.resize(num_sl, std::vector<std::complex<double>>(k_size));
    }

    std::vector<std::complex<double>>& operator[](int sl) { return data[sl]; }
    const std::vector<std::complex<double>>& operator[](int sl) const { return data[sl]; }

    FourierBuffer& operator+=(const FourierBuffer& other) {
        for (int sl = 0; sl < num_sublattices; ++sl)
            for (size_t i = 0; i < data[sl].size(); ++i)
                data[sl][i] += other.data[sl][i];
        return *this;
    }
};

// Accumulator for sublattice-resolved correlations.
//
// corr(mu, nu)[k] = conj(Ã_mu(k)) * Ã_nu(k),   summed over MC steps.
//
// Raw DFT output Ã_mu does NOT include the sublattice phase exp(-i q·r_mu);
// that phase is applied later via SublatWeightMatrix::phase_factors().contract().
template<typename T>
struct FourierCorrelator {
    int num_sublattices;
    ivec3_t k_dims;
    std::vector<std::vector<std::complex<double>>> data;  // [mu*num_sl+nu][k_index]

    FourierCorrelator(int num_sl, ivec3_t dims)
        : num_sublattices(num_sl), k_dims(dims) {
        int k_size = dims[0] * dims[1] * dims[2];
        data.resize(num_sl * num_sl,
                    std::vector<std::complex<double>>(k_size, 0.0));
    }

    std::vector<std::complex<double>>& operator()(int mu, int nu) {
        return data[mu * num_sublattices + nu];
    }
    const std::vector<std::complex<double>>& operator()(int mu, int nu) const {
        return data[mu * num_sublattices + nu];
    }

    FourierCorrelator& operator+=(const FourierCorrelator& other) {
        for (int i = 0; i < num_sublattices * num_sublattices; ++i)
            for (size_t k = 0; k < data[i].size(); ++k)
                data[i][k] += other.data[i][k];
        return *this;
    }
};



template<typename T>
void correlate_add(FourierCorrelator<T>&acc, const FourierBuffer<T>& a, const FourierBuffer<T>& b){
    assert(a.num_sublattices == b.num_sublattices);
    int k_size = a.k_dims[0] * a.k_dims[1] * a.k_dims[2];
    for (int mu = 0; mu < a.num_sublattices; ++mu)
        for (int nu = 0; nu < a.num_sublattices; ++nu)
            for (int k = 0; k < k_size; ++k)
                acc(mu, nu)[k] += std::conj(a.data[mu][k]) * b.data[nu][k];
}

// Build the sublattice correlator matrix from two k-space buffers:
//   result(mu, nu)[k] = conj(a[mu][k]) * b[nu][k]
// This is the only operation needed inside a MC accumulation loop.
// Later, we may need to multiply by a global phase correciton factor to 
// deal with different sublattice positions.
template<typename T>
FourierCorrelator<T> correlate(const FourierBuffer<T>& a, const FourierBuffer<T>& b) {
    // allocate the temporary and initialise to zero
    FourierCorrelator<T> result(a.num_sublattices, a.k_dims);
    correlate_add(result, a, b);
    return result;
}




// Per-k-point sublattice weight matrix w[mu][nu][k].
//
// Used to contract a FourierCorrelator into a scalar per k-point:
//   result[k] = Σ_{μ,ν} w(μ,ν)[k] · corr(μ,ν)[k]
//
// Two factory functions cover the common cases:
//   phase_factors()  — sublattice structure-factor phases exp(+i q·(r_μ−r_ν))
//   constant()       — q-independent real weight matrix (e.g. local-axis projections)
//
// Combining both is done with operator*:
//   auto w = SublatWeightMatrix::phase_factors(lat, sl_pos)
//          * SublatWeightMatrix::constant(num_sl, k_dims, axis_weights);
struct SublatWeightMatrix {
    int num_sublattices;
    ivec3_t k_dims;
    std::vector<std::vector<std::complex<double>>> weights;  // [mu*num_sl+nu][k]

    SublatWeightMatrix(int num_sl, ivec3_t dims)
        : num_sublattices(num_sl), k_dims(dims) {
        int k_size = dims[0] * dims[1] * dims[2];
        weights.resize(num_sl * num_sl,
                       std::vector<std::complex<double>>(k_size, 0.0));
    }

    std::vector<std::complex<double>>& operator()(int mu, int nu) {
        return weights[mu * num_sublattices + nu];
    }
    const std::vector<std::complex<double>>& operator()(int mu, int nu) const {
        return weights[mu * num_sublattices + nu];
    }

    // w(μ,ν)[k] = exp(+i q_k · (r_μ − r_ν))
    //
    // Sign convention: the DFT produces Ã_mu^raw(q) = Σ_R X(R,μ) exp(−i q·R).
    // The full sublattice-aware transform is Ã_μ(q) = Ã_μ^raw(q) · exp(−i q·r_μ),
    // so conj(Ã_μ)·Ã_ν = conj(Ã_μ^raw)·Ã_ν^raw · exp(+i q·(r_μ−r_ν)).
    static SublatWeightMatrix phase_factors(
            const LatticeIndexing& lat,
            const std::vector<ipos_t>& sl_positions) {
        const int num_sl = static_cast<int>(sl_positions.size());
        SublatWeightMatrix w(num_sl, lat.size());
        const auto B = lat.get_reciprocal_lattice_vectors();
        idx3_t Q;
        for (Q[0] = 0; Q[0] < lat.size(0); Q[0]++)
            for (Q[1] = 0; Q[1] < lat.size(1); Q[1]++)
                for (Q[2] = 0; Q[2] < lat.size(2); Q[2]++) {
                    const int k = lat.flat_from_idx3(Q);
                    const auto q = B * vector3::vec3<double>(Q);
                    for (int mu = 0; mu < num_sl; ++mu)
                        for (int nu = 0; nu < num_sl; ++nu) {
                            const double arg = dot(q, vector3::vec3<double>(
                                    sl_positions[mu] - sl_positions[nu]));
                            w(mu, nu)[k] = std::polar(1.0, arg);
                        }
                }
        return w;
    }

    // q-independent real weights: w(μ,ν)[k] = w_mn[mu][nu] for all k.
    static SublatWeightMatrix constant(
            int num_sl, ivec3_t k_dims,
            const std::vector<std::vector<double>>& w_mn) {
        assert(static_cast<int>(w_mn.size()) == num_sl);
        SublatWeightMatrix w(num_sl, k_dims);
        for (int mu = 0; mu < num_sl; ++mu) {
            assert(static_cast<int>(w_mn[mu].size()) == num_sl);
            for (int nu = 0; nu < num_sl; ++nu)
                std::fill(w(mu, nu).begin(), w(mu, nu).end(),
                          std::complex<double>(w_mn[mu][nu], 0.0));
        }
        return w;
    }

    // Element-wise product: combine two weight matrices (e.g. phase * local-axis).
    SublatWeightMatrix operator*(const SublatWeightMatrix& other) const {
        assert(num_sublattices == other.num_sublattices);
        SublatWeightMatrix result(num_sublattices, k_dims);
        for (int i = 0; i < num_sublattices * num_sublattices; ++i)
            for (size_t k = 0; k < weights[i].size(); ++k)
                result.weights[i][k] = weights[i][k] * other.weights[i][k];
        return result;
    }

    // Contract: result[k] = Σ_{μ,ν} w(μ,ν)[k] · corr(μ,ν)[k]
    template<typename T>
    std::vector<std::complex<double>> contract(
            const FourierCorrelator<T>& corr) const {
        assert(corr.num_sublattices == num_sublattices);
        int k_size = k_dims[0] * k_dims[1] * k_dims[2];
        std::vector<std::complex<double>> result(k_size, 0.0);
        for (int mu = 0; mu < num_sublattices; ++mu)
            for (int nu = 0; nu < num_sublattices; ++nu)
                for (int k = 0; k < k_size; ++k)
                    result[k] += weights[mu * num_sublattices + nu][k]
                                * corr(mu, nu)[k];
        return result;
    }
};


template<typename T>
FourierBuffer<T> empty_FT_buffer_like(const FourierBuffer<T>& template_buf) {
    return FourierBuffer<T>(template_buf.num_sublattices, template_buf.k_dims);
}


/*
 * Computes sublattice-resolved Fourier transforms over a Supercell.
 *
 * For a field X(I, μ) the output is:
 *
 *   Ã_μ^raw(K) = Σ_I  X(I,μ) · exp(−i 2π Σ_s  I[s]·K[s] / L[s])
 *
 * Note: the sublattice phase exp(−i q·r_μ) is NOT included here; apply it
 * post-hoc via SublatWeightMatrix::phase_factors().contract().
 */
template<typename T, typename Accessor, GeometricObject... Ts>
class FourierTransformC2C {
    Supercell<Ts...>* sc;
    int num_sublattices;
    ivec3_t real_dims;

    std::vector<fftw_complex*> rs_data;
    std::vector<fftw_complex*> ks_data;
    std::vector<fftw_plan> plans;

    FourierBuffer<T> buffer;

public:
    FourierTransformC2C(Supercell<Ts...>& supercell)
        : sc(&supercell),
          real_dims(sc->lattice.size()),
          buffer(std::get<SlPos<T>>(sc->sl_positions).size(), real_dims)
    {
        num_sublattices = buffer.num_sublattices;
        int real_size = real_dims[0] * real_dims[1] * real_dims[2];

        for (int sl = 0; sl < num_sublattices; ++sl) {
            rs_data.push_back(fftw_alloc_complex(real_size));
            ks_data.push_back(fftw_alloc_complex(real_size));
            plans.push_back(fftw_plan_dft_3d(
                static_cast<int>(real_dims[0]),
                static_cast<int>(real_dims[1]),
                static_cast<int>(real_dims[2]),
                rs_data[sl], ks_data[sl],
                FFTW_FORWARD, FFTW_MEASURE
            ));
        }
    }

    ~FourierTransformC2C() {
        for (int sl = 0; sl < num_sublattices; ++sl) {
            fftw_destroy_plan(plans[sl]);
            fftw_free(rs_data[sl]);
            fftw_free(ks_data[sl]);
        }
    }

    void transform() {
        auto& objects = sc->template get_objects<T>();
        int num_prim = sc->lattice.num_primitive_cells();

        for (int sl = 0; sl < num_sublattices; ++sl) {
            for (int i = 0; i < num_prim; ++i) {
                rs_data[sl][i][0] = Accessor::get(objects[sl * num_prim + i]);
                rs_data[sl][i][1] = 0.0;
            }
            fftw_execute(plans[sl]);
            for (int i = 0; i < num_prim; ++i)
                buffer[sl][i] = {ks_data[sl][i][0], ks_data[sl][i][1]};
        }
    }

    const FourierBuffer<T>& get_buffer() const { return buffer; }
    FourierBuffer<T>& get_buffer() { return buffer; }
};

// Factory function with nice syntax:
//   auto ft = make_fourier_transform<Spin, &Spin::Sz>(sc);
template<typename T, auto MemberPtr, GeometricObject... Ts>
auto make_fourier_transform(Supercell<Ts...>& sc) {
    return FourierTransformC2C<T, FieldAccessor<T, MemberPtr>, Ts...>(sc);
}


// Describes a 2D plane in BZ-index space, parameterised as:
//
//   K(n1, n2) = n1·e1 + n2·e2,   n1 ∈ [0, N1),  n2 ∈ [0, N2)
//
// N1 and N2 are computed automatically from e1/e2 and the lattice dimensions D:
//   N = lcm_j( D[j] / gcd(|e[j]|, D[j]) )
// This is the smallest period such that the fold coordinate
//   φ(I) = Σ_j e[j]·I[j]·N/D[j]   mod N
// is integer-valued and hits every value in [0,N) uniformly.
//
// Example — h,h,l plane on an L×L×L supercell:
//   KPlaneSpec spec({1,1,0}, {0,0,1}, sc.lattice.size());  → N1=L, N2=L
struct KPlaneSpec {
    ivec3_t e1, e2;
    int N1, N2;

    // Compute the period for one step vector
    static int auto_N(const ivec3_t& e, const ivec3_t& D) {
        int n = 1;
        for (int j = 0; j < 3; ++j) {
            int ej = static_cast<int>(std::abs(e[j]));
            int dj = static_cast<int>(D[j]);
            n = std::lcm(n, dj / std::gcd(ej, dj));
        }
        return n;
    }

    KPlaneSpec(ivec3_t e1_, ivec3_t e2_, const ivec3_t& D)
        : e1(e1_), e2(e2_),
          N1(auto_N(e1_, D)), N2(auto_N(e2_, D)) {}
};


// Sublattice-resolved Fourier transform restricted to a 2D plane in k-space.
//
// At each transform() call, real-space data is folded along the direction
// perpendicular to the plane (O(N_sites)), then a 2D FFTW transform is
// executed (O(N1·N2·log(N1·N2))).  This avoids allocating or computing the
// full 3D k-space grid.
//
// The output FourierBuffer has k_dims = {N1, N2, 1}; flat index n1*N2+n2
// corresponds to k-point K = n1·e1 + n2·e2.
// correlate(), FourierCorrelator, and SublatWeightMatrix all work unchanged.
// Call make_phase_weights() to get the sublattice phase matrix for this plane.
template<typename T, typename Accessor, GeometricObject... Ts>
class FourierTransformPlanar {
    Supercell<Ts...>* sc;
    KPlaneSpec spec;
    int num_sublattices;
    int num_prim;

    // fold_idx[i_flat] = n1*N2 + n2  for the 2D output grid
    std::vector<int> fold_idx;

    std::vector<fftw_complex*> rs_data;  // plane_size per sublattice
    std::vector<fftw_complex*> ks_data;
    std::vector<fftw_plan> plans;

    FourierBuffer<T> buffer;  // k_dims = {N1, N2, 1}

    void build_fold_map() {
        const auto D = sc->lattice.size();
        const int N1 = spec.N1, N2 = spec.N2;

        // α[j] = e[j] * N / D[j]  — guaranteed integer by construction of N
        ivec3_t alpha1, alpha2;
        for (int j = 0; j < 3; ++j) {
            alpha1[j] = spec.e1[j] * N1 / static_cast<int>(D[j]);
            alpha2[j] = spec.e2[j] * N2 / static_cast<int>(D[j]);
        }

        fold_idx.resize(num_prim);
        std::vector<int> count(N1 * N2, 0);

        for (int i = 0; i < num_prim; ++i) {
            const auto I = sc->lattice.idx3_from_flat(i);
            int raw1 = static_cast<int>(alpha1[0]*I[0] + alpha1[1]*I[1] + alpha1[2]*I[2]);
            int raw2 = static_cast<int>(alpha2[0]*I[0] + alpha2[1]*I[1] + alpha2[2]*I[2]);
            int j1 = ((raw1 % N1) + N1) % N1;
            int j2 = ((raw2 % N2) + N2) % N2;
            fold_idx[i] = j1 * N2 + j2;
            ++count[j1 * N2 + j2];
        }

        // Validate: every 2D grid point is hit equally often
        const int expected = num_prim / (N1 * N2);
        for (int c : count)
            if (c != expected)
                throw std::runtime_error(
                    "KPlaneSpec: fold is not uniform — "
                    "e1 and e2 are not compatible with these lattice dimensions");
    }

public:
    FourierTransformPlanar(Supercell<Ts...>& supercell, const KPlaneSpec& spec_)
        : sc(&supercell), spec(spec_),
          num_prim(sc->lattice.num_primitive_cells()),
          buffer(std::get<SlPos<T>>(sc->sl_positions).size(),
                 {spec_.N1, spec_.N2, 1})
    {
        num_sublattices = buffer.num_sublattices;
        const int plane_size = spec.N1 * spec.N2;

        build_fold_map();

        for (int sl = 0; sl < num_sublattices; ++sl) {
            rs_data.push_back(fftw_alloc_complex(plane_size));
            ks_data.push_back(fftw_alloc_complex(plane_size));
            plans.push_back(fftw_plan_dft_2d(
                spec.N1, spec.N2,
                rs_data[sl], ks_data[sl],
                FFTW_FORWARD, FFTW_MEASURE
            ));
        }
    }

    ~FourierTransformPlanar() {
        for (int sl = 0; sl < num_sublattices; ++sl) {
            fftw_destroy_plan(plans[sl]);
            fftw_free(rs_data[sl]);
            fftw_free(ks_data[sl]);
        }
    }

    void transform() {
        auto& objects = sc->template get_objects<T>();
        const int plane_size = spec.N1 * spec.N2;

        for (int sl = 0; sl < num_sublattices; ++sl) {
            // Zero fold buffer
            for (int k = 0; k < plane_size; ++k)
                rs_data[sl][k][0] = rs_data[sl][k][1] = 0.0;

            // Fold: accumulate real-space values into the 2D grid
            for (int i = 0; i < num_prim; ++i) {
                rs_data[sl][fold_idx[i]][0] +=
                    Accessor::get(objects[sl * num_prim + i]);
                // imaginary part stays 0 (real-valued input field)
            }

            fftw_execute(plans[sl]);

            for (int k = 0; k < plane_size; ++k)
                buffer[sl][k] = {ks_data[sl][k][0], ks_data[sl][k][1]};
        }
    }

    // Sublattice phase weight matrix for this plane.
    // w(μ,ν)[k] = exp(+i q_k · (r_μ − r_ν))  where  q_k = B·K(n1,n2).
    // Flat index k = n1*N2 + n2.
    SublatWeightMatrix make_phase_weights(
            const std::vector<ipos_t>& sl_positions) const {
        const int num_sl = static_cast<int>(sl_positions.size());
        const int N1 = spec.N1, N2 = spec.N2;
        SublatWeightMatrix w(num_sl, {N1, N2, 1});
        const auto B = sc->lattice.get_reciprocal_lattice_vectors();

        for (int n1 = 0; n1 < N1; ++n1)
            for (int n2 = 0; n2 < N2; ++n2) {
                const int k_flat = n1 * N2 + n2;
                const auto q = B * vector3::vec3<double>(
                    spec.e1[0]*n1 + spec.e2[0]*n2,
                    spec.e1[1]*n1 + spec.e2[1]*n2,
                    spec.e1[2]*n1 + spec.e2[2]*n2);
                for (int mu = 0; mu < num_sl; ++mu)
                    for (int nu = 0; nu < num_sl; ++nu) {
                        const double arg = dot(q, vector3::vec3<double>(
                                sl_positions[mu] - sl_positions[nu]));
                        w(mu, nu)[k_flat] = std::polar(1.0, arg);
                    }
            }
        return w;
    }

    const FourierBuffer<T>& get_buffer() const { return buffer; }
    FourierBuffer<T>& get_buffer() { return buffer; }
    const KPlaneSpec& get_spec() const { return spec; }
};

// Factory function:
//   KPlaneSpec hhl({1,1,0}, {0,0,1}, sc.lattice.size());
//   auto ft = make_planar_fourier_transform<Spin, &Spin::Sz>(sc, hhl);
template<typename T, auto MemberPtr, GeometricObject... Ts>
auto make_planar_fourier_transform(Supercell<Ts...>& sc, const KPlaneSpec& spec) {
    return FourierTransformPlanar<T, FieldAccessor<T, MemberPtr>, Ts...>(sc, spec);
}
