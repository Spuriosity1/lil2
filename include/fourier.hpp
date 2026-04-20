#pragma once

#include "supercell.hpp"
#include <complex>
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
