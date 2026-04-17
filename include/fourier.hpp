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
    
    // Arithmetic operations
    FourierBuffer& operator+=(const FourierBuffer& other) {
        for (int sl = 0; sl < num_sublattices; ++sl) {
            for (size_t i = 0; i < data[sl].size(); ++i) {
                data[sl][i] += other.data[sl][i];
            }
        }
        return *this;
    }
};

// Computes the k-space inner product (unnormalised structure factor):
//
//   result[K] = Σ_{μ,ν}  conj(Ã_μ(K)) · Ã_ν(K) · exp(+i q_K · (r_μ - r_ν))
//
// where q_K = B · K is the physical k-vector (B from get_reciprocal_lattice_vectors),
// and r_μ / r_ν are sublattice positions.  Dividing by the total site count N gives
// the physical structure factor S(q).
//
// The result is a single-sublattice FourierBuffer (one complex scalar per k-point).
template<typename T>
FourierBuffer<T> inner(const FourierBuffer<T>& a, const FourierBuffer<T>& b,
        const LatticeIndexing& lat, const std::vector<ipos_t>& sl_positions) {
    assert(a.num_sublattices == b.num_sublattices);
    assert(static_cast<int>(sl_positions.size()) == a.num_sublattices);

    // Scalar result: one entry per k-point
    FourierBuffer<T> result(1, a.k_dims);

    const auto B = lat.get_reciprocal_lattice_vectors();

    idx3_t Q;
    for (Q[0]=0; Q[0]<lat.size(0); Q[0]++)
    for (Q[1]=0; Q[1]<lat.size(1); Q[1]++)
    for (Q[2]=0; Q[2]<lat.size(2); Q[2]++) {
        const int i_flat = lat.flat_from_idx3(Q);
        const auto q = B * vector3::vec3<double>(Q);

        std::complex<double> s = 0;
        for (int sl1 = 0; sl1 < a.num_sublattices; ++sl1)
            for (int sl2 = 0; sl2 < a.num_sublattices; ++sl2) {
                const double arg = dot(q, vector3::vec3<double>(sl_positions[sl1] - sl_positions[sl2]));
                s += std::conj(a.data[sl1][i_flat]) * b.data[sl2][i_flat]
                   * std::exp(std::complex<double>(0, arg));
            }
        result.data[0][i_flat] = s;
    }
    return result;
}

template<typename T>
FourierBuffer<T> empty_FT_buffer_like(const FourierBuffer<T>& template_buf) {
    return FourierBuffer<T>(template_buf.num_sublattices, template_buf.k_dims);
}


/*
 * This class is a general purpose way to compute Fourier transforms of 
 * fields belonging to any GeometricObject.
 *
 * A general field may be expressed as 
 * X(I0, I1, I2, mu), where
 * 0 <= I0 < L0
 * 0 <= I1 < L1
 * 0 <= I2 < L2
 * mu is an integer sublattice label.
 *
 * We wish to compute (with some normalisation)
 *
 * Y(K0, K1, K2, mu) =
 *   \sum_{I0 in [0, L0)} 
 *     \sum_{I1 in [0, L1)} 
 *       \sum_{I2 in [0, L2)} 
 *         X(I0, I1, I2, mu) * exp(- i 2 \pi \sum_s=0^2  I[s] K[s] / L[s] )
 * 
 * The actual positions: If primitive cell vectors are in matrix [a],
 * R_i = [a]_ij I_j
 * BZ positions:
 * q_i = 2 pi * ([a]^-1 ^T)_ij  K_j / L_j
 *
 */
template<typename T, typename Accessor, GeometricObject... Ts>
class FourierTransformC2C {
    Supercell<Ts...>* sc;
    int num_sublattices;
    ivec3_t real_dims;  // real-space grid dimensions
    
    // FFTW data and plans per sublattice
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
        
        // Allocate FFTW buffers and create plans for each sublattice
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
    
    // Execute the transform
    void transform() {
        auto& objects = sc->template get_objects<T>();
        int num_prim = sc->lattice.num_primitive_cells();
        
        // For each sublattice
        for (int sl = 0; sl < num_sublattices; ++sl) {
            // Copy data from objects to real_data buffer
            for (int i = 0; i < num_prim; ++i) {
                idx_t obj_idx = sl * num_prim + i;
                rs_data[sl][i][0] = Accessor::get(objects[obj_idx]);
                rs_data[sl][i][1] = 0.0;
            }
            
            // Execute FFT
            fftw_execute(plans[sl]);
            
            // Copy results to buffer
            for (int i = 0; i < num_prim; ++i) {
                buffer[sl][i] = std::complex<double>(
                    ks_data[sl][i][0], 
                    ks_data[sl][i][1]
                );
            }
        }
    }
    
    // Access the k-space buffer; index per sublattice with get_buffer()[sl]
    const FourierBuffer<T>& get_buffer() const { return buffer; }
    FourierBuffer<T>& get_buffer() { return buffer; }
};

// Factory function with nice syntax
template<typename T, auto MemberPtr, GeometricObject... Ts>
auto make_fourier_transform(Supercell<Ts...>& sc) {
    return FourierTransformC2C<T, FieldAccessor<T, MemberPtr>, Ts...>(sc);
}

// Usage example:
/*
// Create transformers for each spin component
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
*/
