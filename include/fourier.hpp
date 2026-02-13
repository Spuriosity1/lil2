#pragma once

#include "supercell.hpp"
#include "unitcellspec.hpp"
#include <complex>
#include <type_traits>
#include <fftw3.h>
#include <typeindex>

// Accessor: extracts a double from an object at compile-time
// No virtual functions, no std::function - pure template magic
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

// Inner product in k-space
template<typename T>
FourierBuffer<T> inner(const FourierBuffer<T>& a, const FourierBuffer<T>& b) {
    FourierBuffer<T> result(a.num_sublattices, a.k_dims);
    for (int sl = 0; sl < a.num_sublattices; ++sl) {
        for (size_t i = 0; i < a.data[sl].size(); ++i) {
            result.data[sl][i] = std::conj(a.data[sl][i]) * b.data[sl][i];
        }
    }
    return result;
}

template<typename T>
FourierBuffer<T> empty_FT_buffer_like(const FourierBuffer<T>& template_buf) {
    return FourierBuffer<T>(template_buf.num_sublattices, template_buf.k_dims);
}

// The transformer: sets up FFTW plans for a specific field
template<typename T, typename Accessor, GeometricObject... Ts>
class FourierTransform {
    Supercell<Ts...>* sc;
    int num_sublattices;
    ivec3_t real_dims;  // real-space grid dimensions
    
    // FFTW data and plans per sublattice
    std::vector<double*> real_data;
    std::vector<fftw_complex*> k_data;
    std::vector<fftw_plan> plans;
    
    FourierBuffer<T> buffer;
    
public:
    FourierTransform(Supercell<Ts...>& supercell) 
        : sc(&supercell),
          real_dims(sc->lattice.size()),
          buffer(sc->sl_positions[std::type_index(typeid(T))].size(), real_dims)
    {
        num_sublattices = buffer.num_sublattices;
        int real_size = real_dims[0] * real_dims[1] * real_dims[2];
        
        // Allocate FFTW buffers and create plans for each sublattice
        for (int sl = 0; sl < num_sublattices; ++sl) {
            real_data.push_back(fftw_alloc_real(real_size));
            k_data.push_back(fftw_alloc_complex(real_size));
            
            plans.push_back(fftw_plan_dft_r2c_3d(
                real_dims[0], real_dims[1], real_dims[2],
                real_data[sl], k_data[sl],
                FFTW_MEASURE
            ));
        }
    }
    
    ~FourierTransform() {
        for (int sl = 0; sl < num_sublattices; ++sl) {
            fftw_destroy_plan(plans[sl]);
            fftw_free(real_data[sl]);
            fftw_free(k_data[sl]);
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
                real_data[sl][i] = Accessor::get(objects[obj_idx]);
            }
            
            // Execute FFT
            fftw_execute(plans[sl]);
            
            // Copy results to buffer
            for (int i = 0; i < num_prim; ++i) {
                buffer[sl][i] = std::complex<double>(
                    k_data[sl][i][0], 
                    k_data[sl][i][1]
                );
            }
        }
    }
    
    // Access the k-space buffer
    FourierBuffer<T>& operator[](int sl) { return buffer; }
    const FourierBuffer<T>& get_buffer() const { return buffer; }
    FourierBuffer<T>& get_buffer() { return buffer; }
};

// Factory function with nice syntax
template<typename T, auto MemberPtr, GeometricObject... Ts>
auto make_fourier_transform(Supercell<Ts...>& sc) {
    return FourierTransform<T, FieldAccessor<T, MemberPtr>, Ts...>(sc);
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
