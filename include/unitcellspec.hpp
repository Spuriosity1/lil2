#pragma once


#include "unitcell_types.hpp"
#include "modulus.hpp"

// Computes the closed-form, simplified matrix inverse
constexpr imat33_t unnormed_inverse(const imat33_t& A){
	std::array<int64_t, 3> b0 = {-A(1, 2) * A(2, 1) + A(1, 1) * A(2, 2),
		A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2),
		-A(0, 2) * A(1, 1) + A(0, 1) * A(1, 2)};
	std::array<int64_t, 3> b1 = {A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2),
		-A(0, 2) * A(2, 0) + A(0, 0) * A(2, 2),
		A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2)};
	std::array<int64_t, 3> b2 = {-A(1, 1) * A(2, 0) + A(1, 0) * A(2, 1),
		A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1),
		-A(0, 1) * A(1, 0) + A(0, 0) * A(1, 1)};
	return imat33_t::from_rows(b0,b1,b2);
}


class BadMatrixError : public std::exception {
    public:
        BadMatrixError(const std::string& mat_name, const imat33_t& M, const std::string& context) {
            message_ = context + "\n"+mat_name+"="+vector3::to_string(M);
        }
    const char* what() const throw() {
        return message_.c_str();
    }

    private:
    std::string message_;
};

// handles the maths for wrapping to within a cell
// By design, requires a right-handed coordinate system in latvecs (det > 0).
struct CellWrapper {
    const imat33_t latvecs;
    const imat33_t inv_latvecs_times_det;
    const int64_t abs_det_latvecs;

    CellWrapper(const imat33_t& latvecs_) :
        	latvecs(latvecs_),
	inv_latvecs_times_det( unnormed_inverse(latvecs) ),
	abs_det_latvecs(det(latvecs))
    {
        if(abs_det_latvecs == 0){
            throw BadMatrixError("latvecs", latvecs_, 
                    "Cannot initalise CellWrapper with singular lattice vectors");
        } else if (abs_det_latvecs < 0){
            throw BadMatrixError("latvecs", latvecs_,
                    "Cannot initalise CellWrapper with left-handed lattice vectors");
        }
    }


    void wrap(ipos_t& X) const {
        ipos_t x = inv_latvecs_times_det * X; // this / |det(A)| is the true x
        for (int i=0; i<3; i++){
            x[i] = mod(x[i], abs_det_latvecs);
        }
        X = latvecs*x;
        for (int i=0; i<3; i++){
            assert(X[i] % abs_det_latvecs == 0);
            X[i] /= abs_det_latvecs;
        }
    }

};

template <typename T>
concept GeometricObject = 
  requires(T t) {
      { t.ipos } -> std::same_as<ipos_t&>;
  };


template<GeometricObject... Ts>
class UnitCellSpecifier {
public:
    imat33_t primitive_cell;
    std::tuple<std::vector<Ts>...> basis_objects;
    
    explicit UnitCellSpecifier(imat33_t cell) 
        : primitive_cell(cell) {}
    
    template<typename T>
        requires (std::same_as<T, Ts> || ...)
    void add(T&& obj) {
        std::get<std::vector<T>>(basis_objects).push_back(std::forward<T>(obj));
    }
};

