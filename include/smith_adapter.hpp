#pragma once

#include "smithNormalForm.hpp"
#include "unitcell_types.hpp"
#include "vec3.hpp"


// type converters
template<typename T>
inline vector3::mat33<T> from_snfmat(SmithNormalFormCalculator::Matrix<T> m){
	vector3::mat33<T> out;
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			out(i,j) = m[i][j];
	return out;
}

template<typename T>
inline SmithNormalFormCalculator::Matrix<T> to_snfmat(vector3::mat33<T>m){
	SmithNormalFormCalculator::Matrix<T> out(3,3);
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			out[i][j] = m(i,j);
	return out;
}


// convention: decomposition of a matrix A is L*A*R = D, D diagonal
// Further: to work with internal LIL routines, need 
// i) all D > 0
// ii) det L > 0
// iii) det R > 0
//
struct SNF_decomp {
	SNF_decomp(
			SmithNormalFormCalculator::SmithNormalFormDecomposition<int64_t>decomp,
            bool ensure_positive=true) : 
		L(from_snfmat(decomp.L)),
		Linv(from_snfmat(SmithNormalFormCalculator::inverse(decomp.L))),
		D(from_snfmat(decomp.D).diagonal()),
		R(from_snfmat(decomp.R)),
		Rinv(from_snfmat(SmithNormalFormCalculator::inverse(decomp.R)))
	{
        for (int i=0; i<3; i++){
            if (D[i] < 0) {
                // left-multiply L by a sign-flip on row i; right-multiply Linv by the same
                for (int j=0; j<3; j++){
                    L(i,j) *= -1;
                    Linv(j,i) *= -1;  // column i of Linv
                }
                D[i] *= -1;
            }
        }

        if (det(L) < 0){
            L *= -1;
            Linv *= -1;

            R *= -1;
            Rinv *= -1;
        }

        // mathematical fact guaranteed by det(D) >0, det(A)>0 and det(L)>0
        if (ensure_positive && (det(R) <= 0) ) {
            throw std::runtime_error(
                    "Unexpected negative determinant encountered (L =\n"
                    +vector3::to_string(L) + "D =\n" 
                    +vector3::to_string(D) + "R =\n" 
                    +vector3::to_string(R) + ")\n"
                    );
        }
    }

	imat33_t L;
	imat33_t Linv;
	ivec3_t D;
	imat33_t R;
	imat33_t Rinv;
};


