#pragma once

#include "smithNormalForm.hpp"
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
// Further: to work woth internal LIL routines, need 
// i) all D > 0
// ii) det L > 0
// iii) det R > 0 ?
struct SNF_decomp {
	SNF_decomp(
			SmithNormalFormCalculator::SmithNormalFormDecomposition<int64_t>decomp ) : 
		L(from_snfmat(decomp.L)),
		Linv(from_snfmat(SmithNormalFormCalculator::inverse(decomp.L))),
		D(from_snfmat(decomp.D).diagonal()),
		R(from_snfmat(decomp.R)),
		Rinv(from_snfmat(SmithNormalFormCalculator::inverse(decomp.R)))
	{
        for (int i=0; i<3; i++){
            if (D[i] < 0) { 
                // left-multiply by (-1,1,1) or similar
                for (int j=0; j<3; j++){
                    L(i,j) *= -1;
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

//        if (det(R) < 0){
//            R *= -1;
//            Rinv *= -1;
//            D *= -1;
//        }

    }

	imat33_t L;
	imat33_t Linv;
	ivec3_t D;
	imat33_t R;
	imat33_t Rinv;
};


