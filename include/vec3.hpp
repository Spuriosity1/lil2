#ifndef VEC3_CUSTOM_HPP
#define VEC3_CUSTOM_HPP


#include <concepts>
#include <cstddef>
#include <array>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace vector3 {


template <typename T>
struct mat33;
    
template <typename T>
struct vec3 {
	constexpr vec3() = default;
	constexpr vec3(const vec3&) = default;

    template<typename S>
    requires std::convertible_to<S, T>
    constexpr vec3(const vec3<S>& other){
        m_x[0] = static_cast<T>(other[0]);
        m_x[1] = static_cast<T>(other[1]);
        m_x[2] = static_cast<T>(other[2]);
    }

	constexpr vec3(T x,T y,T z){
		m_x[0] = x;
		m_x[1] = y;
		m_x[2] = z;
	}

    constexpr vec3(std::initializer_list<T> list) {
        assert(list.size() == 3);
        auto it = list.begin();
        m_x[0] = *it++;
        m_x[1] = *it++;
        m_x[2] = *it++;
    }

    constexpr vec3& operator=(const vec3&) = default;

	constexpr inline T& operator[](size_t idx){ return m_x[idx]; }
	constexpr inline T operator[](size_t idx) const { return m_x[idx];	}

	constexpr inline T& operator()(size_t idx){ return m_x[idx]; }
	constexpr inline T operator()(size_t idx) const { return m_x[idx];	}

	static constexpr size_t size(){ return 3; }

/*
    constexpr vec3& operator=(const vec3& v){
		m_x[0] = v.m_x[0];
		m_x[1] = v.m_x[1];
		m_x[2] = v.m_x[2];
        return *this;
    }
*/
    constexpr vec3& operator+=(const vec3& v){
        m_x[0] += v.m_x[0];
        m_x[1] += v.m_x[1];
        m_x[2] += v.m_x[2];
        return *this;
    }

    constexpr vec3& operator-=(const vec3& v){
        m_x[0] -= v.m_x[0];
        m_x[1] -= v.m_x[1];
        m_x[2] -= v.m_x[2];
        return *this;
    }

    template<typename S>
    constexpr vec3& operator*=(S alpha){
		T tmp = static_cast<T>(alpha);
        m_x[0] *= tmp;
        m_x[1] *= tmp;
        m_x[2] *= tmp;
        return *this;
    }

    template<typename S>
    vec3& operator/=(S alpha){
		T tmp = static_cast<T>(alpha);
        m_x[0] /= tmp;
        m_x[1] /= tmp;
        m_x[2] /= tmp;
        return *this;
    }

    template<typename S>
    vec3& operator%=(S alpha){
		T tmp = static_cast<T>(alpha);
        m_x[0] %= tmp;
        m_x[1] %= tmp;
        m_x[2] %= tmp;
        return *this;
    }

    vec3 operator+(const vec3& v) const {
        vec3 res(*this);
        return res += v;
    }

    vec3 operator-(const vec3& v) const {
        vec3 res(*this);
        return res -= v;
    }

	vec3 operator-() const {	
        vec3 res(*this);
		res.m_x[0] = -res.m_x[0];
		res.m_x[1] = -res.m_x[1];
		res.m_x[2] = -res.m_x[2];
		return res;
	}

	bool operator==(const vec3& other) const {
		return (other.m_x[0] == m_x[0]) && 
			(other.m_x[1] == m_x[1]) &&
			(other.m_x[2] == m_x[2]);
	}

	friend vec3<T> mat33<T>::operator*(const vec3<T>&) const;

protected:

    T m_x[3]={0,0,0};
};


// Free vector functions
template <typename T, typename S>
requires std::convertible_to<S, T>
vec3<S> operator*(T alpha, const vec3<S>& v){
    vec3<S> copy(v);
	copy *= static_cast<S>(alpha);
	return copy;
}

template <typename T, typename S>
requires std::convertible_to<S, T>
T dot(const vec3<T>& u, const vec3<S>& v){
	return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];	
}

template<typename T>
vec3<T> operator+(const vec3<T>& v1, const vec3<T>& v2){
	vec3<T> w(v1);
	w += v2;
	return w;
}

template<typename T>
vec3<T> operator-(const vec3<T>& v1, const vec3<T>& v2){
	vec3<T> w(v1);
	w -= v2;
	return w;
}
	

template <typename T>
struct mat33 {
	static constexpr size_t size(){ return 9; }
	
	template<typename S=T>
	requires std::convertible_to<S, T>
	static constexpr mat33 from_cols(std::array<S,3> a0,
			std::array<S,3> a1,
			std::array<S,3> a2
			){
		mat33 out;
		for (int row=0; row<3; row++){
			out(row, 0) =a0[row];
			out(row, 1) =a1[row];
			out(row, 2) =a2[row];
		}
		return out;
	}
	
	template <typename S=T>
	requires std::convertible_to<S, T>
	static constexpr mat33 from_rows(std::array<S, 3> r0,
			std::array<S, 3> r1,
			std::array<S, 3> r2){
		mat33 out;

		for (int col=0; col<3; col++){
			out(0,col) =r0[col];
			out(1,col) =r1[col];
			out(2,col) =r2[col];
		}
		return out;
	}
	
	template <typename S>
	requires std::convertible_to<S, T>
	static constexpr mat33 from_other(mat33<S> other){
		mat33 retval;
		for (int i=0; i<9; i++){
			retval[i] = static_cast<T>(other[i]);
		}
		return retval;
	}

    constexpr vec3<T> col(int i) const {
        return vec3<T>(m_x[i],m_x[3+i],m_x[6+i]);
    }

    constexpr vec3<T> row(int i) const {
        return vec3<T>(m_x[3*i],m_x[3*i+1],m_x[3*i+2]);
    }

    // transpose
    constexpr mat33<T> tr() const {
        mat33<T> retval;
        for (int i=0; i<3; i++){
            retval(0,i) = this->operator()(i,0);
            retval(1,i) = this->operator()(i,1);
            retval(2,i) = this->operator()(i,2);
        }
        return retval;
    }

    static constexpr mat33 eye() {
        mat33 retval;
        retval(0,0) = 1;
        retval(1,1) = 1;
        retval(2,2) = 1;
        return retval;
    }

	vec3<T> operator*(const vec3<T>& v) const {
		vec3<T> res;
		res[0] = __dot(m_x, v.m_x);
		res[1] = __dot(m_x+3, v.m_x);
		res[2] = __dot(m_x+6, v.m_x);	
		return res;
	}

	inline constexpr T& operator()(int i, int j){return m_x[3*i + j];	}
	inline constexpr T  operator()(int i, int j) const {return m_x[3*i + j];	}

	inline constexpr T& operator[](int i)		  {return m_x[i];}
	inline constexpr T  operator[](int i) const {return m_x[i];}

	mat33<T> operator*(const mat33<T>& x) const {
		mat33<T> res;
		// ah hell the compiler can optimise this
		for (int i=0; i<3; i++){
			for (int k=0; k<3; k++){
				for (int j=0; j<3; j++){
					res(i,j) += (*this)(i, k) * x(k, j);
				}
			}
		}
		return res;
	}

    mat33<T> operator+(const mat33<T>& x) const {
        mat33<T> res(*this);
        for (int i=0; i<9; i++){
            res.m_x[i] += x.m_x[i];
        }
        return res;
    }
	
	template <typename S>
	requires std::convertible_to<S, T>
	mat33<T> operator*=(const S& alpha){
		for (int i=0; i<9; i++){
			this->m_x[i] *= alpha;
		}
		return *this;
	}

	vec3<T> diagonal(){
		vec3<T> out;
		for (int i=0; i<3; i++)
			out[i] = (*this)(i,i);

		return out;
	}
		
		
protected:
	T m_x[9]={0,0,0, 0,0,0, 0,0,0}; 
	// convention:
	// [ 0 1 2 ]
	// [ 3 4 5 ]
	// [ 6 7 8 ]
	//
	T __dot(const T* x1, const T* x2) const {
		return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2];
	}
};

// standard scalar multiplication
template <typename T, typename S>
requires std::convertible_to<S, T>
mat33<S> operator*(T alpha, const mat33<S>& v){
    mat33<S> copy(v);
	copy *= static_cast<S>(alpha);
	return copy;
}


// LEFT multiplicaiton, treating v as row vector
// evaluates sum_j v[j] M[j,i]
template <typename T, typename S>
requires std::convertible_to<S, T>
vec3<T> operator*(vec3<T> v, const mat33<S>& M){
    vec3<T> u{0,0,0};
    for (int i=0;i<3;i++){
        for (int j=0; j<3; j++){
            u[i] +=  v[j] * M(j,i);
        }
    }

	return u;
}


template <typename T, typename S>
requires std::convertible_to<S, T>
bool operator==(const mat33<T>& a, const mat33<S>& b){
    for (int i=0; i<9; i++){
        if (a[i] != b[i]) return false;
    }
    return true;
}

// JSON IO
template<typename T>
void to_json(nlohmann::json& j, const vec3<T>& v){
	j = nlohmann::json({v[0], v[1], v[2]});
}


template<typename T>
void from_json(const nlohmann::json& j, vec3<T>& v){
	j.at(0).get_to(v[0]);
	j.at(1).get_to(v[1]);
	j.at(2).get_to(v[2]);
}

template<typename T>
void to_json(nlohmann::json &j, const mat33<T>& M){
	j = nlohmann::json({
		{M(0,0),M(0,1),M(0,2)},
		{M(1,0),M(1,1),M(1,2)},
		{M(2,0),M(2,1),M(2,2)}
	});
}

template<typename T>
void from_json(const nlohmann::json &j, mat33<T> &M){
	for (size_t i=0; i<3; i++) {
		for (size_t k=0; k<3; k++) {
			j.at(i).at(k).get_to<T>(M(i,k));
		}
	}	
}

/// CONVENIENT TYPEDEFS
typedef vec3<int> vec3i;
typedef vec3<double> vec3d;

template<typename T>
std::ostream& operator<<(std::ostream& os, const vec3<T>& v){
	os << "["<<v[0]<<" "<<v[1]<<" "<<v[2]<<"]";
	return os;
}


template<typename T>
std::ostream& operator<<(std::ostream& os, const mat33<T>& m){
	os << "["<<m(0,0)<<" "<<m(0,1)<<" "<<m(0,2)<<"] \\n";
	os << "["<<m(1,0)<<" "<<m(1,1)<<" "<<m(1,2)<<"] \\n";
	os << "["<<m(2,0)<<" "<<m(2,1)<<" "<<m(2,2)<<"] ";
	return os;
}

template<typename T>
std::string to_string(const vec3<T>& v){
    std::ostringstream oss;
    oss<<v;
    return oss.str();
}


template<typename T>
std::string to_string(const mat33<T>& m){
    std::ostringstream oss;
    oss<<m;
    return oss.str();
}

// Explicit functions preserving intness
template<typename V>
inline constexpr V det(mat33<V> a){
    return (a(0,0) * (a(1,1) * a(2,2) - a(2,1) * a(1,2))
           -a(1,0) * (a(0,1) * a(2,2) - a(2,1) * a(0,2))
           +a(2,0) * (a(0,1) * a(1,2) - a(1,1) * a(0,2)));
}

};



// HASHABILITY
//
template <typename T>
struct std::hash<vector3::vec3<T>>
{
  std::size_t operator()(const vector3::vec3<T>& k) const
  {
    using std::size_t;
    using std::hash;

    // Compute individual hash values for first,
    // second and third and combine them using XOR
    // and bit shifting:

    return hash<T>()(k[0]) ^ (hash<T>()(k[1]) << 1) ^ (hash<T>()(k[2]) << 2);
  }
};


#endif // !VEC3_CUSTOM_HPP
