#pragma once

#include <string_view>
#include <typeinfo>
#include <string>

#if defined(__clang__) || defined(__GNUC__)
    #define HAS_PRETTY_FUNCTION 1
#elif defined(_MSC_VER)
    #define HAS_FUNCSIG 1
#endif

#if defined(__GNUC__) && !defined(__clang__)
    #include <cxxabi.h>
    #define HAS_DEMANGLE 1
#endif

inline std::string demangle(const char* name) {
#if HAS_DEMANGLE
    int status = 0;
    char* dem = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0 && dem) {
        std::string out(dem);
        std::free(dem);
        return out;
    }
#endif
    return name; // fallback
}

template <typename T>
std::string type_name() {
#if HAS_PRETTY_FUNCTION
    std::string_view p = __PRETTY_FUNCTION__;
    auto start = p.find('=') + 2;
    auto end   = p.rfind(';');
    return std::string(p.substr(start, end - start));
#elif HAS_FUNCSIG
    std::string_view p = __FUNCSIG__;
    auto start = p.find('<') + 1;
    auto end   = p.rfind('>');
    return std::string(p.substr(start, end - start));
#else
    // Absolute fallback: works on *every* C++ compiler
    return demangle(typeid(T).name());
#endif
}

