#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <climits>

inline double division(int numerator, int denominator)
{
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

inline int floorDivision (int numerator, int denominator)
{
    return static_cast<int>(std::floor(division(numerator, denominator)));
}

inline int ceilDivision (int numerator, int denominator)
{
    return static_cast<int>(std::ceil(division(numerator, denominator)));
}


inline double power(int base, double exponent)
{
    return std::pow(static_cast<double>(base), exponent);
}

inline int floorPower (int base, double exponent)
{
    return static_cast<int>(std::floor(power(base, exponent)));
}

inline int ceilPower (int base, double exponent)
{
    return static_cast<int>(std::ceil(power(base, exponent)));
}

inline double logarithm(int base, int number)
{
    return log2(static_cast<double>(number)) / log2(static_cast<double>(base));
}

inline int floorLogarithm (int base, int number)
{
    return static_cast<int>(std::floor(logarithm(base, number)));
}

inline int ceilLogarithm(int base, int number)
{
    return static_cast<int>((std::ceil(logarithm(base, number))));
}

inline void printError(std::string const & error)
{
    std::cerr << "% [ERROR] " << error << std::endl;
}

template<typename T>
void printVector(std::ostream& os, std::vector<T> const & vector)
{
    if (not vector.empty())
    {
        os << vector[0];
        for (size_t i = 1; i < vector.size(); i += 1)
        {
            os << "," << vector[i];
        }
    }
}

template<typename T>
void printArray(std::ostream& os, int size, T* data)
{
    if (size != 0)
    {
        os << data[0];
        for (size_t i = 1; i < size; i += 1)
        {
            os << "," << data[i];
        }
        os << "\n";
    }
}

inline unsigned int getLeftmostOneIndex64(unsigned long long int const & val)
{
    return __builtin_clzll(val);
}


inline int getLeftmostOneIndex32(unsigned int val)
{
    return __builtin_clz(val);
}

inline int getRightmostOneIndex64(unsigned long long int const & val)
{
    return 64 - __builtin_ffsll(val);
}

inline int getRightmostOneIndex32(unsigned int val)
{
    return 32 - __builtin_ffs(val);
}

inline unsigned int getMask32(int bitIndex)
{
    return 1 << (31 - bitIndex);
}

inline unsigned int getLeftFilledMask32(int bitIndex)
{
    return UINT_MAX << (31 - bitIndex);
}

inline unsigned int getRightFilledMask32(int bitIndex)
{
    return UINT_MAX >> bitIndex;
}

inline int getPopCount(unsigned int val)
{
    return __builtin_popcount(val);
}