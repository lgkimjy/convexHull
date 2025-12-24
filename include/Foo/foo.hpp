#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>


class Foo
{
private:
public:
    Foo();
    ~Foo();
    
    void printGreet();
    
    double add(double A, double B);

    template<typename T>
    T subtraction(T A, T B)
    {
        return A - B;
    }

    Eigen::MatrixXd multiply(Eigen::MatrixXd A, Eigen::MatrixXd B);
};


template <int a, typename T>
class Foo2 {
public:
    std::vector<T> elements;
    Foo2(int size) {
        elements = std::vector<T>(size);
    }

    template<typename D>
    D add(D A, D B) {
        return A + B;
    }

    T add2(int index) {
        return a + elements[index];
    }
};