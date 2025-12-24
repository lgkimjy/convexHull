#include "foo.hpp"

Foo::Foo()
{
}

Foo::~Foo()
{

}

void Foo::printGreet()
{
    std::cout << "Hello, World!" << std::endl;
}

// template<typename T>
// T Foo::subtraction(T A, T B)
// {
//     return A - B;
// }

double Foo::add(double A, double B)
{
    return A + B;
}

Eigen::MatrixXd Foo::multiply(Eigen::MatrixXd A, Eigen::MatrixXd B)
{
    return A * B;
}