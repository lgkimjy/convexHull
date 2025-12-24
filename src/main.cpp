#include "main.hpp"

int main()
{
    Foo A;
    Foo B;
    Foo2<3, int> C(5);

    A.printGreet();
    B.printGreet();
    
    double a = 5.0;
    double b = 3.5;
    
    std::cout << C.add(a, b) << std::endl;
    std::cout << A.subtraction(a, b) << std::endl;

    std::cout << C.add2(3) << std::endl;

    return 0;
}