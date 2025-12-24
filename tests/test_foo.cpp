#include "gtest/gtest.h"
#include "Foo/foo.hpp"

class TestFoo : public ::testing::Test
{
protected:
    virtual void SetUp()
    {

    }
    virtual void TearDown()
    {

    }
    Foo obj;
};

// TEST(TestSuiteName, TestCaseName)
TEST_F(TestFoo, add1)
{
    EXPECT_EQ(obj.add(1,2), 3);
}

TEST_F(TestFoo, add2)
{
    ASSERT_NEAR(obj.add(3,2.000001), 5, 1e-5);
}

TEST_F(TestFoo, print)
{
    testing::internal::CaptureStdout();
    obj.printGreet();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "Hello, World!\n");
}

TEST_F(TestFoo, multiplication1)
{
    Eigen::MatrixXd A(1000,1000);
    A.setRandom();
    Eigen::MatrixXd B(1000,1000);
    B.setRandom();
    Eigen::MatrixXd C(1000,1000);
    C = A * B;
    // ASSERT_TRUE(obj.multiply(A,B).isApprox(C));
    EXPECT_EQ(obj.multiply(A,B), C);
}

TEST_F(TestFoo, multiplication2)
{
    Eigen::MatrixXd A(2,2);
    A << 1, 2, 3, 4;
    Eigen::MatrixXd B(2,2);
    B << 5, 6, 7, 8;
    Eigen::MatrixXd C(2,2);
    C << 19, 22, 43, 50;
    // ASSERT_TRUE(obj.multiply(A,B).isApprox(C));
    EXPECT_EQ(obj.multiply(A,B), C);
}

TEST(TestFoo2, add1)
{
    Foo obj;
    EXPECT_EQ(obj.add(1,2), 3);
}

TEST(TestFoo2, add2)
{
    Foo obj;
    EXPECT_EQ(obj.add(3,2), 5);
}