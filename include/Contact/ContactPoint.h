#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>

struct ContactPoint {
    double x, y;

    bool operator<(const ContactPoint& p) const {
        return x < p.x || (x == p.x && y < p.y);
    }
};
