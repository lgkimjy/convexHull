/**
 * @file: Polygon.hpp
 * @author: Jun Young Kim ( lgkimjy@kist.re.kr )
 * @department: Cognitive & Collaborative Robotics Research Group 
 *              @ Korea Institute of Science Technology (KIST)
 * @date: 2024. 6. 20.
 * 
 * @description: Compute Convex Hull of a set of points and check if a point (ZMP) is within the support polygon
 */

#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>

#include "Contact/ContactPoint.h"

class SupportPolygon {
public:
    SupportPolygon(const std::vector<ContactPoint>& points) : points_(points) {
        vertices_ = convexHull(points_);
        calculateEdgeEquations();
    }

    // Function to check if a point (x, y) is within the support polygon
    virtual bool isWithinSupportPolygon(double x_zmp, double y_zmp) {
        if(edges_.size() < 1) return false;

        for (const auto& edge : edges_) {
            double a = edge(0);
            double b = edge(1);
            double c = edge(2);
            if (a * x_zmp + b * y_zmp + c < 0) {
                return false;
            }
        }
        return true;
    }

    // Print edge equations
    void printEdges() {
        for (const auto& edge : edges_) {
            std::cout << "Edge equation: " << edge(0) << " * x + " << edge(1) << " * y + " << edge(2) << " >= 0" << std::endl;
        }
    }

    // Function to compute the cross product of vectors OA and OB
    double cross(const ContactPoint& O, const ContactPoint& A, const ContactPoint& B) {
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    }

    // Function to calculate the convex hull of a set of points
    std::vector<ContactPoint> convexHull(std::vector<ContactPoint>& points) {
        int n = points.size(), k = 0;
        std::vector<ContactPoint> hull(2 * n);

        // Sort points lexicographically
        std::sort(points.begin(), points.end());

        // Build lower hull
        for (int i = 0; i < n; ++i) {
            while (k >= 2 && cross(hull[k-2], hull[k-1], points[i]) <= 0) k--;
            hull[k++] = points[i];
        }

        // Build upper hull
        for (int i = n-1, t = k+1; i >= 0; --i) {
            while (k >= t && cross(hull[k-2], hull[k-1], points[i]) <= 0) k--;
            hull[k++] = points[i];
        }

        hull.resize(k-1);
        return hull;
    }
    
    // Return the vertices of the support polygon
    std::vector<ContactPoint> getVertices() const {
        return vertices_;
    }
    
    // Return the edges of the support polygon
    std::vector<Eigen::Vector3d> getEdges() const {
        return edges_;
    }

protected:
    std::vector<ContactPoint> points_;
    std::vector<ContactPoint> vertices_;
    std::vector<Eigen::Vector3d> edges_;

    void calculateEdgeEquations() {
        int N = vertices_.size();
        for (int i = 0; i < N; ++i) {
            int j = (i + 1) % N;  // Next vertex
            double a = vertices_[i].y - vertices_[j].y;
            double b = vertices_[j].x - vertices_[i].x;
            double c = vertices_[i].x * vertices_[j].y - vertices_[j].x * vertices_[i].y;
            edges_.emplace_back(a, b, c);
        }
    }
};
