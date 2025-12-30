#ifndef INNER_APPROX_HULL_HPP
#define INNER_APPROX_HULL_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>

#include "Contact/ContactPoint.h"
#include "Contact/Polygon.hpp"

// Simple KDTree for 2D points
class SimpleKDTree {
private:
    struct Node {
        ContactPoint point;
        Node* left;
        Node* right;
        
        Node(const ContactPoint& p) : point(p), left(nullptr), right(nullptr) {}
    };
    
    Node* root;
    
    Node* buildTree(std::vector<ContactPoint>& points, int depth) {
        if (points.empty()) return nullptr;
        
        int axis = depth % 2; // 0 for x, 1 for y
        int median = points.size() / 2;
        
        // Sort by current axis
        if (axis == 0) {
            std::nth_element(points.begin(), points.begin() + median, points.end(),
                [](const ContactPoint& a, const ContactPoint& b) { return a.x < b.x; });
        } else {
            std::nth_element(points.begin(), points.begin() + median, points.end(),
                [](const ContactPoint& a, const ContactPoint& b) { return a.y < b.y; });
        }
        
        Node* node = new Node(points[median]);
        
        std::vector<ContactPoint> leftPoints(points.begin(), points.begin() + median);
        std::vector<ContactPoint> rightPoints(points.begin() + median + 1, points.end());
        
        node->left = buildTree(leftPoints, depth + 1);
        node->right = buildTree(rightPoints, depth + 1);
        
        return node;
    }
    
    void nearestNeighbor(Node* node, const ContactPoint& target, double& bestDist, ContactPoint& bestPoint, int depth) {
        if (node == nullptr) return;
        
        double dist = std::sqrt(std::pow(node->point.x - target.x, 2) + 
                               std::pow(node->point.y - target.y, 2));
        if (dist < bestDist) {
            bestDist = dist;
            bestPoint = node->point;
        }
        
        int axis = depth % 2;
        double axisDist = (axis == 0) ? (target.x - node->point.x) : (target.y - node->point.y);
        
        Node* near = (axisDist < 0) ? node->left : node->right;
        Node* far = (axisDist < 0) ? node->right : node->left;
        
        nearestNeighbor(near, target, bestDist, bestPoint, depth + 1);
        
        if (std::abs(axisDist) < bestDist) {
            nearestNeighbor(far, target, bestDist, bestPoint, depth + 1);
        }
    }
    
    void destroyTree(Node* node) {
        if (node == nullptr) return;
        destroyTree(node->left);
        destroyTree(node->right);
        delete node;
    }
    
public:
    SimpleKDTree(const std::vector<ContactPoint>& points) {
        if (points.empty()) {
            root = nullptr;
            return;
        }
        std::vector<ContactPoint> pointsCopy = points;
        root = buildTree(pointsCopy, 0);
    }
    
    ~SimpleKDTree() {
        destroyTree(root);
    }
    
    double query(const ContactPoint& target) {
        if (root == nullptr) return std::numeric_limits<double>::max();
        
        double bestDist = std::numeric_limits<double>::max();
        ContactPoint bestPoint;
        nearestNeighbor(root, target, bestDist, bestPoint, 0);
        return bestDist;
    }
};

// Gaussian KDE implementation
class GaussianKDE {
private:
    std::vector<ContactPoint> data;
    double bandwidth;
    
    double gaussianKernel(double x, double y) {
        return std::exp(-0.5 * (x * x + y * y)) / (2.0 * M_PI);
    }
    
    double computeBandwidth() {
        // Silverman's rule of thumb for 2D
        int n = data.size();
        if (n < 2) return 1.0;
        
        // Compute standard deviations
        double meanX = 0.0, meanY = 0.0;
        for (const auto& p : data) {
            meanX += p.x;
            meanY += p.y;
        }
        meanX /= n;
        meanY /= n;
        
        double varX = 0.0, varY = 0.0;
        for (const auto& p : data) {
            varX += (p.x - meanX) * (p.x - meanX);
            varY += (p.y - meanY) * (p.y - meanY);
        }
        varX /= (n - 1);
        varY /= (n - 1);
        
        double stdX = std::sqrt(varX);
        double stdY = std::sqrt(varY);
        
        // Silverman's rule: h = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)) * sigma
        // For 2D: h = (4/4)^(1/6) * n^(-1/6) * sigma = n^(-1/6) * sigma
        double h = std::pow(n, -1.0/6.0) * std::sqrt(stdX * stdX + stdY * stdY) / std::sqrt(2.0);
        return std::max(h, 0.1); // Minimum bandwidth
    }
    
public:
    GaussianKDE(const std::vector<ContactPoint>& points, double bw = -1.0) : data(points) {
        if (bw < 0) {
            bandwidth = computeBandwidth();
        } else {
            bandwidth = bw;
        }
    }
    
    std::vector<double> evaluate(const std::vector<ContactPoint>& points) {
        std::vector<double> densities;
        densities.reserve(points.size());
        
        int n = data.size();
        double invBandwidth = 1.0 / bandwidth;
        double invBandwidthSq = invBandwidth * invBandwidth;
        double norm = 1.0 / (n * bandwidth * bandwidth);
        
        for (const auto& p : points) {
            double density = 0.0;
            for (const auto& d : data) {
                double dx = (p.x - d.x) * invBandwidth;
                double dy = (p.y - d.y) * invBandwidth;
                density += gaussianKernel(dx, dy);
            }
            densities.push_back(density * norm);
        }
        
        return densities;
    }
};

// Helper function to compute percentile
inline double percentile(const std::vector<double>& values, double p) {
    if (values.empty()) return 0.0;
    
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    
    double index = p / 100.0 * (sorted.size() - 1);
    int lower = static_cast<int>(std::floor(index));
    int upper = static_cast<int>(std::ceil(index));
    
    if (lower == upper) {
        return sorted[lower];
    }
    
    double weight = index - lower;
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

// Helper function to compute convex hull area
inline double computeHullArea(const std::vector<ContactPoint>& hull) {
    if (hull.size() < 3) return 0.0;
    
    double area = 0.0;
    int n = hull.size();
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += hull[i].x * hull[j].y;
        area -= hull[j].x * hull[i].y;
    }
    return std::abs(area) / 2.0;
}

// Helper function for convex hull (using Graham scan from Polygon.hpp)
inline std::vector<ContactPoint> computeConvexHull(std::vector<ContactPoint> points) {
    if (points.size() < 3) return points;
    
    int n = points.size(), k = 0;
    std::vector<ContactPoint> hull(2 * n);
    
    // Sort points lexicographically
    std::sort(points.begin(), points.end());
    
    // Cross product function
    auto cross = [](const ContactPoint& O, const ContactPoint& A, const ContactPoint& B) {
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    };
    
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

// Main function: find_best_inner_hull
struct InnerApproxHull {
    std::vector<ContactPoint> hull_vertices;
    std::vector<ContactPoint> inner_points;
    bool success;
};

inline InnerApproxHull find_best_inner_hull(
    const std::vector<ContactPoint>& feasible_points,
    const std::vector<ContactPoint>* sample_points = nullptr,
    double safe_margin = 0.05
) {
    InnerApproxHull result;
    result.success = false;
    
    if (feasible_points.size() < 3) {
        result.hull_vertices = feasible_points;
        result.inner_points = feasible_points;
        return result;
    }
    
    // Kernel Density Estimation
    try {
        GaussianKDE kde(feasible_points);
        std::vector<double> densities = kde.evaluate(feasible_points);
        
        // Try different percentile thresholds
        std::vector<double> percentiles = {20.0, 30.0, 40.0, 50.0};
        std::vector<ContactPoint> best_hull_vertices;
        std::vector<ContactPoint> best_inner_points;
        double best_area = 0.0;
        bool found_best = false;
        
        for (double p : percentiles) {
            double threshold = percentile(densities, p);
            
            // Filter points by density
            std::vector<ContactPoint> inner_points;
            for (size_t i = 0; i < feasible_points.size(); ++i) {
                if (densities[i] > threshold) {
                    inner_points.push_back(feasible_points[i]);
                }
            }
            
            // Filter by safety distance from sample_points
            if (sample_points != nullptr && !sample_points->empty()) {
                SimpleKDTree sample_tree(*sample_points);
                
                std::vector<ContactPoint> safe_points;
                for (const auto& point : inner_points) {
                    // double dist = sample_tree.query(point);
                    // if (dist > safe_margin) {
                        safe_points.push_back(point);
                    // }
                }
                
                if (!safe_points.empty()) {
                    inner_points = safe_points;
                }
            }
            
            // Compute convex hull
            if (inner_points.size() >= 3) {
                try {
                    std::vector<ContactPoint> hull = computeConvexHull(inner_points);
                    double area = computeHullArea(hull);
                    
                    if (area > best_area) {
                        best_area = area;
                        best_hull_vertices = hull;
                        best_inner_points = inner_points;
                        found_best = true;
                    }
                } catch (...) {
                    continue;
                }
            }
        }
        
        if (found_best) {
            result.hull_vertices = best_hull_vertices;
            result.inner_points = best_inner_points;
            result.success = true;
            return result;
        }
    } catch (...) {
        // KDE failed, fall through to fallback
    }
    
    // Fallback: use points near center
    Eigen::Vector2d center(0.0, 0.0);
    for (const auto& p : feasible_points) {
        center(0) += p.x;
        center(1) += p.y;
    }
    center /= feasible_points.size();
    
    // Compute distances and select closest points
    std::vector<std::pair<double, size_t>> distances;
    for (size_t i = 0; i < feasible_points.size(); ++i) {
        double dx = feasible_points[i].x - center(0);
        double dy = feasible_points[i].y - center(1);
        double dist = std::sqrt(dx * dx + dy * dy);
        distances.push_back({dist, i});
    }
    
    std::sort(distances.begin(), distances.end());
    
    int n_select = std::max(10, static_cast<int>(feasible_points.size() * 0.3));
    n_select = std::min(n_select, static_cast<int>(feasible_points.size()));
    
    std::vector<ContactPoint> inner_points;
    for (int i = 0; i < n_select; ++i) {
        inner_points.push_back(feasible_points[distances[i].second]);
    }
    
    if (inner_points.size() >= 3) {
        try {
            std::vector<ContactPoint> hull = computeConvexHull(inner_points);
            result.hull_vertices = hull;
            result.inner_points = inner_points;
            result.success = true;
            return result;
        } catch (...) {
            // Fall through
        }
    }
    
    result.hull_vertices = inner_points;
    result.inner_points = inner_points;
    return result;
}

#endif // INNER_APPROX_HULL_HPP






















// #pragma once
// #include <iostream>
// #include <vector>
// #include <Eigen/Dense>
// #include <algorithm>
// #include <numeric>
// #include <cmath>
// #include <limits>

// struct ContactPoint {
//     double x, y;

//     bool operator<(const ContactPoint& p) const {
//         return x < p.x || (x == p.x && y < p.y);
//     }

//     bool operator==(const ContactPoint& p) const {
//         return std::abs(x - p.x) < 1e-10 && std::abs(y - p.y) < 1e-10;
//     }
// };

// namespace inner_hull
// {

// struct InnerHullResult {
//     std::vector<ContactPoint> hull_vertices;  // CCW hull vertices (no repeated last vertex)
//     std::vector<ContactPoint> inner_points;   // selected inner points used for the best hull (or fallback selection)
//     bool used_kde = false;                    // true if KDE branch produced a best hull
//     bool success = false;                     // true if hull_vertices has >=3 points
// };

// // ---------- small utilities ----------
// inline double sqr(double v) { return v * v; }

// inline double dist2(const ContactPoint& a, const ContactPoint& b) {
//     return sqr(a.x - b.x) + sqr(a.y - b.y);
// }

// inline double dist(const ContactPoint& a, const ContactPoint& b) {
//     return std::sqrt(dist2(a, b));
// }

// inline std::vector<ContactPoint> unique_points(std::vector<ContactPoint> pts) {
//     std::sort(pts.begin(), pts.end());
//     pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
//     return pts;
// }

// inline ContactPoint mean_point(const std::vector<ContactPoint>& pts) {
//     ContactPoint c{0.0, 0.0};
//     if (pts.empty()) return c;
//     for (const auto& p : pts) { c.x += p.x; c.y += p.y; }
//     c.x /= static_cast<double>(pts.size());
//     c.y /= static_cast<double>(pts.size());
//     return c;
// }

// inline double percentile(std::vector<double> v, double pct) {
//     // pct in [0,100]. Uses linear interpolation.
//     if (v.empty()) return 0.0;
//     std::sort(v.begin(), v.end());
//     double pos = (pct / 100.0) * (static_cast<double>(v.size()) - 1.0);
//     std::size_t lo = static_cast<std::size_t>(std::floor(pos));
//     std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
//     double t = pos - static_cast<double>(lo);
//     if (hi >= v.size()) hi = v.size() - 1;
//     return (1.0 - t) * v[lo] + t * v[hi];
// }

// // ---------- convex hull (Monotone Chain) ----------
// inline double cross(const ContactPoint& O, const ContactPoint& A, const ContactPoint& B) {
//     // (A - O) x (B - O)
//     return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
// }

// inline std::vector<ContactPoint> convex_hull(std::vector<ContactPoint> pts) {
//     pts = unique_points(std::move(pts));
//     if (pts.size() < 3) return pts;

//     std::vector<ContactPoint> H;
//     H.reserve(pts.size() * 2);

//     // lower
//     for (const auto& p : pts) {
//         while (H.size() >= 2 && cross(H[H.size()-2], H[H.size()-1], p) <= 1e-12) H.pop_back();
//         H.push_back(p);
//     }
//     // upper
//     std::size_t lower_size = H.size();
//     for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
//         const auto& p = pts[static_cast<std::size_t>(i)];
//         while (H.size() > lower_size && cross(H[H.size()-2], H[H.size()-1], p) <= 1e-12) H.pop_back();
//         H.push_back(p);
//     }

//     if (!H.empty()) H.pop_back(); // last = first
//     return H;
// }

// inline double polygon_area(const std::vector<ContactPoint>& poly) {
//     // Shoelace area (absolute). poly assumed in CCW/CW without repeated last.
//     if (poly.size() < 3) return 0.0;
//     long double a = 0.0;
//     for (std::size_t i = 0; i < poly.size(); ++i) {
//         const auto& p = poly[i];
//         const auto& q = poly[(i + 1) % poly.size()];
//         a += static_cast<long double>(p.x) * static_cast<long double>(q.y)
//            - static_cast<long double>(q.x) * static_cast<long double>(p.y);
//     }
//     return static_cast<double>(std::abs(a) * 0.5L);
// }

// // ---------- KDE (manual gaussian kernel, 2D) ----------
// inline double estimate_bandwidth_silverman_2d(const std::vector<ContactPoint>& pts) {
//     // Simple Silverman-like rule using pooled stddev.
//     // h = 1.06 * sigma * n^{-1/5}
//     if (pts.size() < 2) return 1e-3;

//     ContactPoint c = mean_point(pts);
//     double sx2 = 0.0, sy2 = 0.0;
//     for (const auto& p : pts) {
//         sx2 += sqr(p.x - c.x);
//         sy2 += sqr(p.y - c.y);
//     }
//     sx2 /= static_cast<double>(pts.size() - 1);
//     sy2 /= static_cast<double>(pts.size() - 1);
//     double sigma = std::sqrt(0.5 * (sx2 + sy2));
//     if (!std::isfinite(sigma) || sigma < 1e-12) sigma = 1e-3;

//     double n = static_cast<double>(pts.size());
//     double h = 1.06 * sigma * std::pow(n, -0.2); // n^{-1/5}
//     if (!std::isfinite(h) || h < 1e-6) h = 1e-6;
//     return h;
// }

// inline std::vector<double> gaussian_kde_densities(const std::vector<ContactPoint>& pts, double h) {
//     // density_i = sum_j exp(-||pi-pj||^2/(2h^2)) (normalization not needed for ranking/percentiles)
//     std::vector<double> d(pts.size(), 0.0);
//     const double inv_2h2 = 1.0 / (2.0 * h * h);

//     for (std::size_t i = 0; i < pts.size(); ++i) {
//         double acc = 0.0;
//         for (std::size_t j = 0; j < pts.size(); ++j) {
//             const double r2 = dist2(pts[i], pts[j]);
//             acc += std::exp(-r2 * inv_2h2);
//         }
//         d[i] = acc;
//     }
//     return d;
// }

// // ---------- nearest distance to sample_points (bruteforce) ----------
// inline double nearest_distance_to_samples(
//     const ContactPoint& p,
//     const std::vector<ContactPoint>& samples)
// {
//     if (samples.empty()) return std::numeric_limits<double>::infinity();
//     double best = std::numeric_limits<double>::infinity();
//     for (const auto& s : samples) {
//         double ds = dist(p, s);
//         if (ds < best) best = ds;
//     }
//     return best;
// }

// // ---------- main algorithm ----------
// inline InnerHullResult find_best_inner_hull(
//     const std::vector<ContactPoint>& feasible_points_in,
//     const std::vector<ContactPoint>* sample_points = nullptr,
//     double safe_dist = 0.05)
// {
//     InnerHullResult out;

//     std::vector<ContactPoint> feasible = unique_points(feasible_points_in);
//     if (feasible.size() < 3) {
//         out.hull_vertices = feasible;
//         out.inner_points = feasible;
//         out.success = (out.hull_vertices.size() >= 3);
//         return out;
//     }

//     // KDE densities
//     const double h = estimate_bandwidth_silverman_2d(feasible);
//     std::vector<double> dens = gaussian_kde_densities(feasible, h);

//     std::vector<int> percentiles = {20, 30, 40, 50};

//     double best_area = -1.0;
//     std::vector<ContactPoint> best_hull;
//     std::vector<ContactPoint> best_inner;

//     for (int pct : percentiles) {
//         double thr = percentile(dens, static_cast<double>(pct));

//         // high density points
//         std::vector<ContactPoint> inner;
//         inner.reserve(feasible.size());
//         for (std::size_t i = 0; i < feasible.size(); ++i) {
//             if (dens[i] > thr) inner.push_back(feasible[i]);
//         }
//         inner = unique_points(std::move(inner));
//         if (inner.size() < 3) continue;

//         // safety filtering vs sample points (if provided)
//         if (sample_points && !sample_points->empty()) {
//             std::vector<ContactPoint> safe;
//             safe.reserve(inner.size());
//             for (const auto& p : inner) {
//                 double nd = nearest_distance_to_samples(p, *sample_points);
//                 if (nd > safe_dist) safe.push_back(p);
//             }
//             if (!safe.empty()) inner = unique_points(std::move(safe));
//             // if safe becomes <3, just skip this percentile (matches python intent more closely)
//             if (inner.size() < 3) continue;
//         }

//         // hull + area
//         std::vector<ContactPoint> hull = convex_hull(inner);
//         if (hull.size() < 3) continue;

//         double a = polygon_area(hull);
//         if (a > best_area) {
//             best_area = a;
//             best_hull = std::move(hull);
//             best_inner = std::move(inner);
//         }
//     }

//     if (!best_hull.empty()) {
//         out.hull_vertices = best_hull;
//         out.inner_points = best_inner;
//         out.used_kde = true;
//         out.success = (out.hull_vertices.size() >= 3);
//         return out;
//     }

//     // ----- fallback: center-near points (same as your python fallback) -----
//     ContactPoint c = mean_point(feasible);
//     std::vector<std::pair<double, ContactPoint>> dist_pts;
//     dist_pts.reserve(feasible.size());
//     for (const auto& p : feasible) {
//         dist_pts.push_back({dist2(p, c), p});
//     }
//     std::sort(dist_pts.begin(), dist_pts.end(),
//               [](const auto& a, const auto& b){ return a.first < b.first; });

//     std::size_t n_select = std::max<std::size_t>(10, static_cast<std::size_t>(std::floor(feasible.size() * 0.3)));
//     n_select = std::min(n_select, feasible.size());

//     std::vector<ContactPoint> inner;
//     inner.reserve(n_select);
//     for (std::size_t i = 0; i < n_select; ++i) inner.push_back(dist_pts[i].second);
//     inner = unique_points(std::move(inner));

//     out.inner_points = inner;

//     if (inner.size() >= 3) {
//         out.hull_vertices = convex_hull(inner);
//         out.success = (out.hull_vertices.size() >= 3);
//     } else {
//         out.hull_vertices = inner;
//         out.success = false;
//     }

//     return out;
// }

// } // namespace inner_hull