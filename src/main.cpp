#include "main.hpp"
#include <fstream>
#include <sstream>

// Read feasible points from file (format: x y z x y z per line, use first x y)
std::vector<ContactPoint> readFeasiblePoints(const std::string& filename) {
    std::vector<ContactPoint> points;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return points;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        double x, y, z1, x2, y2, z2;
        
        if (iss >> x >> y >> z1 >> x2 >> y2 >> z2) {
            points.push_back({x, y});
        }
    }
    
    file.close();
    return points;
}

// Read sample points from file (format: x y z per line, use first x y)
std::vector<ContactPoint> readSamplePoints(const std::string& filename) {
    std::vector<ContactPoint> points;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return points;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        double x, y, z;
        
        if (iss >> x >> y >> z) {
            points.push_back({x, y});
        }
    }
    
    file.close();
    return points;
}

// Write vertices to file (format: x y per line)
void writeVerticesToFile(const std::vector<ContactPoint>& vertices, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    for (const auto& v : vertices) {
        file << v.x << " " << v.y << std::endl;
    }
    
    file.close();
    std::cout << "Saved " << vertices.size() << " vertices to " << filename << std::endl;
}

int main()
{
    // Read data files
    std::string dataDir = "../data/left";
    std::string feasibleFile = dataDir + "/feasible_points.txt";
    std::string sampleFile = dataDir + "/sample_points.txt";
    
    std::vector<ContactPoint> feasible_points = readFeasiblePoints(feasibleFile);
    std::vector<ContactPoint> sample_points = readSamplePoints(sampleFile);
    
    if (feasible_points.empty()) {
        std::cerr << "Error: No feasible points loaded!" << std::endl;
        return 1;
    }
    
    // Compute Support Polygon vertices
    SupportPolygon supportPolygon(feasible_points);
    std::vector<ContactPoint> support_vertices = supportPolygon.getVertices();
    std::cout << "Support Polygon vertices (" << support_vertices.size() << " points):" << std::endl;
        
    // Save Support Polygon vertices to file
    std::string supportOutputFile = dataDir + "/convex_hull_vertices.txt";
    writeVerticesToFile(support_vertices, supportOutputFile);
    
    // Compute Inner Approximate Hull vertices
    InnerApproxHull innerResult = find_best_inner_hull(feasible_points, &sample_points, 0.05);
    
    std::vector<ContactPoint> inner_vertices;
    inner_vertices = innerResult.hull_vertices;
    std::cout << "Inner Approximate Hull vertices (" << inner_vertices.size() << " points):" << std::endl;
    std::cout << "Inner points used: " << innerResult.inner_points.size() << std::endl;
    
    // Save Inner Approximate Hull vertices to file
    std::string innerOutputFile = dataDir + "/inner_approx_hull_vertices.txt";
    writeVerticesToFile(inner_vertices, innerOutputFile);
    
    return 0;
}   