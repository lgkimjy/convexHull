import numpy as np
from scipy.spatial import ConvexHull, KDTree
from scipy.stats import gaussian_kde


def find_best_inner_hull(feasible_points, sample_points=None):
    """)
        Find the best inner hull based on Kernel Density Estimation
        Args:
            feasible_points: numpy array of feasible points
            sample_points: numpy array of sample points
        Returns:
            hull_vertices: numpy array of hull vertices
            inner_points: numpy array of inner points
    """
    if len(feasible_points) < 3:
        return feasible_points, None
    
    # Kernel Density Estimation : find the best inner hull based on Kernel Density Estimation (KDE)
    try:
        kde = gaussian_kde(feasible_points.T)
        densities = kde(feasible_points.T)
        
        # Try different percentile thresholds
        best_hull = None
        best_area = 0
        
        for percentile in [20, 30, 40, 50]:
            threshold = np.percentile(densities, percentile)
            high_density_indices = densities > threshold
            inner_points = feasible_points[high_density_indices]
            
            # Check the distance between inner_points and sample_points (if provided)
            if sample_points is not None and len(sample_points) > 0:
                sample_tree = KDTree(sample_points)
                
                # Filter out unsafe points
                safe_points = []
                for point in inner_points:
                    dist, _ = sample_tree.query([point], k=1)
                    if dist[0] > 0.05:  # Only keep points that are at least 0.05 units away from sample_points
                        safe_points.append(point)
                
                inner_points = np.array(safe_points) if safe_points else inner_points
            
            # Calculate Convex Hull
            if len(inner_points) >= 3:
                try:
                    hull = ConvexHull(inner_points)
                    if hull.area > best_area:
                        best_area = hull.area
                        best_hull = hull
                        best_inner_points = inner_points
                except:
                    continue
        
        if best_hull is not None:
            hull_vertices = best_inner_points[best_hull.vertices]
            return hull_vertices, best_inner_points
        else:
            # If failed, use points near the center
            center = np.mean(feasible_points, axis=0)
            distances = np.linalg.norm(feasible_points - center, axis=1)
            sorted_indices = np.argsort(distances)
            n_select = max(10, int(len(feasible_points) * 0.3))
            inner_points = feasible_points[sorted_indices[:n_select]]
            
            if len(inner_points) >= 3:
                hull = ConvexHull(inner_points)
                return inner_points[hull.vertices], inner_points
            else:
                return inner_points, inner_points
                
    except:
        # If Kernel Density fails, use a simple method
        center = np.mean(feasible_points, axis=0)
        distances = np.linalg.norm(feasible_points - center, axis=1)
        sorted_indices = np.argsort(distances)
        n_select = max(10, int(len(feasible_points) * 0.3))
        inner_points = feasible_points[sorted_indices[:n_select]]
        
        if len(inner_points) >= 3:
            hull = ConvexHull(inner_points)
            return inner_points[hull.vertices], inner_points
        else:
            return inner_points, inner_points