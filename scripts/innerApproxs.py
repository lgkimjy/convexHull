import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull

def kernel_density_inner_approximation(points, threshold_percentile=25):
    """
    커널 밀도 추정 기반 내부 근사
    높은 밀도 영역 선택
    """
    from scipy.stats import gaussian_kde
    
    # 커널 밀도 추정
    kde = gaussian_kde(points.T)
    densities = kde(points.T)
    
    # 밀도 임계값 설정 (상위 threshold_percentile%)
    threshold = np.percentile(densities, 10)
    
    # 높은 밀도 점 선택
    high_density_indices = densities > threshold
    inner_points = points[high_density_indices]
    
    return inner_points


def is_point_in_polygon(point, polygon):
    """점이 다각형 내부에 있는지 확인 (ray casting 알고리즘)"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def grid_based_inner_approximation(points, grid_size=0.1):
    """
    그리드 기반 내부 근사
    """
    # 경계 상자
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # 그리드 생성
    x_grid = np.arange(x_min, x_max, grid_size)
    y_grid = np.arange(y_min, y_max, grid_size)
    
    inner_points = []
    
    # 각 그리드 포인트가 데이터 내부에 있는지 확인
    for x in x_grid:
        for y in y_grid:
            grid_point = np.array([x, y])
            
            # 주변 점들과의 거리 계산
            distances = np.linalg.norm(points - grid_point, axis=1)
            
            # 최근접 k개 점 찾기
            k = min(10, len(points))
            nearest_indices = np.argsort(distances)[:k]
            nearest_points = points[nearest_indices]
            
            # 주변 점들의 convex hull 계산
            if len(nearest_points) >= 3:
                try:
                    local_hull = ConvexHull(nearest_points)
                    hull_points = nearest_points[local_hull.vertices]
                    
                    # 그리드 점이 local hull 내부에 있는지 확인
                    if is_point_in_polygon(grid_point, hull_points):
                        inner_points.append(grid_point)
                except:
                    continue
    
    return np.array(inner_points)


if __name__ == "__main__":
    data_dir = "../data/left/"
    sample_file = data_dir + "sample_points.txt"
    feasible_file = data_dir + "feasible_points.txt"
    sample_points = np.loadtxt(sample_file)
    feasible_points = np.loadtxt(feasible_file)
    inner_points = grid_based_inner_approximation(feasible_points, grid_size=0.05)
    print(inner_points)