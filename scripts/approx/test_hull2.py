import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, KDTree
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

def read_data(filename):
    """데이터 파일에서 x, y 좌표 읽기"""
    x, y = [], []
    try:
        with open(filename, 'r') as file:
            for line in file:
                if ',' in line:
                    values = [v.strip() for v in line.strip().split(',')]
                else:
                    values = line.strip().split()
                
                if len(values) >= 2:
                    try:
                        x.append(float(values[0]))
                        y.append(float(values[1]))
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    return np.array(x), np.array(y)

def plot_all_methods_comparison(feasible_points, sample_points=None, filename="comparison"):
    """
    원래 있던 4가지 method + 안전 조건으로 비교 시각화
    """
    if len(feasible_points) < 3:
        print("Error: Need at least 3 feasible points")
        return
    
    # sample_points가 None이면 빈 배열로
    if sample_points is None:
        sample_points = np.array([])
    
    # KDTree for sample points (빠른 거리 계산)
    if len(sample_points) > 0:
        sample_tree = KDTree(sample_points)
        print(f"Sample points tree built with {len(sample_points)} points")
    else:
        sample_tree = None
        print("No sample points provided")
    
    def is_safe_point(point, safe_margin=0.05):
        """점이 sample_points에서 안전한 거리에 있는지 확인"""
        if sample_tree is None or len(sample_points) == 0:
            return True
        dist, _ = sample_tree.query([point], k=1)
        return dist[0] > safe_margin
    
    # Method 1: Alpha-shape 기반 내부 근사 (원래 있던 method)
    def alpha_shape_inner(points, alpha=0.3):
        if len(points) < 4:
            return points
        
        try:
            tri = Delaunay(points)
            valid_points = []
            
            for simplex in tri.simplices:
                triangle = points[simplex]
                
                # 삼각형 면적 계산
                area = 0.5 * abs(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]))
                if area < 1e-10:
                    continue
                
                # 삼각형 변 길이
                a = np.linalg.norm(triangle[0] - triangle[1])
                b = np.linalg.norm(triangle[1] - triangle[2])
                c = np.linalg.norm(triangle[2] - triangle[0])
                
                # 외접원 반지름
                radius = (a * b * c) / (4 * area)
                
                if radius < 1.0 / alpha:
                    # 삼각형 내부에 점 생성
                    for _ in range(3):
                        r1, r2 = np.random.random(2)
                        if r1 + r2 > 1:
                            r1, r2 = 1 - r1, 1 - r2
                        point = (1 - r1 - r2) * triangle[0] + r1 * triangle[1] + r2 * triangle[2]
                        if is_safe_point(point):
                            valid_points.append(point)
            
            return np.array(valid_points) if valid_points else points
        except:
            return points
    
    # Method 2: 그리드 기반 내부 근사 (원래 있던 method)
    def grid_based_inner(points, grid_size=0.05):
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        x_grid = np.arange(x_min, x_max, grid_size)
        y_grid = np.arange(y_min, y_max, grid_size)
        
        valid_points = []
        
        for x in x_grid:
            for y in y_grid:
                point = np.array([x, y])
                
                # 안전 확인
                if not is_safe_point(point):
                    continue
                
                # 간단한 내부 판별: 주변에 충분히 많은 점이 있는지
                distances = np.linalg.norm(points - point, axis=1)
                k = min(20, len(points))
                nearest_indices = np.argsort(distances)[:k]
                nearest_points = points[nearest_indices]
                
                # 평균 거리가 너무 크면 바깥쪽으로 판단
                mean_dist = np.mean(distances[nearest_indices])
                if mean_dist < grid_size * 3:  # 임계값
                    valid_points.append(point)
        
        return np.array(valid_points) if valid_points else points
    
    # Method 3: Kernel Density 기반 내부 근사 (원래 있던 method - 복원)
    def kernel_density_inner(points, threshold_percentile=100):
        if len(points) < 10:
            return points
        
        try:
            # 커널 밀도 추정
            kde = gaussian_kde(points.T)
            densities = kde(points.T)
            
            # 밀도 임계값 설정
            threshold = np.percentile(densities, threshold_percentile)
            
            # 높은 밀도 점 선택 (안전 조건 추가)
            print(threshold)
            print(densities)
            high_density_indices = densities > threshold
            high_density_points = points[high_density_indices]
            
            # 안전한 점들만 필터링
            safe_points = []
            for point in high_density_points:
                if is_safe_point(point):
                    safe_points.append(point)
            
            return np.array(safe_points) if safe_points else points
        except:
            return points
    
    # Method 4: Convex Inner Polygon (원래 있던 method)
    def convex_inner_polygon(points, n_vertices=12):
        if len(points) < n_vertices:
            return points
        
        # 데이터의 중심
        center = np.mean(points, axis=0)
        
        # 중심에서 각 점까지의 각도 계산
        vectors = points - center
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        
        # 각도별로 점 정렬
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        # 각 구간에서 중심에 가장 가까운 점 선택 (안전 조건 추가)
        step = len(sorted_points) // n_vertices
        inner_vertices = []
        
        for i in range(n_vertices):
            idx = i * step % len(sorted_points)
            segment_start = idx
            segment_end = min(idx + step, len(sorted_points))
            segment = sorted_points[segment_start:segment_end]
            
            if len(segment) > 0:
                # 안전한 점들 중에서 중심에 가장 가까운 점 선택
                safe_segment = []
                for point in segment:
                    if is_safe_point(point):
                        safe_segment.append(point)
                
                if safe_segment:
                    safe_segment = np.array(safe_segment)
                    distances = np.linalg.norm(safe_segment - center, axis=1)
                    closest_idx = np.argmin(distances)
                    inner_vertices.append(safe_segment[closest_idx])
        
        return np.array(inner_vertices) if inner_vertices else points
    
    # 각 method로 내부 근사 계산 (원래 4가지 method)
    methods = {
        "Alpha-shape (α=0.3)": alpha_shape_inner(feasible_points, alpha=0.3),
        "Grid-based (0.05)": grid_based_inner(feasible_points, grid_size=0.05),
        "Kernel Density (30%)": kernel_density_inner(feasible_points, threshold_percentile=30),
        "Convex Polygon (12v)": convex_inner_polygon(feasible_points, n_vertices=12)
    }
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    colors = ['red', 'green', 'blue', 'purple']
    
    for idx, (method_name, inner_points) in enumerate(methods.items()):
        ax = axes[idx]
        
        # 배경: sample points (있는 경우)
        if len(sample_points) > 0:
            ax.scatter(sample_points[:, 0], sample_points[:, 1], 
                      c='lightgray', alpha=0.15, s=8, label='All samples', zorder=1)
        
        # feasible points
        ax.scatter(feasible_points[:, 0], feasible_points[:, 1], 
                  c='skyblue', alpha=0.4, s=35, label='Feasible', zorder=2)
        
        # 내부 근사 점들
        if len(inner_points) > 0:
            ax.scatter(inner_points[:, 0], inner_points[:, 1], 
                      c=colors[idx], alpha=0.7, s=50, label='Inner points', zorder=3)
            
            # Convex hull 그리기
            if len(inner_points) >= 3:
                try:
                    hull = ConvexHull(inner_points)
                    hull_vertices = inner_points[hull.vertices]
                    hull_closed = np.vstack([hull_vertices, hull_vertices[0]])
                    
                    ax.plot(hull_closed[:, 0], hull_closed[:, 1], 
                           c=colors[idx], linewidth=2.5, linestyle='-',
                           label=f'Hull ({len(hull.vertices)}v)', zorder=4)
                    ax.fill(hull_closed[:, 0], hull_closed[:, 1], 
                           alpha=0.2, color=colors[idx], zorder=4)
                    
                    # 면적 정보
                    area = hull.area
                    ax.text(0.02, 0.98, f'Area: {area:.4f}', 
                           transform=ax.transAxes, fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # 점 개수 정보
                    ax.text(0.02, 0.90, f'Points: {len(inner_points)}', 
                           transform=ax.transAxes, fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except Exception as e:
                    print(f"Error drawing hull for {method_name}: {e}")
        
        ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(loc='best', fontsize=7)
        ax.axis('equal')
    
    plt.suptitle('Inner Approximation Methods (Original 4 Methods)\nAvoiding Sample Points', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 저장
    output_file = f"{filename}_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved as: {output_file}")
    
    plt.show()
    
    return methods

def find_best_inner_hull(feasible_points, sample_points=None):
    """
    가장 좋은 내부 hull 찾기 (Kernel Density 기반)
    """
    if len(feasible_points) < 3:
        return feasible_points, None
    
    # Kernel Density로 높은 밀도 영역 찾기
    try:
        kde = gaussian_kde(feasible_points.T)
        densities = kde(feasible_points.T)
        
        # 다양한 임계값 시도
        best_hull = None
        best_area = 0
        
        for percentile in [20, 30, 40, 50]:
            threshold = np.percentile(densities, percentile)
            high_density_indices = densities > threshold
            inner_points = feasible_points[high_density_indices]
            
            # sample_points와의 거리 확인 (있는 경우)
            if sample_points is not None and len(sample_points) > 0:
                sample_tree = KDTree(sample_points)
                
                # 안전한 점들만 필터링
                safe_points = []
                for point in inner_points:
                    dist, _ = sample_tree.query([point], k=1)
                    if dist[0] > 0.05:  # 0.05 거리 이상 떨어진 점만
                        safe_points.append(point)
                
                inner_points = np.array(safe_points) if safe_points else inner_points
            
            # Convex hull 계산
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
            # 실패 시 중심 근처 점들 사용
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
        # Kernel Density 실패 시 간단한 방법
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

def main():
    """메인 함수 - 원래 있던 모든 method 포함"""
    print("=== Inner Approximation Program ===")
    print("Using original 4 methods with safety constraints")
    
    # 파일 읽기
    sample_file = "../../data/left/sample_points.txt" #input("Sample points file (press Enter if none): ").strip()
    feasible_file = "../../data/left/feasible_points.txt" #input("Feasible points file: ").strip()
    
    if not feasible_file:
        print("Feasible points file is required!")
        return
    
    # 데이터 읽기
    feasible_x, feasible_y = read_data(feasible_file)
    if len(feasible_x) == 0:
        print("No feasible points found!")
        return
    
    feasible_points = np.column_stack([feasible_x, feasible_y])
    print(f"Loaded {len(feasible_points)} feasible points")
    
    sample_points = None
    if sample_file:
        sample_x, sample_y = read_data(sample_file)
        if len(sample_x) > 0:
            sample_points = np.column_stack([sample_x, sample_y])
            print(f"Loaded {len(sample_points)} sample points")
    
    # 1. 원래 4가지 method 비교
    print("\nComparing original 4 methods...")
    methods = plot_all_methods_comparison(feasible_points, sample_points, "inner_approx")
    
    # 2. Kernel Density 기반 최적 hull 찾기
    print("\nFinding best inner hull using Kernel Density...")
    best_hull_vertices, best_inner_points = find_best_inner_hull(feasible_points, sample_points)
    
    # 최적 결과 시각화
    plt.figure(figsize=(12, 10))
    
    # 배경: sample points
    if sample_points is not None and len(sample_points) > 0:
        plt.scatter(sample_points[:, 0], sample_points[:, 1], 
                   c='lightgray', alpha=0.1, s=10, label='Sample points', zorder=1)
    
    # feasible points
    plt.scatter(feasible_points[:, 0], feasible_points[:, 1], 
               c='skyblue', alpha=0.3, s=40, label='Feasible points', zorder=2)
    
    # best inner points
    if len(best_inner_points) > 0:
        plt.scatter(best_inner_points[:, 0], best_inner_points[:, 1], 
                   c='orange', alpha=0.6, s=60, label='Kernel Density points', zorder=3)
    
    # best hull
    if len(best_hull_vertices) >= 3:
        hull_closed = np.vstack([best_hull_vertices, best_hull_vertices[0]])
        plt.plot(hull_closed[:, 0], hull_closed[:, 1], 
                'red', linewidth=3, label='Best inner hull', zorder=4)
        plt.fill(hull_closed[:, 0], hull_closed[:, 1], 
                alpha=0.2, color='red', zorder=4)
        
        # 꼭지점 강조
        plt.scatter(best_hull_vertices[:, 0], best_hull_vertices[:, 1], 
                   c='red', s=100, edgecolors='black', linewidth=2, 
                   zorder=5, label=f'Hull vertices ({len(best_hull_vertices)})')
        
        # hull 정보
        try:
            hull = ConvexHull(best_hull_vertices)
            area = hull.area
            info_text = f"""Best Inner Hull (Kernel Density):
            • Vertices: {len(best_hull_vertices)}
            • Area: {area:.4f}
            • Inner points: {len(best_inner_points)}"""
            
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
    
    plt.title('Best Inner Approximation Hull (Kernel Density Method)\nAvoiding Sample Points', 
              fontsize=14, fontweight='bold')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.axis('equal')
    plt.tight_layout()
    
    plt.savefig("best_kernel_hull.png", dpi=150, bbox_inches='tight')
    print("Best hull plot saved as: best_kernel_hull.png")
    plt.show()
    
    # 각 method 결과 저장 옵션
    print("\n=== Save Results ===")
    
    # Kernel Density 결과 저장
    if len(best_hull_vertices) > 0:
        save = input("Save Kernel Density hull vertices? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Filename (default=kernel_hull_vertices.txt): ").strip()
            if not filename:
                filename = "kernel_hull_vertices.txt"
            np.savetxt(filename, best_hull_vertices, fmt='%.6f')
            print(f"Hull vertices saved to: {filename}")
    
    if len(best_inner_points) > 0:
        save = input("Save Kernel Density inner points? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Filename (default=kernel_inner_points.txt): ").strip()
            if not filename:
                filename = "kernel_inner_points.txt"
            np.savetxt(filename, best_inner_points, fmt='%.6f')
            print(f"Inner points saved to: {filename}")
    
    # 모든 method 결과 저장
    save_all = input("\nSave results from all 4 methods? (y/n): ").strip().lower()
    if save_all == 'y':
        for method_name, points in methods.items():
            if len(points) > 0:
                # 파일명 생성
                safe_name = method_name.replace('(', '').replace(')', '').replace(' ', '_').replace('=', '')
                filename = f"{safe_name}_points.txt"
                np.savetxt(filename, points, fmt='%.6f')
                print(f"  - {method_name}: {len(points)} points -> {filename}")
    
    print("\nDone!")

# 테스트 데이터 생성 (원래 샘플과 유사하게)
def create_test_data():
    """테스트용 데이터 생성 (원래 예제와 유사하게)"""
    np.random.seed(42)
    
    # Feasible points: 오목한 모양 (별 모양)
    n_feasible = 200
    t = np.linspace(0, 2*np.pi, n_feasible)
    r = 1 + 0.4 * np.cos(5*t)  # 별 모양
    feasible_x = r * np.cos(t) + np.random.normal(0, 0.05, n_feasible)
    feasible_y = r * np.sin(t) + np.random.normal(0, 0.05, n_feasible)
    
    # Sample points: feasible + infeasible
    n_sample = 300
    angles = np.random.uniform(0, 2*np.pi, n_sample)
    radii = np.random.uniform(0.2, 1.5, n_sample)
    sample_x = radii * np.cos(angles) + np.random.normal(0, 0.1, n_sample)
    sample_y = radii * np.sin(angles) + np.random.normal(0, 0.1, n_sample)
    
    feasible_points = np.column_stack([feasible_x, feasible_y])
    sample_points = np.column_stack([sample_x, sample_y])
    
    np.savetxt("feasible_points.txt", feasible_points, fmt='%.6f')
    np.savetxt("sample_points.txt", sample_points, fmt='%.6f')
    
    print("Test data created:")
    print(f"- feasible_points.txt: {len(feasible_points)} points")
    print(f"- sample_points.txt: {len(sample_points)} points")
    
    return feasible_points, sample_points

if __name__ == "__main__":
    print("=== Inner Approximation with Original 4 Methods ===")
    print("1. Use existing files")
    print("2. Create test data and run")
    
    choice = input("Choose (1 or 2): ").strip()
    
    if choice == "2":
        print("\nCreating test data (concave shape with outliers)...")
        feasible_pts, sample_pts = create_test_data()
        
        # 모든 method 비교
        methods = plot_all_methods_comparison(feasible_pts, sample_pts, "test_comparison")
        
        # 최적 hull 찾기
        best_hull, best_points = find_best_inner_hull(feasible_pts, sample_pts)
        
        print(f"\nBest hull found with {len(best_hull)} vertices")
        if len(best_hull) >= 3:
            hull = ConvexHull(best_hull)
            print(f"Hull area: {hull.area:.4f}")
    else:
        main()