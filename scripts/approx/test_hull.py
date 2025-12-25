import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon, Point
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def read_data(filename):
    """데이터 파일에서 x, y 좌표 읽기"""
    x, y = [], []
    with open(filename, 'r') as file:
        for line in file:
            if ',' in line:
                values = line.strip().split(',')
            else:
                values = line.strip().split()
            
            try:
                if len(values) >= 2:
                    x.append(float(values[0]))
                    y.append(float(values[1]))
            except (IndexError, ValueError):
                continue
    return np.array(x), np.array(y)

def get_convex_hull(points):
    """볼록 껍질 계산"""
    if len(points) < 3:
        return None
    try:
        return ConvexHull(points)
    except:
        return None

def alpha_shape_inner_approximation(points, alpha=0.5):
    """
    Alpha-shape 기반 내부 근사
    alpha 값이 클수록 더 convex에 가까워짐
    """
    if len(points) < 4:
        return points
    
    # Delaunay 삼각분할
    tri = Delaunay(points)
    
    # 외접원 반지름이 alpha 이하인 삼각형만 선택
    valid_triangles = []
    for simplex in tri.simplices:
        # 삼각형의 세 점
        triangle = points[simplex]
        
        # 외접원 반지름 계산
        a = np.linalg.norm(triangle[0] - triangle[1])
        b = np.linalg.norm(triangle[1] - triangle[2])
        c = np.linalg.norm(triangle[2] - triangle[0])
        
        # 외접원 반지름 (삼각형 공식)
        radius = (a * b * c) / (4 * triangle_area(triangle))
        
        if radius < 1/alpha:  # alpha-shape 조건
            valid_triangles.append(simplex)
    
    if not valid_triangles:
        return points
    
    # 유효한 삼각형들의 내부 점 생성
    inner_points = []
    for simplex in valid_triangles:
        triangle = points[simplex]
        
        # 삼각형 내부에 균일하게 점 생성
        for _ in range(5):  # 각 삼각형당 5개 점
            r1, r2 = np.random.random(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            
            point = (1 - r1 - r2) * triangle[0] + r1 * triangle[1] + r2 * triangle[2]
            inner_points.append(point)
    
    return np.array(inner_points)

def triangle_area(triangle):
    """삼각형 면적 계산"""
    return 0.5 * abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))

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

def convex_inner_polygon(points, n_vertices=8):
    """
    데이터를 완전히 포함하는 내부 볼록 다각형 생성
    """
    # 데이터의 중심
    center = np.mean(points, axis=0)
    
    # 중심에서 각 점까지의 각도 계산
    vectors = points - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # 각도별로 점 정렬
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    # 균일한 간격으로 꼭지점 선택
    step = len(sorted_points) // n_vertices
    inner_vertices = []
    
    for i in range(n_vertices):
        idx = i * step % len(sorted_points)
        # 해당 구간에서 중심에 가장 가까운 점 선택
        segment_start = idx
        segment_end = min(idx + step, len(sorted_points))
        segment = sorted_points[segment_start:segment_end]
        
        if len(segment) > 0:
            # 중심까지의 거리 계산
            distances = np.linalg.norm(segment - center, axis=1)
            closest_idx = np.argmin(distances)
            inner_vertices.append(segment[closest_idx])
    
    return np.array(inner_vertices)

def plot_inner_approximations(x, y, filename="inner_approximation"):
    """다양한 내부 근사 방법 비교 시각화"""
    points = np.column_stack((x, y))
    
    # 다양한 방법으로 내부 근사 계산
    methods = {
        "Alpha-shape": alpha_shape_inner_approximation(points, alpha=0.3),
        "Grid-based": grid_based_inner_approximation(points, grid_size=0.05),
        "Kernel Density": kernel_density_inner_approximation(points, threshold_percentile=30),
        "Convex Inner Polygon": convex_inner_polygon(points, n_vertices=12)
    }
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (method_name, inner_points) in enumerate(methods.items()):
        ax = axes[idx]
        
        # 원본 데이터
        ax.scatter(x, y, c='lightblue', alpha=0.3, s=20, label='Original Data')
        
        # 내부 근사 점
        if len(inner_points) > 0:
            ax.scatter(inner_points[:, 0], inner_points[:, 1], 
                      c='red', alpha=0.6, s=30, label='Inner Approximation')
            
            # 내부 근사의 convex hull
            if len(inner_points) >= 3:
                try:
                    inner_hull = ConvexHull(inner_points)
                    hull_points = inner_points[inner_hull.vertices]
                    hull_points_closed = np.vstack([hull_points, hull_points[0]])
                    
                    ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 
                           'g-', linewidth=2, label='Inner Convex Hull')
                    ax.fill(hull_points_closed[:, 0], hull_points_closed[:, 1], 
                           alpha=0.2, color='green')
                except:
                    pass
        
        ax.set_title(f'{method_name}\nPoints: {len(inner_points)}', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(loc='best', fontsize=8)
    
    plt.suptitle('Comparison of Inner Approximation Methods', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 저장
    output_filename = f"{filename}_inner_approximation.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    
    plt.show()

def interactive_inner_approximation(x, y):
    """파라미터 조정 가능한 대화형 내부 근사"""
    points = np.column_stack((x, y))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 원본 데이터와 convex hull
    ax1.scatter(x, y, c='blue', alpha=0.3, s=20)
    hull = get_convex_hull(points)
    if hull is not None:
        hull_points = points[hull.vertices]
        hull_points_closed = np.vstack([hull_points, hull_points[0]])
        ax1.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 
                'r-', linewidth=2, label='Outer Convex Hull')
        ax1.fill(hull_points_closed[:, 0], hull_points_closed[:, 1], 
                alpha=0.1, color='red')
    
    ax1.set_title(f'Original Data (Convex Hull)\nPoints: {len(points)}', fontsize=12)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    
    # 내부 근사 (파라미터 조정 가능)
    # Alpha-shape 기반 (기본값)
    inner_points = alpha_shape_inner_approximation(points, alpha=0.3)
    
    ax2.scatter(x, y, c='lightblue', alpha=0.2, s=15, label='Original')
    ax2.scatter(inner_points[:, 0], inner_points[:, 1], 
               c='red', alpha=0.7, s=30, label='Inner Points')
    
    if len(inner_points) >= 3:
        inner_hull = ConvexHull(inner_points)
        inner_hull_points = inner_points[inner_hull.vertices]
        inner_hull_closed = np.vstack([inner_hull_points, inner_hull_points[0]])
        
        ax2.plot(inner_hull_closed[:, 0], inner_hull_closed[:, 1], 
                'g-', linewidth=2, label='Inner Convex Hull')
        ax2.fill(inner_hull_closed[:, 0], inner_hull_closed[:, 1], 
                alpha=0.3, color='green')
    
    ax2.set_title(f'Inner Approximation\nInner Points: {len(inner_points)}', fontsize=12)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend()
    
    plt.suptitle('Interactive Inner Approximation', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return inner_points

def save_inner_points(inner_points, filename="inner_points.txt"):
    """내부 근사 점들을 파일로 저장"""
    np.savetxt(filename, inner_points, fmt='%.6f', delimiter=', ')
    print(f"Inner points saved to: {filename}")
    return inner_points

def main():
    """메인 함수"""
    # filename = input("데이터 파일명을 입력하세요 (예: data.txt): ").strip()
    filename = "../../data/left/feasible_points.txt"
    
    if not filename:
        print("샘플 데이터 생성 중...")
        # 비-convex 샘플 데이터 생성 (별 모양)
        np.random.seed(42)
        t = np.linspace(0, 2*np.pi, 200)
        r = 1 + 0.5 * np.cos(5*t)  # 별 모양
        x = r * np.cos(t) + np.random.normal(0, 0.1, len(t))
        y = r * np.sin(t) + np.random.normal(0, 0.1, len(t))
        
        # 추가 노이즈 포인트
        x_noise = np.random.uniform(-1.5, 1.5, 150)
        y_noise = np.random.uniform(-1.5, 1.5, 150)
        
        # 거리 필터링 (중심에서 너무 먼 점 제거)
        dist = np.sqrt(x_noise**2 + y_noise**2)
        mask = dist < 1.5
        x = np.concatenate([x, x_noise[mask]])
        y = np.concatenate([y, y_noise[mask]])
        
        # 샘플 데이터 저장
        with open("non_convex_sample.txt", "w") as f:
            for xi, yi in zip(x, y):
                f.write(f"{xi:.6f} {yi:.6f}\n")
        filename = "non_convex_sample.txt"
        print(f"샘플 데이터 생성됨: {filename}")
    else:
        try:
            x, y = read_data(filename)
            if len(x) == 0:
                print("데이터를 읽을 수 없습니다.")
                return
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {filename}")
            return
    
    print(f"\n데이터 로드 완료: {len(x)} points")
    
    # 방법 선택
    print("\n=== 내부 근사 방법 선택 ===")
    print("1. Alpha-shape 기반 (기본값)")
    print("2. Grid 기반")
    print("3. Kernel Density 기반")
    print("4. Convex Inner Polygon")
    print("5. 모든 방법 비교")
    print("6. 대화형 모드")
    
    choice = input("선택 (1-6, 기본=1): ").strip()
    
    points = np.column_stack((x, y))
    
    if choice == "2":
        grid_size = float(input("Grid 크기 입력 (기본=0.05): ") or "0.05")
        inner_points = grid_based_inner_approximation(points, grid_size)
    elif choice == "3":
        threshold = float(input("밀도 임계값 퍼센타일 (기본=30): ") or "30")
        inner_points = kernel_density_inner_approximation(points, threshold)
    elif choice == "4":
        n_vertices = int(input("다각형 꼭지점 수 (기본=12): ") or "50")
        inner_points = convex_inner_polygon(points, 50)
    elif choice == "5":
        plot_inner_approximations(x, y, filename.split('.')[0])
        return
    elif choice == "6":
        inner_points = interactive_inner_approximation(x, y)
    else:  # 기본값: Alpha-shape
        alpha = float(input("Alpha 값 입력 (작을수록 더 concave, 기본=0.3): ") or "0.3")
        inner_points = alpha_shape_inner_approximation(points, alpha)
    
    # 결과 저장
    save_choice = input("\n내부 근사 점들을 저장하시겠습니까? (y/n): ").strip().lower()
    if save_choice == 'y':
        save_filename = input("저장할 파일명 (기본=inner_points.txt): ").strip()
        if not save_filename:
            save_filename = "inner_points.txt"
        save_inner_points(inner_points, save_filename)
    
    print(f"\n내부 근사 완료: {len(inner_points)} points")

if __name__ == "__main__":
    main()