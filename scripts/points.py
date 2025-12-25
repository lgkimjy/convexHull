import matplotlib.pyplot as plt
import numpy as np
from InnerApprox import find_best_inner_hull

# txt 파일에서 데이터 읽기
def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as file:
        for line in file:
            # 각 줄을 쉼표로 분리하고 공백 제거
            # values = line.strip().split(',')
            values = line.strip().split()
            try:
                x.append(float(values[0]))  # x 좌표
                y.append(float(values[1]))  # y 좌표
            except (IndexError, ValueError):
                print(f"Error reading line in {filename}: {line}")
    return x, y


def read_data_with_average(filename):
    x, y, z = [], [], []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()
            try:
                x.append(float(values[3]))
                y.append(float(values[4]))
                z.append(float(values[5]))
            except (IndexError, ValueError):
                print(f"Error reading line in {filename}: {line}")
    return x, y, z

# 두 파일 읽기
path = '../data/left/'
file1 = path + 'sample_points.txt'
file2 = path + 'feasible_points.txt'
file3 = path + 'convex_hull_vertices.txt'
file4 = path + 'inner_approx_hull_vertices.txt'

x1, y1 = read_data(file1)
x2, y2 = read_data(file2)
x3, y3, z3 = read_data_with_average(file2)

plt.figure(figsize=(8, 8))
plt.axis('equal')
plt.gca().invert_xaxis()

sc1 = plt.scatter(y1, x1 , color='blue', label='sample_points : ' + str(len(x1)), alpha=0.6)
sc2 = plt.scatter(y2, x2, color='red', label='feasible_points : ' + str(len(x2)), alpha=0.6)

vertice_x, vertice_y = read_data(file3)
vertices = np.column_stack((vertice_y, vertice_x))
vertices = np.vstack([vertices, vertices[0]])
plt.plot(vertices[:, 0], vertices[:, 1], '-', color='orange', linewidth=4)

sc3 = plt.scatter(y3, x3, color='green', s=250)

inner_vertices, inner_points = find_best_inner_hull(np.column_stack((y2, x2)), np.column_stack((y1, x1)))
inner_vertices = np.vstack([inner_vertices, inner_vertices[0]])
plt.plot(inner_vertices[:, 0], inner_vertices[:, 1], '-', color='grey', linewidth=4)

inner_x, inner_y = read_data(file4)
inner_vertices = np.column_stack((inner_y, inner_x))
inner_vertices = np.vstack([inner_vertices, inner_vertices[0]])
plt.plot(inner_vertices[:, 0], inner_vertices[:, 1], '-', color='black', linewidth=4)

x6, y6 = read_data(path + 'support_polygon_vertices.txt')
vertices_support = np.column_stack((y6, x6))
vertices_support = np.vstack([vertices_support, vertices_support[0]])
plt.plot(vertices_support[:, 0], vertices_support[:, 1], '-', color='blue', linewidth=4)

sc6 = plt.quiver(0, 0, 0, 1, color='red', scale=10, label='X-axis')  # x축 (오른쪽)
sc7 = plt.quiver(0, 0, -1, 0, color='green', scale=10, label='Y-axis')  # y축 (위쪽)

# sc3 = plt.scatter(0.3, 0.7, color='cyan', alpha=1, s=250)
sc3 = plt.scatter(0.0, 0.7, color='cyan', alpha=1, s=250)
# sc3 = plt.scatter(0.0, 0.75, color='cyan', alpha=1, s=100)
# sc3 = plt.scatter(-0.3, 0.8, color='cyan', alpha=1, s=100)
# sc3 = plt.scatter(0.15, 0.7, color='cyan', alpha=1, s=100) # scene_b

handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Sample sets : '+str(len(x1))),
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='red', markersize=8, label='Feasible sets : '+str(len(x2))),
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Feasible CoM sets'),
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='green', markersize=8, label='Mean of Feasible sets'),
    # plt.Line2D([], [], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='left_hand_target')
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='Manipulation Target'),
    plt.Line2D([], [], color='orange', linewidth=2, label='Convex Hull'),
    # plt.Line2D([], [], color='yellow', linewidth=2, label='inner_convex_hull'),
    plt.Line2D([], [], color='black', linewidth=2, label='Inner Convex Hull'),
]

# 그래프 설정
font_size = 13
plt.xlabel('x-axis (m)', fontsize=font_size)
plt.ylabel('y-axis (m)', fontsize=font_size)
# plt.title('Scatter Plot of X, Y Coordinates of Sample and Feasible Points')
plt.legend(handles=handles, loc='best', fontsize=font_size)
plt.grid(True)
plt.tight_layout()
plt.savefig(path + 'ConvexHulls.pdf')

# 그래프 표시
plt.show()