import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# mpl.rcParams.update({
#     "text.usetex": True,          # LaTeX 엔진 사용
#     "font.family": "serif",       # 기본 serif
#     # "font.serif": ["Times"],      # Times 계열 (시스템/TeX에 따라 대체될 수 있음)
#     # "mathtext.fontset": "cm",     # 수식 폰트(Computer Modern)
#     # "legend.fontsize": 14,
# })

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
file1 = 'sample_points.txt'
file2 = 'feasible_points.txt'
file3 = 'feasible_p_CoM.txt'

x1, y1 = read_data(file1)
x2, y2 = read_data(file2)
x3, y3, z3 = read_data_with_average(file2)
x4, y4 = read_data(file3)

# foot size -> support polygon
# <geom size="0.005" pos="-0.05 0.025 -0.03" rgba="0.2 0.2 0.2 1"/>
# <geom size="0.005" pos="-0.05 -0.025 -0.03" rgba="0.2 0.2 0.2 1"/>
# <geom size="0.005" pos="0.12 0.03 -0.03" rgba="0.2 0.2 0.2 1"/>
# <geom size="0.005" pos="0.12 -0.03 -0.03" rgba="0.2 0.2 0.2 1"/>

# Scatter 플롯 그리기
plt.figure(figsize=(6, 6))
plt.axis('equal')
plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()

# points = np.array([[0.025, 0.12],
#                    [-0.025, 0.12],
#                    [-0.025, -0.05],
#                    [0.025, -0.05]])
# points = np.vstack([points, points[0]])
# plt.plot(points[:, 0] + 0.1, points[:, 1], '-', color='blue', linewidth=2, label='Support Polygon')
# plt.plot(points[:, 0] - 0.1, points[:, 1], '-', color='blue', linewidth=2, label='Support Polygon')
x6, y6 = read_data('support_polygon_vertices.txt')
vertices_support = np.column_stack((y6, x6))
vertices_support = np.vstack([vertices_support, vertices_support[0]])
plt.plot(-vertices_support[:, 0], vertices_support[:, 1], '-', color='blue', linewidth=4)

sc6 = plt.quiver(0, 0, 0, 1, color='red', scale=10, label='X-axis')  # x축 (오른쪽)
sc7 = plt.quiver(0, 0, -1, 0, color='green', scale=10, label='Y-axis')  # y축 (위쪽)

sc1 = plt.scatter(y1, x1 , color='blue', label='sample_points : ' + str(len(x1)), alpha=0.6)
sc2 = plt.scatter(y2, x2, color='red', label='feasible_points : ' + str(len(x2)), alpha=0.6)

sc3 = plt.scatter(0.3, 0.7, color='cyan', alpha=1, s=200)
# sc3 = plt.scatter(0.0, 0.75, color='cyan', alpha=1, s=100)
# sc3 = plt.scatter(-0.3, 0.8, color='cyan', alpha=1, s=100)
# sc3 = plt.scatter(0.15, 0.7, color='cyan', alpha=1, s=100) # scene_b

# foot position
sc4 = plt.scatter(0.1, 0.0, color='black', label='rfoot', alpha=1, s=200)
plt.text(0.1 + 0.02, 0.0 + 0.02, "LFoot", fontsize=11, color='black')
sc5 = plt.scatter(-0.1,0.0,  color='black', label='lfoot', alpha=1, s=200)
plt.text(-0.1 + 0.02, 0.0 + 0.02, "RFoot", fontsize=11, color='black')
plt.text(-0.18 + 0.02, - 0.07, "Support\n   Hull", fontsize=11, color='blue')

# average feasible points
sc3 = plt.scatter(y3[0], x3[0], color='green', label='average_feasible_points : ' + str(len(x3)), alpha=1, s=200)

# # feasible p CoM
sc8 = plt.scatter(y4, x4, color='purple', label='feasible_p_CoM : ' + str(len(x4)), alpha=1, s=100)

# original feasible vertices
x5, y5 = read_data('feasible_outer_vertices.txt')
# sc9 = plt.scatter(y5, x5, color='orange', label='feasible_vertices : ' + str(len(x5)), alpha=1, s=150)
# Draw lines connecting the feasible vertices to form a polygon
vertices = np.column_stack((y5, x5))
vertices = np.vstack([vertices, vertices[0]])  # Close the polygon by repeating the first point

# feasible inner vertices
x7, y7 = read_data('feasible_inner_vertices.txt')
vertices_inner = np.column_stack((y7, x7))
vertices_inner = np.vstack([vertices_inner, vertices_inner[0]])
plt.plot(vertices_inner[:, 0], vertices_inner[:, 1], '-', color='black', linewidth=4)

# fltered feasible vertices
x6, y6 = read_data('feasible_filtered_vertices.txt')
vertices_filtered = np.column_stack((y6, x6))
vertices_filtered = np.vstack([vertices_filtered, vertices_filtered[0]])

plt.plot(vertices[:, 0], vertices[:, 1], '-', color='orange', linewidth=4)
# plt.plot(vertices_inner[:, 0], vertices_inner[:, 1], '-', color='yellow', linewidth=4)
# plt.plot(vertices_filtered[:, 0], vertices_filtered[:, 1], '-', color='black', linewidth=4)

handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Sample sets: '+str(len(x1))),
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='red', markersize=10, label='Feasible sets: '+str(len(x2))),
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='magenta', markersize=10, label='Feasible CoM sets'),
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='green', markersize=10, label='Mean of Feasible sets'),
    # plt.Line2D([], [], marker='o', color='w', markerfacecolor='cyan', markersize=10, label='left_hand_target')
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='cyan', markersize=10, label=f'Manipulation Target'),
    plt.Line2D([], [], color='orange', linewidth=3, label=f'Convex Hull'),
    plt.Line2D([], [], color='black', linewidth=3, label=f'Inner Convex Hull'),
    # plt.Line2D([], [], color='yellow', linewidth=2, label='inner_convex_hull'),
    # plt.Line2D([], [], color='black', linewidth=2, label='filtered_convex_hull'),
]

# 그래프 설정
plt.xlabel('X')
plt.ylabel('Y')
# plt.title('Scatter Plot of X, Y Coordinates of Sample and Feasible Points')
plt.legend(handles=handles, loc='best', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig('kinFeasibilitys.pdf', dpi=500)
plt.savefig('kinFeasibilitys.svg', dpi=500)

# 그래프 표시
plt.show()