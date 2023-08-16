import numpy as np
import matplotlib.pyplot as plt
import function
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

for root, dirs, files in os.walk('../dataset/KITTI/image_2'):
    for file in files:
        (filename, extension) = os.path.splitext(file)
        id = int(filename)

# 得到增广矩阵
def get_augmented_matrix(matrix, b):  # matrix 4×4
    row, col = np.shape(matrix)
    matrix = np.concatenate((matrix, b.reshape(row, 1)), axis=1)
    return matrix


# 取出增广矩阵的系数矩阵（第一列到倒数第二列）
def get_matrix(a_matrix):
    return a_matrix[:, :a_matrix.shape[1] - 1]

# 矩阵的第k行后，行交换
def exchange_row(matrix, r1, r2, k):
    matrix[[r1, r2], k:] = matrix[[r2, r1], k:]
    return matrix


# 消元计算(初等变化)
def elimination(matrix, k):  # matrix 4×7
    row, col = np.shape(matrix)
    for i in range(k + 1, row):
        m_ik = matrix[i][k] / matrix[k][k]
        matrix[i] = -m_ik * matrix[k] + matrix[i]
    return matrix


# 选列主元，在第k行后的矩阵里，找出最大值和其对应的行号和列号
def get_pos_j_max(matrix, k):
    max_v = np.max(matrix[k:, :])
    pos = np.argwhere(matrix == max_v)
    i, _ = pos[0]
    return i, max_v


# 回代求解
def backToSolve(a_matrix):
    matrix = a_matrix[:, :a_matrix.shape[1] - 1]  # 得到系数矩阵
    b_matrix = a_matrix[:, -1]  # 得到值矩阵
    row, col = np.shape(matrix)
    x = [None] * col  # 待求解空间X
    x[-1] = b_matrix[col - 1] / matrix[col - 1][col - 1]
    for _ in range(col - 1, 0, -1):
        i = _ - 1
        sij = 0
        xidx = len(x) - 1
        for j in range(col - 1, i, -1):
            sij += matrix[i][j] * x[xidx]
            xidx -= 1
        x[xidx] = (b_matrix[i] - sij) / matrix[i][i]
    return x


# 求解非齐次线性方程组：ax=b
def solve_NLQ(a, b):
    a_matrix = get_augmented_matrix(a, b)  # 增广矩阵
    for k in range(len(a_matrix) - 1):
        max_i, max_v = get_pos_j_max(get_matrix(a_matrix), k=k)  # 找列主元，系数矩阵
        if a_matrix[max_i][k] == 0:
            print('矩阵A奇异')
            return None, []
        if max_i != k:
            a_matrix = exchange_row(a_matrix, k, max_i, k=k)  # 交换行
        a_matrix = elimination(a_matrix, k=k)  # 消元初等变换
    X = backToSolve(a_matrix)  # 带回求解
    return a_matrix, X


# 计算最小二乘法当前的误差
def last_square_current_loss(xs, ys, A):
    error = 0.0
    for i in range(len(xs)):
        y1 = 0.0
        for k in range(len(A)):
            y1 += A[k] * xs[i] ** k
        error += (ys[i] - y1) ** 2
    return error


def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    num_groups = len(lines) // 4

    for i in range(0, num_groups * 4, 4):
        x_values = [float(line.split()[0]) for line in lines[i:i+4]]
        y_values = [float(line.split()[1]) for line in lines[i:i+4]]
        data.append((x_values, y_values))

    return data

def last_square_fit_curve_Gauss(xs, ys):
    sigmax = function.sigmax
    sigmay = function.sigmay
    x0 = function.x0
    y0 = function.y0

    X = np.zeros((4, 4))

    for i in range(4):
        X[i, 0] = -2 * xs[i] * sigmay ** 2
        X[i, 1] = -2 * ys[i] * sigmax ** 2
        X[i, 2] = sigmay ** 2
        X[i, 3] = sigmax ** 2

    Y = np.zeros((4, 1))

    for i in range(4):
        Y[i, 0] = 2 * (sigmay ** 2) * (sigmax ** 2) * np.exp(-(((xs[i] - x0) ** 2) / (2 * sigmax ** 2) + ((ys[i] - y0) ** 2) / (2 * sigmay ** 2))) \
                    - xs[i] ** 2 * sigmay ** 2 - ys[i] ** 2 * sigmax ** 2

    A = np.linalg.solve(np.array(X), np.array(Y))
    error = last_square_current_loss(xs=xs, ys=ys, A=A)
    return A, error


def main():
    image_path = 'E:\code\Cascade_RCNN\data\object\KITTI-10\img1'
    image_files = os.listdir(image_path)
    num_images = len(image_files)  # 图像文件的数量

    input_file = 'data/pc_rgb/foreground_points_FPS3.txt'
    output_file = 'E:/code/Cascade_RCNN/data/object/KITTI-10/det/det.txt'
    # output_file = 'det.txt'
    data = read_data(input_file)

    num_data_rows = len(data)  # 数据文件中的行数
    step = num_images / num_data_rows

    file_path = 'data/KITTI-13/gt/gt.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()


    results = []
    for idx, (xs, ys) in enumerate(data):
        column5_values = []
        column6_values = []

        for line in lines:
            data = line.split(',')
            column5_values.append(float(data[4]))
            column6_values.append(float(data[5]))

        average_column5 = sum(column5_values) / len(column5_values)
        average_column6 = sum(column6_values) / len(column6_values)

        A, error = last_square_fit_curve_Gauss(xs, ys)
        A_shortened = A[:2]

        current_idx = int(idx * step)
        result_str = f"{current_idx - 1},-1,{A_shortened[0, 0]:.3f},{A_shortened[1, 0]:.3f},{average_column5:.3f},{average_column6:.3f},100.000,-1,-1,-1\n"
        results.append((current_idx - 1, result_str))
        #result_str = f"{int(idx / num_images)},-1,{A_shortened[0, 0]:.3f},{A_shortened[1, 0]:.3f},{average_column5:.3f},{average_column6:.3f},100.000,-1,-1,-1\n"
        # results.append((int(idx / num_images), result_str))

    with open(output_file, 'r') as file:
        existing_data = file.readlines()
    all_data = []
    existing_data_idx = 0
    results_idx = 0
    while existing_data_idx < len(existing_data) and results_idx < len(results):
        existing_line = existing_data[existing_data_idx]
        existing_number = int(existing_line.split(',')[0])

        result_number, result_line = results[results_idx]

        if existing_number <= result_number:
            all_data.append(existing_line)
            existing_data_idx += 1
        else:
            all_data.append(result_line)
            results_idx += 1

    all_data.extend(existing_data[existing_data_idx:])

    # 将可能剩余的新数据添加到all_data中
    for _, result_line in results[results_idx:]:
        all_data.append(result_line)

    with open(output_file, 'w') as file:
        file.writelines(all_data)

    #with open(output_file, 'a') as file:
        #idx = 0
        #for xs, ys in data:
            #A, error = last_square_fit_curve_Gauss(xs, ys)
            #A_shortened = A[:2]  # 只保留 A 中的前两个元素
            #file.write(str(int(idx / num_images)) + ",")
            #file.write("-1" + ",")
            #file.write("{:.3f}, {:.3f}".format(A_shortened[0, 0], A_shortened[1, 0]) + ",")
            #file.write("1" + ",")
            #file.write(str(-1) + "," + str(-1) + "," + str(-1) + "\n")
            #result_str = f"{int(idx / num_images)},-1,{A_shortened[0, 0]:.3f},{A_shortened[1, 0]:.3f},1,-1,-1,-1\n"
            #file.write(result_str)
            #idx += 1
            # file.write("Error: {}\n".format(error))
            # file.write("-" * 20 + "\n")


if __name__ == '__main__':
    main()



