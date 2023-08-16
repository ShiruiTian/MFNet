import numpy as np
from scipy.optimize import minimize
import scipy.io as sio
import configparser
import matplotlib.pyplot as plt


def gauss2d(sigmax, sigmay, x0, y0):

    # x, y = data0
    inner = ((x - x0) ** 2) / (2 * sigmax ** 2) + ((y - y0) ** 2) / (2 * sigmay ** 2)
    z = np.exp(-inner)
    return z


f = open('data/label/000000.txt', 'r')
line = f.readline()
i = 0
x0, y0 = None, None
while line:
    line = f.readline()
    a = line.strip().split(' ')
    if len(a) >= 2:  # 确保至少有两个值才进行赋值
        x0, y0 = np.array([eval(a[0]), eval(a[1])])  # 将坐标值存储到变量 x0, y0 中
        break

f.close()



# 已知的高斯点中心和坐标数据
#x0, y0 = 0.3, 0.7
data0 = np.loadtxt('data/pc_rgb/foreground_points_FPS3.txt')
data0 = data0[:277]
x = data0[:, 0]
y = data0[:, 1]
zobs = gauss2d(1, 2, x0, y0)

zobs[zobs <= 0] = 1e-20
log_values = np.log(zobs)


# np.savetxt('../dataset/target/gaussian/zobs.txt', log_values, delimiter=',', fmt='%.6e')


# 定义目标函数
def objective(params):
    sigmax, sigmay, x0, y0 = params
    zpred = gauss2d(sigmax, sigmay, x0, y0)
    error = zpred - zobs
    return np.sum(error**2)


'''def save_parameters_to_ini_file(filename, sigmax, sigmay, x0, y0):
    config = configparser.ConfigParser()
    config['PARAMETERS'] = {
        'sigmax': str(sigmax),
        'sigmay': str(sigmay),
        'x0': str(x0),
        'y0': str(y0)
    }
    with open(filename, 'w') as file:
        file.write(str("{:.6e}".format(sigmax)) + '\n')
        file.write(str("{:.6e}".format(sigmay)) + '\n')
        file.write(str("{:.6e}".format(x0)) + '\n')
        file.write(str("{:.6e}".format(y0)) + '\n')
    return sigmax, sigmay, x0, y0'''


def main():
    # 初始猜测值
    guess = [1, 1, 0, 0]
    # 求解未知参数
    result = minimize(objective, guess, method='Nelder-Mead')

    # 提取求解得到的参数
    sigmax, sigmay, x0, y0 = result.x

    # 打印结果
    print('sigmax:',sigmax)
    print('sigmay:',sigmay)
    # print("c =", c)
    print('x0:',x0)
    print('y0:',y0)

    data = {
        'sigmax': sigmax,
        'sigmay': sigmay,
        'x0': x0,
        'y0': y0
    }
    sio.savemat('gaussian_parameters.mat', data)

    return sigmax, sigmay, x0, y0

    # save_parameters_to_ini_file('../dataset/target/gaussian/parameters.txt', sigmax, sigmay, x0, y0)


# main()
sigmax, sigmay, x0, y0 = main()

