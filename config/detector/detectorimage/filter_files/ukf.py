# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy import linalg
import filter_files.utils as utils
from filter_files.matcher import Matcher

# --------------------------------Kalman参数---------------------------------------
# 状态转移矩阵，上一时刻的状态转移到当前时刻
A1 = np.array([[1, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 1],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1]])
# 控制输入矩阵B
B1 = None
# 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
# 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
Q1 = np.eye(A1.shape[0]) * 0.1
# 状态观测矩阵
H1 = np.array([[1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0]])
# 观测噪声协方差矩阵R，p(v)~N(0,R)
# 观测噪声来自于检测框丢失、重叠等
R1 = np.eye(H1.shape[0]) * 1
# 状态估计协方差矩阵P初始化
P1 = np.eye(A1.shape[0])


# -------------------------------------------------------------------------------

def f1(x):
    return np.dot(A1, x)


def h1(x):
    return np.dot(H1, x)


class UKF:
    # 参数初始化
    def __init__(self, X, class1, score1, f=f1, Q=Q1, h=h1, R=R1, P=P1, n=6, m=4, alpha=0.5, beta=2.0, kk=0.0):

        self.f = f  # 状态转移方程
        self.Q = Q  # 过程噪声
        self.h = h  # 观测方程
        self.R = R  # 量测噪声
        self.P = P  # 估计状态协方差矩阵
        self.X_posterior = X  # 初始状态
        self.class1 = class1  # 类别标签
        self.score1 = score1  # 类别分数
        self.n = n  # 状态维度数
        self.m = m  # 测量值维度数
        self.alpha = alpha  # 外部参数
        self.beta = beta  # 外部参数
        self.kk = kk  # 外部参数
        self.lamb = self.alpha ** 2 * (self.n + self.kk) - self.n  # 外部参数
        c = self.n + self.lamb
        self.W_m = np.concatenate((np.array(self.lamb / c).reshape((-1, 1)), 0.5 / c
                                   + np.zeros((1, 2 * self.n))), axis=1)
        self.W_c = self.W_m.copy()
        self.W_c[0][0] += 1 + self.beta - self.alpha ** 2
        self.K = None
        self.X_mean = None
        self.P_X = None
        self.Z_mean = None
        self.P_Z = None

        self.last_frame = 1

    def delete(self):
        self.last_frame -= 1
        return self.last_frame == -1

    def sigmas(self, X0):  # 生成 Sigma 采样点
        A = math.sqrt(self.n + self.lamb) * linalg.cholesky(self.P).T
        Y = (X0 * np.ones((self.n, self.n))).T
        X_set = np.concatenate((X0.reshape(-1, 1), Y + A, Y - A), axis=1)
        return X_set  # (n, 2n+1)

    # 第一次无迹变换
    def ut1(self, Xsigma):
        num = Xsigma.shape[1]  # Sigma 点的数目
        Xmeans = np.zeros((self.n, 1))  # 均值
        Xsigma_pre = np.zeros((self.n, num))  # Sigma 点集
        for j in range(num):
            Xsigma_pre[:, j] = self.f(Xsigma[:, j])
            Xmeans = Xmeans + self.W_m[0, j] * Xsigma_pre[:, j].reshape((self.n, 1))
        Xdiv = Xsigma_pre - np.tile(Xmeans, (1, num))
        P = np.dot(np.dot(Xdiv, np.diag(self.W_c.reshape((num,)))), Xdiv.T) + self.Q
        """
           Xmeans: 均值，(self.n, 1)
           P: 协方差矩阵 (self.n, self.n)
        """
        return Xmeans, Xsigma_pre, P, Xdiv

    # 第二次无迹变换
    def ut2(self, Xsigma):
        num = Xsigma.shape[1]
        Xmeans = np.zeros((self.m, 1))
        Xsigma_pre = np.zeros((self.m, num))
        for j in range(num):
            Xsigma_pre[:, j] = self.h(Xsigma[:, j])
            Xmeans = Xmeans + self.W_m[0, j] * Xsigma_pre[:, j].reshape((self.m, 1))
        Xdiv = Xsigma_pre - np.tile(Xmeans, (1, num))
        P = np.dot(np.dot(Xdiv, np.diag(self.W_c.reshape((num,)))), Xdiv.T) + self.R
        """
           Xmeans: 均值，(self.n, 1)
           P: 协方差矩阵 (self.n, self.n)
        """
        return Xmeans, Xsigma_pre, P, Xdiv

    def predict(self):
        X_set = self.sigmas(self.X_posterior)  # Sigma 点集
        # 第一次无迹变换
        self.X_mean, X1_set, self.P_X, X2 = self.ut1(X_set)
        self.Z_mean, Z1_set, self.P_Z, Z2 = self.ut2(X1_set)

        Pxz = np.dot(np.dot(X2, np.diag(self.W_c.reshape((self.n * 2 + 1,)))), Z2.T)
        self.K = np.dot(Pxz, np.linalg.inv(self.P_Z))

    @staticmethod
    def association(ukf_list, mea_list):
        """
        多目标关联，使用最大权重匹配
        :param ukf_list: 状态列表，存着每个 ukf 对象，已经完成预测外推
        :param mea_list: 量测列表，存着矩阵形式的目标量测 ndarray [c_x, c_y, w, h].T
        :return:
        """
        # 记录需要匹配的状态和量测
        state_rec = {i for i in range(len(ukf_list))}
        mea_rec = {i for i in range(len(mea_list))}

        # # 将状态类进行转换便于统一匹配类型
        # state_list = list()  # [c_x, c_y, w, h].T
        # for ukf in ukf_list:
        #     state = ukf.Z_mean
        #     state_list.append(state[0:4])

        # 进行匹配得到一个匹配字典
        match_dict = Matcher.match(ukf_list, mea_list)

        # 根据匹配字典，将匹配上的直接进行更新，没有匹配上的返回
        state_used = set()
        mea_used = set()
        match_list = list()
        for state, mea in match_dict.items():
            state_index = int(state.split('_')[1])
            mea_index = int(mea.split('_')[1])
            match_list.append([ukf_list[state_index], mea_list[mea_index]])
            ukf_list[state_index].update(mea_list[mea_index])
            state_used.add(state_index)
            mea_used.add(mea_index)

        # 求出未匹配状态和量测，返回
        return list(state_rec - state_used), list(mea_rec - mea_used), match_list

    def update(self, mea):
        self.X_posterior = self.X_mean + np.dot(self.K, mea[0] - self.Z_mean)
        self.P = self.P_X - np.dot(np.dot(self.K, self.P_Z.T), self.K.T)

        self.score1 = (self.score1 + mea[2]) / 2
        self.last_frame = 1


if __name__ == "__main__":
    X0 = np.array([5, 5, 2, 2, 0.5, 0.5])
    ukf1 = UKF(X0, f1, Q1, h1, R1, P1, n=6, m=4)
    ukf1.predict()
    Z0 = np.array([6, 6, 2, 2]).reshape(-1, 1)
    ukf1.update(Z0)
    print(ukf1.X_posterior.shape)
    print(ukf1.P.shape)
    print(ukf1.X_posterior)
