# -*- coding: UTF-8 -*-

import random
import numpy as np
import filter_files.utils as utils
from filter_files.matcher import Matcher

# 状态转移矩阵，上一时刻的状态转移到当前时刻
A1 = np.array([[1, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 1],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1]])
# 状态观测矩阵
H1 = np.array([[1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0]])


class GHFilter:
    def __init__(self, X0, class1, score1, alpha=0.5, beta=0.5):
        # 迭代参数
        self.X_posterior = X0  # 后验状态, 定义为 [中心x,中心y,宽w,高h,dx,dy]
        self.class1 = class1  # 类别标签
        self.score1 = score1  # 类别分数
        self.X_prior = None  # 先验状态
        self.alpha = alpha
        self.beta = beta

        self.last_frame = 1

    def delete(self):
        self.last_frame -= 1
        return self.last_frame == -1

    def predict(self):
        """
        预测外推
        :return:
        """
        self.X_prior = np.dot(A1, self.X_posterior)
        return self.X_prior

    @staticmethod
    def association(ghfilter_list, mea_list):
        """
        多目标关联，使用最大权重匹配
        :param ghfilter_list: 状态列表，存着每个kalman对象，已经完成预测外推
        :param mea_list: 量测列表，存着矩阵形式的目标量测 ndarray [c_x, c_y, w, h].T
        :return:
        """
        # 记录需要匹配的状态和量测
        state_rec = {i for i in range(len(ghfilter_list))}
        mea_rec = {i for i in range(len(mea_list))}

        # # 将状态类进行转换便于统一匹配类型
        # state_list = list()  # [c_x, c_y, w, h].T
        # for kalman in kalman_list:
        #     state = kalman.X_prior
        #     state_list.append(state[0:4])

        # 进行匹配得到一个匹配字典
        match_dict = Matcher.match(ghfilter_list, mea_list)

        # 根据匹配字典，将匹配上的直接进行更新，没有匹配上的返回
        state_used = set()
        mea_used = set()
        match_list = list()
        for state, mea in match_dict.items():
            state_index = int(state.split('_')[1])
            mea_index = int(mea.split('_')[1])
            match_list.append([ghfilter_list[state_index], mea_list[mea_index]])
            ghfilter_list[state_index].update(mea_list[mea_index])
            state_used.add(state_index)
            mea_used.add(mea_index)

        # 求出未匹配状态和量测，返回
        return list(state_rec - state_used), list(mea_rec - mea_used), match_list

    def update(self, mea=None):
        """
        完成一次kalman滤波
        :param mea:
        :return:
        """
        if mea is not None:  # 有关联量测匹配上
            error = [mea[0][i] - self.X_prior[i] for i in range(4)]
            for i in range(4):
                self.X_posterior[i] = self.X_prior[i] + self.alpha * error[i]
            for i in range(4, 6):
                self.X_posterior[i] = self.X_prior[i] + self.beta * error[i-4]
        else:  # 无关联量测匹配上
            pass
        self.score1 = (self.score1 + mea[2]) / 2
        self.last_frame = 1
        return self.X_posterior


