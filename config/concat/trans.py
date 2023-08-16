# coding=gbk
import numpy as np
from PIL import Image
import os
import cv2
import time
import math
import torch
from scipy.optimize import curve_fit
from collections import Counter
import matplotlib.pyplot as plt
import shutil



def second_choose_colorization_points():
    f = open('data/calib.txt', 'r')
    # f = open('../dataset/KITTI/calib.txt', 'r')
    line = f.readline()
    i = 0
    p2 = tr = None
    while line:
        line = f.readline()
        a = line.strip().split(' ')
        i = i + 1
        if i == 2:
            p2 = np.array(
                [[eval(a[1]), eval(a[2]), eval(a[3]), eval(a[4])],
                 [eval(a[5]), eval(a[6]), eval(a[7]), eval(a[8])],
                 [eval(a[9]), eval(a[10]), eval(a[11]), eval(a[12])]])
        if i == 5:
            tr = np.array(
                [[eval(a[1]), eval(a[2]), eval(a[3]), eval(a[4])],
                 [eval(a[5]), eval(a[6]), eval(a[7]), eval(a[8])],
                 [eval(a[9]), eval(a[10]), eval(a[11]), eval(a[12])], [0, 0, 0, 1]])
    f.close()
    transformation_matrix = np.dot(p2, tr)


    # points = np.fromfile('data/velodyne/' + str(id).zfill(6) + ".bin", dtype=np.float32, count=-1).reshape([-1, 4])
    # points = np.fromfile('../dataset/object/velodyne/' + str(id).zfill(6) + ".bin", dtype=np.float32, count=-1).reshape([-1, 4])
    points_xyz = np.loadtxt('data/foreground_points_FPS3.txt', dtype=np.float32)
    points_with_reflectance = np.zeros((points_xyz.shape[0], 4), dtype=np.float32)
    points_with_reflectance[:, :3] = points_xyz
    points = points_with_reflectance
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    reflex = points[:, 3]

    img = cv2.imread('data/image_2/000000.png')
    # img = cv2.imread('../dataset/object/image_2/' + str(id).zfill(6) + ".png")
    rgb_img = np.zeros([375, 1242, 3], dtype='float64')
    # rgb_img.fill(255)
    rgb_img_R = np.zeros([375, 1242, 3], dtype='float64')
    # rgb_img_R.fill(255)

    depth_img = np.zeros([375, 1242], dtype='float64')
    depth_img.fill(1000000)

    scan = []
    fw = open('data/pc_rgb/foreground_points_FPS3.txt', 'w')
    # fw = open('../dataset/object/pc_rgb/' + str(id).zfill(6) + ".txt", 'w')
    for i in range(points.shape[0]):
        if x[i] >= 0:
            p = np.array([[x[i]], [y[i]], [z[i]], [1]])
            p = np.dot(transformation_matrix, p)
            u = p[0, 0] / p[2, 0]
            v = p[1, 0] / p[2, 0]

            if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
                r = img[int(v), int(u)][2]
                g = img[int(v), int(u)][1]
                b = img[int(v), int(u)][0]
                scan.append([x[i], y[i], z[i], reflex[i], r, g, b, v, u])

                # fw.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i]) + " " + str(r) + " " + str(g) + " " + str(b) + " " + str(v) + " " + str(u))
                fw.write(str(v) + " " + str(u))
                fw.write("\n")

    fw.close()

    print('u:', str(u))
    print('v:', str(v))

    scan = np.array(scan)
    # scan = scan[np.argsort(scan[:, -1])]
    #print('u:', scan[:, 7])
    #print('v:', scan[:, 6])

    points = scan[:, 0:3]
    remission = scan[:, 3]
    colors = scan[:, 4:7]
    depth = np.linalg.norm(points, 2, axis=1)
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]


    # max_depth = np.max(depth)
    # min_depth = np.min(depth)
    # depth = (depth - min_depth) / (max_depth - min_depth) * 255.0
    # max_remission = np.max(remission)
    # min_remission = np.min(remission)
    # remission = (remission - min_remission) / (max_remission - min_remission) * 255.0
    # max_z = np.max(scan_z)
    # min_z = np.min(scan_z)
    # scan_z = (scan_z - min_z) / (max_z - min_z) * 255.0

    allnp = np.array(depth)
    allnp = np.sort(allnp)
    allnp = np.around(allnp, 0)
    c = Counter(allnp.flatten())
    labels, values = zip(*c.items())
    labels = np.array(list(labels))
    values = np.array(list(values))
    values = values / allnp.shape[0]
    for i in range(1, values.shape[0]):
        values[i] = values[i] + values[i - 1]
    para, pcov = curve_fit(fun_five, labels, values)
    depth = fun_five(depth, para[0], para[1], para[2], para[3], para[4], para[5]) * 255.0
    remission = remission * 255.0

    znp = np.array(scan_z)
    znp = np.sort(znp)
    znp = np.around(znp, 1)
    c = Counter(znp.flatten())
    zlabels, zvalues = zip(*c.items())
    zlabels = np.array(list(zlabels))
    zvalues = np.array(list(zvalues))
    zvalues = zvalues / znp.shape[0]
    for i in range(1, zvalues.shape[0]):
        zvalues[i] = zvalues[i] + zvalues[i - 1]
    zpara, zpcov = curve_fit(fun_six, zlabels, zvalues)
    scan_z = fun_six(scan_z, zpara[0], zpara[1], zpara[2], zpara[3], zpara[4], zpara[5], zpara[6]) * 255.0


    for i in range(scan.shape[0]):
        if points[i, 0] >= 0:

            u = scan[i, 8]
            v = scan[i, 7]

            if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
                if depth_img[int(v), int(u)] > depth[i]:
                    depth_img[int(v), int(u)] = depth[i]
                    rgb_img[int(v), int(u)][0] = depth[i]
                    rgb_img[int(v), int(u)][1] = remission[i]
                    rgb_img[int(v), int(u)][2] = scan_z[i]

                    rgb_img_R[int(v), int(u)][0] = colors[i, 0]
                    rgb_img_R[int(v), int(u)][1] = colors[i, 1]
                    rgb_img_R[int(v), int(u)][2] = colors[i, 2]

    rgb_img3 = Image.fromarray(np.uint8(rgb_img))
    rgb_img3.save('data/2D_image/foreground_points_FPS3.png')
    # rgb_img3.save('../dataset/object/2D_projected_image/' + str(id).zfill(6) + '.png')
    # print('u:', str(u))
    # print('v:', str(v))
    # print(id)


def fun_five(x, a1, a2, a3, a4, a5, a6):
    return a1 * x ** 5 + a2 * x ** 4 + a3 * x ** 3 + a4 * x ** 2 + a5 * x + a6


def fun_six(x, a1, a2, a3, a4, a5, a6, a7):
    return a1 * x ** 6 + a2 * x ** 5 + a3 * x ** 4 + a4 * x ** 3 + a5 * x ** 2 + a6 * x + a7


if __name__ == '__main__':
    # first_clip_photo()
    second_choose_colorization_points()
    # third_assign_points()
    # fifth_voxel_1_1()
    # sixth_aggregation_1_1()
