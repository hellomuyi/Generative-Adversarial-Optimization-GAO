#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 2020/11/20 10:36
# @Author : JXF
# @File   : my_cec.py

import numpy as np
import re  # 正则表达式


def loadtxt(infile):
    """读取txt数据"""
    f = open(infile, 'r')
    lines = f.readlines()
    dataset = []
    for line in lines:
        temp1 = line.strip()
        temp2 = re.split('  | |\t', temp1)
        dataset.append(temp2)
    k = np.array(dataset).shape[1]
    for i in range(len(dataset)):
        for j in range(k):
            dataset[i].append(eval(dataset[i][j]))
        del (dataset[i][0:k])
    return dataset


def select_dim(dim):
    """输入测试维度，输出测试集函数字典"""
    # 根据文件，dim只能取以下维度
    dim_list = [2, 5, 10, 20, 30,  40, 50, 60, 70, 80, 90, 100]
    if dim not in dim_list:
        raise ValueError('Dim does not match, please modify the dimensions, such as 2, 5, 10.')
    # 在测试编写的28个函数时，为与C代码保持一致，dim取10，否则须与main.py的test_dim保持一致
    fileOs = 'input_data/shift_data.txt'
    fileMr = 'input_data/M_D' + str(dim) + '.txt'
    # 读取平移数据和旋转矩阵
    Os = np.array(loadtxt(fileOs))  # Oshift,10*100
    Mr = np.array(loadtxt(fileMr))  # Mrotate,(D*D)*10 = 10D*D
    # Mr = np.array(loadtxt(fileMr))  # Mrotate,(D*D)*10 = 10D*D
    # print('Os.shape:', Os.shape)
    # print('Mr.shape:', Mr.shape)

    def shift(x, os_idx=0):
        """
        平移变换
        使用起始下标为os_idx*dim的平移数据
        """
        m = x.shape[0]
        n = x.shape[1]
        x_shift = Os.reshape(1, Os.shape[0] * Os.shape[1])
        x_shift = x_shift[0, os_idx * n:(os_idx + 1) * n]
        x_shift = np.tile(x_shift, (m, 1))
        x_shift = x - x_shift
        return x_shift

    def rotate(x, mr_idx=0, r_flag=False):
        """
        旋转变换
        使用第mr_idx个正交矩阵对x进行变换，mr_idx为下标
        第5个函数本身不旋转，但在第21个组合函数中又旋转，故增加一个旋转标志r_flag
        """
        if r_flag:
            n = x.shape[1]
            if dim != n:
                raise ValueError('if you want to use function "rotate", '
                                 'please motify the variable "dim" to match the dim of x in my_cec.py.')

            mr_cur = Mr[n * mr_idx:n * (mr_idx + 1), :]
            mr_cur = mr_cur.T  # 参照C代码的rotatefunc函数，所用的矩阵是文件中矩阵的转置
            x_rotate = x.dot(mr_cur)
            return x_rotate
        return x

    def dia(x, alpha):
        """
        右乘对角矩阵
        """
        n = x.shape[1]
        i = np.array(range(n))
        return x * alpha ** (i / (n - 1) / 2)

    def asy(x, beta):
        """asy transformation"""
        """
        x_asy = x
        xt = np.where(x_asy > 0)    # 返回元组,包含两个列表，分别是行列下标索引
        for k in range(len(xt[0])):
            i, j = xt[0][k], xt[1][k]
            x_asy[i, j] = x_asy[i, j]**(1+beta*j*np.sqrt(x_asy[i, j])/(x.shape[1]-1))
        return x_asy
        """
        x_asy = x
        xt = np.where(x > 0)  # 返回元组,包含两个列表，分别是行列下标索引
        x_asy[xt] = x_asy[xt] ** (1 + beta * xt[1] / (x.shape[1] - 1) * np.sqrt(x_asy[xt]))
        return x_asy

    def osz(x):
        """osz transformation"""
        m = x.shape[0]
        n = x.shape[1]
        x_osz = x
        for i in range(m):
            for j in [0, n - 1]:
                if 0 != x[i, j]:
                    x_hat = np.log(abs(x[i, j]))
                else:
                    x_hat = 0

                if x[i][j] > 0:
                    c1, c2 = 10, 7.9
                else:
                    c1, c2 = 5.5, 3.1
                x_osz[i, j] = np.sign(x[i, j]) * np.exp(x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat)))
        # print('x_osz:\n', x_osz)
        return x_osz

    def func01_sphere(x, os_idx=0, mr_idx=0, r_flag=False):
        """
        x:sample_num * dim，5*30
        返回值：ndarray，shape:(sample_num,)

        使用起始下标为os_idx的平移数据
        加上mr_idx为了后面组合函数的统一处理，尽管此函数未使用旋转矩阵
        加上r_flag为了后面组合函数的统一处理，当函数本身有旋转时，默认True，否则False
        使用第mr_idx个正交矩阵对x进行变换，Mr_idx为下标，参考C代码，知：
        当有两次旋转变换时，第一次使用第mr_idx个矩阵，第二次使用第mr_idx+1个矩阵
        当有三次旋转变换时，第一次使用第mr_idx个矩阵，第二次使用mR_idx+1，第三次mr_idx
        """
        x_hat = rotate(shift(x, os_idx), mr_idx, r_flag)
        return (x_hat ** 2).sum(axis=1) - 1400

    def func02_ellips(x, os_idx=0, mr_idx=0, r_flag=True):
        n = x.shape[1]
        i = np.array(range(n))
        x_hat = osz(rotate(shift(x, os_idx), mr_idx, r_flag))
        return np.sum(10 ** (6 * i / (n - 1)) * x_hat ** 2, axis=1) - 1300  # 广播

    def func03_bent_cigar(x, os_idx=0, mr_idx=0, r_flag=True):
        x_hat = rotate(asy(rotate(shift(x, os_idx), mr_idx, r_flag), 0.5), mr_idx + 1, r_flag)
        return x_hat[:, 0] ** 2 + 10 ** 6 * np.sum(x_hat[:, 1:] ** 2, axis=1) - 1200

    def func04_discus(x, os_idx=0, mr_idx=0, r_flag=True):
        x_hat = osz(rotate(shift(x, os_idx), mr_idx, r_flag))
        return 10 ** 6 * x_hat[:, 0] ** 2 + np.sum(x_hat[:, 1:] ** 2, axis=1) - 1100

    def func05_dif_powers(x, os_idx=0, mr_idx=0, r_flag=False):
        x_hat = rotate(shift(x, os_idx), mr_idx, r_flag)
        n = x.shape[1]
        i = np.array(range(n))
        return (np.sum(np.fabs(x_hat) ** (2 + 4 * i / (n - 1)), axis=1)) ** 0.5 - 1000

    def func06_rosenbrock(x, os_idx=0, mr_idx=0, r_flag=True):
        x_hat = rotate(2.048 * shift(x, os_idx) / 100, mr_idx, r_flag) + 1
        n = x.shape[1]
        return np.sum(100 * (x_hat[:, 0:n - 1] ** 2 - x_hat[:, 1:n]) ** 2 + (x_hat[:, 0:n - 1] - 1) ** 2, axis=1) - 900

    def func07_schaffer(x, os_idx=0, mr_idx=0, r_flag=True):
        n = x.shape[1]
        y = rotate(dia(asy(rotate(shift(x, os_idx), mr_idx, r_flag), 0.5), 10), mr_idx + 1, r_flag)
        # y = dia(rotate(asy(rotate(shift(x, os_idx), mr_idx, r_flag), 0.5), mr_idx + 1, r_flag), 10)
        z = np.sqrt(y[:, 0:n - 1] ** 2 + y[:, 1:n] ** 2)
        return (1 / (n - 1) * np.sum(np.sqrt(z) + np.sqrt(z) * (np.sin(50 * z ** 0.2)) ** 2, axis=1)) ** 2 - 800

    def func08_ackley(x, os_idx=0, mr_idx=0, r_flag=True):
        n = x.shape[1]
        x_hat = rotate(dia(asy(rotate(shift(x, os_idx), mr_idx, r_flag), 0.5), 10), mr_idx + 1, r_flag)
        return -20 * np.exp(-0.2 * np.sqrt(1 / n * np.sum(x_hat ** 2, axis=1))) - \
               np.exp((1 / n * np.sum(np.cos(2 * np.pi * x_hat), axis=1))) + 20 + np.e - 700

    def func09_weierstrass(x, os_idx=0, mr_idx=0, r_flag=True):
        a, b, kmax = 0.5, 3, 20
        k_v = np.array(range(kmax + 1))
        n = x.shape[1]
        x_hat = rotate(dia(asy(rotate(shift(x, os_idx) * 0.5 / 100, mr_idx, r_flag), 0.5), 10), mr_idx + 1, r_flag)
        # x_hat = np.tile(x_hat, (1, kmax + 1))
        temp = x_hat
        for _ in range(kmax):
            x_hat = np.dstack((x_hat, temp))
        # x_hat = x_hat.transpose((2, 0, 1))  # 转置
        # print(x_hat)
        # print('x_hat:', x_hat.shape)
        k = np.ones(x.shape) * kmax
        for i in range(kmax):
            k = np.dstack((k, np.ones(x.shape) * i))
        # k = k.transpose((2, 0, 1))    # 转置
        # print(k)
        # print('k:', k.shape)
        return np.sum(np.sum(a ** k * np.cos(2 * np.pi * b ** k * (x_hat + 0.5)), axis=2), axis=1) - \
               n * np.sum(a ** k_v * np.cos(2 * np.pi * b ** k_v * 0.5)) - 600

    def func10_griewank(x, os_idx=0, mr_idx=0, r_flag=True):
        n = x.shape[1]
        i = np.array(range(n))
        x_hat = dia(rotate(shift(x, os_idx) * 600 / 100, mr_idx, r_flag), 100)
        return np.sum(x_hat ** 2 / 4000, axis=1) - np.prod(np.cos(x_hat / np.sqrt(i + 1)), axis=1) + 1 - 500

    def func11_rastrigin(x, os_idx=0, mr_idx=0, r_flag=False):
        x_hat = rotate(dia(rotate(asy(osz(rotate(shift(x, os_idx) * 5.12 / 100, mr_idx, r_flag)), 0.2),
                                  mr_idx + 1, r_flag), 10), mr_idx, r_flag)
        return np.sum(x_hat ** 2 - 10 * np.cos(2 * np.pi * x_hat) + 10, axis=1) - 400

    def func12_rotated_rastrigin(x, os_idx=0, mr_idx=0, r_flag=True):
        z = rotate(dia(rotate(asy(osz(rotate(shift(x, os_idx) * 5.12 / 100, mr_idx, r_flag)), 0.2),
                              mr_idx + 1, r_flag), 10), mr_idx, r_flag)
        return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10, axis=1) - 300

    def func13_step_rastrigin(x, os_idx=0, mr_idx=0, r_flag=True):
        x_hat = rotate(shift(x, os_idx) * 5.12 / 100, mr_idx, r_flag)
        x_hat = np.where(np.fabs(x_hat) > 0.5, np.round(2 * x_hat) / 2, x_hat)
        z = rotate(dia(rotate(asy(osz(x_hat), 0.2), mr_idx + 1, r_flag), 10), mr_idx, r_flag)
        return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10, axis=1) - 200

    def func14_schwefel(x, os_idx=0, mr_idx=0, r_flag=False):
        n = x.shape[1]
        z = dia(rotate(shift(x, os_idx) * 10, mr_idx, r_flag), 10) + 4.209687462275036e+002
        gz = z
        idx = np.where(np.fabs(z) <= 500)
        gz[idx] = z[idx] * np.sin(np.fabs(z[idx]) ** 0.5)

        idx = np.where(z > 500)
        tmp = (500 - z[idx] % 500)
        gz[idx] = tmp * np.sin(np.fabs(tmp) ** 0.5) - ((z[idx] - 500) / 100) ** 2 / n

        idx = np.where(z < -500)
        tmp = np.fabs(z[idx]) % 500 - 500
        gz[idx] = tmp * np.sin(np.fabs(tmp) ** 0.5) - ((z[idx] + 500) / 100) ** 2 / n

        return 4.189828872724338e+002 * n - np.sum(gz, axis=1) - 100

    def func15_rotated_schwefel(x, os_idx=0, mr_idx=0, r_flag=True):
        n = x.shape[1]
        z = dia(rotate(shift(x, os_idx) * 10, mr_idx, r_flag), 10) + 4.209687462275036e+002
        gz = z
        idx = np.where(np.fabs(z) <= 500)
        gz[idx] = z[idx] * np.sin(np.fabs(z[idx]) ** 0.5)

        idx = np.where(z > 500)
        tmp = (500 - z[idx] % 500)
        gz[idx] = tmp * np.sin(np.fabs(tmp) ** 0.5) - ((z[idx] - 500) / 100) ** 2 / n

        idx = np.where(z < -500)
        tmp = np.fabs(z[idx]) % 500 - 500
        gz[idx] = tmp * np.sin(np.fabs(tmp) ** 0.5) - ((z[idx] + 500) / 100) ** 2 / n

        return 4.189828872724338e+002 * n - np.sum(gz, axis=1) + 100

    def func16_katsuura(x, os_idx=0, mr_idx=0, r_flag=True):
        z = rotate(dia(rotate(shift(x, os_idx) * 5 / 100, mr_idx, r_flag), 100), mr_idx + 1, r_flag)
        temp = z
        for _ in range(32 - 1):
            z = np.dstack((z, temp))
        k = np.ones(x.shape) * 32
        for i in np.linspace(1, 31, 31):
            k = np.dstack((k, np.ones(x.shape) * i))
        n = x.shape[1]
        i = np.array(range(n)) + 1
        j = np.array(range(32), dtype=np.int64) + 1  # 默认32会溢出
        # print(2**j)
        return 10 / n ** 2 * np.prod(
            (1 + i * np.sum(np.fabs(2 ** j * z - np.round(2 ** j * z)) / 2 ** j, axis=2)) ** (10 / n ** 1.2), axis=1) \
               - 10 / n ** 2 + 200

    def func17_bi_rastrigin(x, os_idx=0, mr_idx=0, r_flag=False):
        n = x.shape[1]
        x_asterisk = -shift(np.zeros((1, n)), os_idx)  # 也就是Os文件的前n个数据
        miu0, d = 2.5, 1
        s = 1 - 1 / (2 * (n + 20) ** 0.5 - 8.2)
        miu1 = -((miu0 ** 2 - d) / s) ** 0.5
        y = shift(x, os_idx) * 10 / 100
        x_hat = 2 * np.sign(x_asterisk) * y + miu0
        z = rotate(dia(rotate(x_hat - miu0, mr_idx, r_flag), 100), mr_idx + 1, r_flag)

        tmp1 = np.sum((x_hat - miu0) ** 2, axis=1)
        tmp2 = d * n + s * np.sum((x_hat - miu1) ** 2, axis=1) + 10 * (n - np.sum(np.cos(2 * np.pi * z), axis=1))
        return np.min(np.vstack((tmp1, tmp2)), axis=0) + 300

    def func18_rotated_bi_rastrigin(x, os_idx=0, mr_idx=0, r_flag=True):
        n = x.shape[1]
        x_asterisk = -shift(np.zeros((1, n)), os_idx)  # 也就是Os文件的前n个数据
        miu0, d = 2.5, 1
        s = 1 - 1 / (2 * (n + 20) ** 0.5 - 8.2)
        miu1 = -((miu0 ** 2 - d) / s) ** 0.5
        y = shift(x, os_idx) * 10 / 100
        x_hat = 2 * np.sign(x_asterisk) * y + miu0
        z = rotate(dia(rotate(x_hat - miu0, mr_idx, r_flag), 100), mr_idx + 1, r_flag)

        tmp1 = np.sum((x_hat - miu0) ** 2, axis=1)
        tmp2 = d * n + s * np.sum((x_hat - miu1) ** 2, axis=1) + 10 * (n - np.sum(np.cos(2 * np.pi * z), axis=1))
        return np.min(np.vstack((tmp1, tmp2)), axis=0) + 400

    def func19_grie_rosen(x, os_idx=0, mr_idx=0, r_flag=True):
        z = rotate(shift(x, os_idx) * 5 / 100, mr_idx, r_flag) + 1
        z_ = z[:, 1:]  # np.delete(z, 0, 1)
        z_ = np.c_[z_, z[:, 0]]  # 按C代码，z(i)-z(i+1), z(n-1)-z(0)
        g2 = 100 * (z ** 2 - z_) ** 2 + (z - 1) ** 2
        i = np.array(range(x.shape[1])) + 1
        return np.sum(g2 ** 2 / 4000 - np.cos(g2 / i ** 0.5) + 1, axis=1) + 500

    def func20_escaffer6(x, os_idx=0, mr_idx=0, r_flag=True):
        z = rotate(asy(rotate(shift(x, os_idx), mr_idx, r_flag), 0.5), mr_idx + 1, r_flag)
        z_ = z[:, 1:]
        z_ = np.c_[z_, z[:, 0]]
        return np.sum(0.5 + (np.sin((z ** 2 + z_ ** 2) ** 0.5) ** 2 - 0.5) /
                      (1 + 0.001 * (z ** 2 + z_ ** 2)) ** 2, axis=1) + 600

    cec13_top20 = {0: func01_sphere,
                   1: func02_ellips,
                   2: func03_bent_cigar,
                   3: func04_discus,
                   4: func05_dif_powers,
                   5: func06_rosenbrock,
                   6: func07_schaffer,
                   7: func08_ackley,
                   8: func09_weierstrass,
                   9: func10_griewank,
                   10: func11_rastrigin,
                   11: func12_rotated_rastrigin,
                   12: func13_step_rastrigin,
                   13: func14_schwefel,
                   14: func15_rotated_schwefel,
                   15: func16_katsuura,
                   16: func17_bi_rastrigin,
                   17: func18_rotated_bi_rastrigin,
                   18: func19_grie_rosen,
                   19: func20_escaffer6,
                   }

    def cf_cal(x, func_id, sigma, bias, mylambda, r_flag=True):
        """计算组合函数值的通用函数"""
        INF = 1.0e99
        (m, n) = x.shape
        cf_num = len(func_id)
        fit = np.zeros(shape=(m, 0))
        for i in range(cf_num):
            cur_id = func_id[i]
            opt = -100.0 * (15 - cur_id) if cur_id <= 14 else 100.0 * (cur_id - 14)
            fit = np.hstack((fit, cec13_top20[cur_id - 1](x, os_idx=i, mr_idx=i, r_flag=r_flag).reshape(m, 1) - opt))

        # print(fit)
        fit = fit * mylambda + bias  # sample_num*cf_num
        cur_Os = Os.reshape(1, Os.shape[0] * Os.shape[1])[0, 0:cf_num * n].reshape(cf_num, n)
        tmp = x
        for _ in range(cf_num - 1):
            x = np.dstack((x, tmp))  # sample_num * dim * cf_num
        x = x.transpose((0, 2, 1))  # sample_num * cf_num * dim
        w = np.sum((x - cur_Os) ** 2, axis=2)  # sample_num*cf_num
        sigma = np.tile(sigma, m).reshape(m, cf_num)  # sample_num*cf_num0
        idx0 = np.where(w != 0)
        idx1 = np.where(0 == w)
        w[idx0] = (1 / w[idx0]) ** 0.5 * np.exp(-w[idx0] / 2 / n / sigma[idx0] ** 2)
        w[idx1] = INF
        # print(w)
        w_max = np.max(w, axis=1)
        w_sum = np.sum(w, axis=1)
        idx = np.where(0 == w_max)
        w[idx] = 1
        w_sum[idx[0]] = cf_num
        w_sum = w_sum.reshape(m, 1)
        # print(w_sum)
        return np.sum(w / w_sum * fit, axis=1)

    def func21_cf1(x):
        func_id = [6, 5, 3, 4, 1]
        sigma = np.array([10, 20, 30, 40, 50])
        bias = np.array([0, 100, 200, 300, 400])
        mylambda = np.array([1, 1e-6, 1e-26, 1e-6, 0.1])
        return cf_cal(x, func_id, sigma, bias, mylambda, r_flag=True) + 700

    def func22_cf2(x):
        func_id = [14, 14, 14]
        sigma = np.array([20, 20, 20])
        bias = np.array([0, 100, 200])
        mylambda = np.array([1, 1, 1])
        return cf_cal(x, func_id, sigma, bias, mylambda, r_flag=False) + 800

    def func23_cf3(x):
        func_id = [15, 15, 15]
        sigma = np.array([20, 20, 20])
        bias = np.array([0, 100, 200])
        mylambda = np.array([1, 1, 1])
        return cf_cal(x, func_id, sigma, bias, mylambda, r_flag=True) + 900

    def func24_cf4(x):
        func_id = [15, 12, 9]
        sigma = np.array([20, 20, 20])
        bias = np.array([0, 100, 200])
        mylambda = np.array([0.25, 1, 2.5])
        return cf_cal(x, func_id, sigma, bias, mylambda, r_flag=True) + 1000

    def func25_cf5(x):
        func_id = [15, 12, 9]
        sigma = np.array([10, 30, 50])
        bias = np.array([0, 100, 200])
        mylambda = np.array([0.25, 1, 2.5])
        return cf_cal(x, func_id, sigma, bias, mylambda, r_flag=True) + 1100

    def func26_cf6(x):
        func_id = [15, 12, 2, 9, 10]
        sigma = np.array([10, 10, 10, 10, 10])
        bias = np.array([0, 100, 200, 300, 400])
        mylambda = np.array([0.25, 1, 1e-7, 2.5, 10])
        return cf_cal(x, func_id, sigma, bias, mylambda, r_flag=True) + 1200

    def func27_cf7(x):
        func_id = [10, 12, 15, 9, 1]
        sigma = np.array([10, 10, 10, 20, 20])
        bias = np.array([0, 100, 200, 300, 400])
        mylambda = np.array([100, 10, 2.5, 25, 0.1])
        return cf_cal(x, func_id, sigma, bias, mylambda, r_flag=True) + 1300

    def func28_cf8(x):
        func_id = [19, 7, 15, 20, 1]
        sigma = np.array([10, 20, 30, 40, 50])
        bias = np.array([0, 100, 200, 300, 400])
        mylambda = np.array([2.5, 2.5e-3, 2.5, 5e-4, 0.1])
        return cf_cal(x, func_id, sigma, bias, mylambda, r_flag=True) + 1400

    # 由cec.py第23行，key值从0开始
    cec13_dict = {0: func01_sphere,
                  1: func02_ellips,
                  2: func03_bent_cigar,
                  3: func04_discus,
                  4: func05_dif_powers,
                  5: func06_rosenbrock,
                  6: func07_schaffer,
                  7: func08_ackley,
                  8: func09_weierstrass,
                  9: func10_griewank,
                  10: func11_rastrigin,
                  11: func12_rotated_rastrigin,
                  12: func13_step_rastrigin,
                  13: func14_schwefel,
                  14: func15_rotated_schwefel,
                  15: func16_katsuura,
                  16: func17_bi_rastrigin,
                  17: func18_rotated_bi_rastrigin,
                  18: func19_grie_rosen,
                  19: func20_escaffer6,
                  20: func21_cf1,
                  21: func22_cf2,
                  22: func23_cf3,
                  23: func24_cf4,
                  24: func25_cf5,
                  25: func26_cf6,
                  26: func27_cf7,
                  27: func28_cf8,
                  }
    return cec13_dict


# def cec13(i):
#     """返回下标i对应的函数"""
#     return cec13_dict[i]


# CEC2017测试集
cec17_dict = {}

if __name__ == '__main__':
    # 测试编写的28个cec13函数
    # 按照C代码，用2*10维来测试，第一个sample是是shift值，第二个sample全0
    test_dim = 10
    if test_dim != 10:
        raise ValueError('Dim must be 10 in my_cec.py. Please motify the variable "dim" to 10 in the 29th line.')
    fileOs = 'input_data/shift_data.txt'
    Os = np.array(loadtxt(fileOs))  # Oshift,10*100
    cec13 = select_dim(test_dim)
    a = Os[0, 0:test_dim]
    b = np.zeros(test_dim)
    x_test = np.vstack((a, b))
    # print('x:\n', x_test)
    # print('x_shift:\n', shift(x_test, Os))
    # print('x_rotate:\n', rotate(x_test, Mr, 1))
    # print('x_asy:\n', asy(x_test, 0.5))
    # print('x_osz:\n', osz(x_test))

    for i in range(28):
        # print('fun%2d: ' % (i+1), cec13(i)(x_test))
        print('fun%2d: %5d   %.4f' % (i+1, cec13[i](x_test)[0], cec13[i](x_test)[1]))
