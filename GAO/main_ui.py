#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 2021/3/8 10:00
# @Author : JXF
# @File   : main_ui.py


from tkinter import *
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk  # pillow 模块
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import numpy as np


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()


class InitFace:
    """主界面，可以加背景图片"""

    def __init__(self, master):
        self.root = master
        self.root.title('test')
        screenwidth = int(root.winfo_screenwidth())
        screenheight = int(root.winfo_screenheight())
        width, height = 900, 450  # 940, 480
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, 0)
        self.root.geometry(alignstr)
        # 基准界面initface
        self.initface = Frame(self.root, width=width / 2, height=height / 2)
        self.initface.rowconfigure(list(range(4)), weight=1)
        self.initface.columnconfigure(list(range(2)), weight=1)
        # self.initface.grid(sticky=W+E+N+S, padx=0, pady=0)
        self.initface.pack(fill=BOTH, expand=True)
        # self.canvas = Canvas(self.initface, width=width, height=height).grid()  # 创建画布
        # self.image_file = PhotoImage(file="sphere.jpg")  # 加载图片文件
        # self.image = self.canvas.create_image(0, 0, anchor='nw', image=self.image_file)  # 将图片置于画布上
        # self.image_file = PhotoImage(file=r"sphere.jpg")  # 加载图片文件
        # im = Image.open("sphere.jpg")
        # tkimg = ImageTk.PhotoImage(im)  # 执行此函数之前， Tk() 必须已经实例化。
        # Label(self.initface, image=tkimg).grid(row=5,column=0)
        # # Label(self.initface, Image=self.image_file).grid()

        # bg2 = "#F7F7F7"      # "#FFFAFA"
        # bg1 = "#F0F0F0"  #  "#A6BBC1"        # 接近背景色
        bg1 = "#E0E0E0"  # 接近背景色灰色
        bg2 = "white"  # "#FFFAFA"
        label1 = Label(self.initface, text="算法演示", bg=bg1, font=('Arial', 50))
        label1.grid(sticky=N + S + W + E, row=0, column=0, columnspan=2)

        btn0 = Button(self.initface, width=30, height=5, text='演示', bg=bg2,
                      font=('华文宋体', 20, 'bold'), command=self.toDemonstration)
        btn1 = Button(self.initface, width=30, height=5, text='实验', bg=bg2,
                      font=('华文宋体', 20, 'bold'), command=self.toExperiment)
        btn_exit = Button(self.initface, text='Exit', bg=bg2, font=('Times new Roman', 20), command=on_closing)
        btn0.grid(sticky=W + E + N + S, row=1, column=0)
        btn1.grid(sticky=W + E + N + S, row=1, column=1)
        btn_exit.grid(sticky=W + E + N + S, row=3, column=0, columnspan=2)
        text = '介绍：\n演示功能用于展示二维寻优过程;\n实验功能用于自定义参数实验'
        label2 = Label(self.initface, text=text, font=('华文宋体', 15), bg=bg1)
        label2.grid(sticky=W + E + N + S, row=2, column=0, columnspan=2)

    def toDemonstration(self, ):
        self.initface.destroy()
        Demonstrate(self.root)

    def toExperiment(self, ):
        self.initface.destroy()
        Experiment(self.root)


class Demonstrate:
    """动态寻优界面"""
    def __init__(self, master=None):
        self.master = master
        self.master.title('demonstration')
        screenwidth = int(self.master.winfo_screenwidth())
        screenheight = int(self.master.winfo_screenheight())
        width, height = 900, 450
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, 0)
        # self.master.geometry(alignstr)

        self.face1 = tk.Frame(self.master, )  # 这个master与Frame的区别
        self.face1.pack(fill=BOTH, expand=True)
        self.face1.rowconfigure(list(range(15)), weight=1)
        self.face1.columnconfigure(list(range(3)), weight=1)
        self.ax1 = None
        self.ax2 = None
        self.canvas = None
        self.arrange()

    def arrange(self):
        # 创建画布
        fig = plt.figure()
        plt.ion()  # 切换交互模式
        gs = gridspec.GridSpec(1, 5)
        self.ax1 = plt.subplot(gs[0, 0:2])
        self.ax2 = plt.subplot(gs[0, 2:], projection='3d')
        self.ax1.grid()
        # plt.tight_layout()
        plt.subplots_adjust(left=0.08, bottom=0.09, right=0.95, top=0.96, wspace=0.01, hspace=0.1)
        plt.ioff()
        self.canvas = FigureCanvasTkAgg(fig, master=self.face1)  # 只需要设置宽度，由于sticky=N+S+W+E，高度会自动对齐
        self.canvas.draw()

        # 创建按钮，anchor设置控件内文字在控件内的位置
        ft = tkfont.Font(family='Times new Roman')                          # 正常字体
        ft_bd = tkfont.Font(family="Times new Roman", weight=tkfont.BOLD)   # 加黑字体
        bg1 = "white"  # "#FFFAFA"
        bg2 = "#EAEAEA"     # 灰色
        btn0 = Button(self.face1, width=20, command=lambda: self.execute(1), text='sphere', font=ft, bg=bg2)
        btn1 = Button(self.face1, width=20, command=lambda: self.execute(2), text='ellips', font=ft, bg=bg2)
        btn2 = Button(self.face1, command=lambda: self.execute(3), text='bent_cigar', font=ft, bg=bg1)
        btn3 = Button(self.face1, command=lambda: self.execute(4), text='discus', font=ft, bg=bg1)
        btn4 = Button(self.face1, command=lambda: self.execute(5), text='dif_powers', font=ft, bg=bg2)
        btn5 = Button(self.face1, command=lambda: self.execute(6), text='rosenbrock', font=ft, bg=bg2)
        btn6 = Button(self.face1, command=lambda: self.execute(7), text='schaffer', font=ft, bg=bg1)
        btn7 = Button(self.face1, command=lambda: self.execute(8), text='ackley', font=ft, bg=bg1)
        btn8 = Button(self.face1, command=lambda: self.execute(9), text='weierstrass', font=ft, bg=bg2)
        btn9 = Button(self.face1, command=lambda: self.execute(10), text='griewank', font=ft, bg=bg2)
        btn10 = Button(self.face1, command=lambda: self.execute(11), text='rastrigin', font=ft, bg=bg1)
        btn11 = Button(self.face1, command=lambda: self.execute(12), text='rotated_rastrigin', font=ft, bg=bg1)
        btn12 = Button(self.face1, command=lambda: self.execute(13), text='step_rastrigin', font=ft, bg=bg2)
        btn13 = Button(self.face1, command=lambda: self.execute(14), text='schwefel', font=ft, bg=bg2)
        btn14 = Button(self.face1, command=lambda: self.execute(15), text='rotated_schwefel', font=ft, bg=bg1)
        btn15 = Button(self.face1, command=lambda: self.execute(16), text='katsuura', font=ft, bg=bg1)
        btn16 = Button(self.face1, command=lambda: self.execute(17), text='bi_rastrigin', font=ft, bg=bg2)
        btn17 = Button(self.face1, command=lambda: self.execute(18), text='rotated_bi_rastrigin', font=ft, bg=bg2)
        btn18 = Button(self.face1, command=lambda: self.execute(19), text='grie_rosen', font=ft, bg=bg1)
        btn19 = Button(self.face1, command=lambda: self.execute(20), text='escaffer6', font=ft, bg=bg1)
        btn20 = Button(self.face1, command=lambda: self.execute(21), text='cf1', font=ft, bg=bg2)
        btn21 = Button(self.face1, command=lambda: self.execute(22), text='cf2', font=ft, bg=bg2)
        btn22 = Button(self.face1, command=lambda: self.execute(23), text='cf3', font=ft, bg=bg1)
        btn23 = Button(self.face1, command=lambda: self.execute(24), text='cf4', font=ft, bg=bg1)
        btn24 = Button(self.face1, command=lambda: self.execute(25), text='cf5', font=ft, bg=bg2)
        btn25 = Button(self.face1, command=lambda: self.execute(26), text='cf6', font=ft, bg=bg2)
        btn26 = Button(self.face1, command=lambda: self.execute(27), text='cf7', font=ft, bg=bg1)
        btn27 = Button(self.face1, command=lambda: self.execute(28), text='cf8', font=ft, bg=bg1)
        btn_exit = Button(self.face1, command=on_closing, text='Exit', font=ft_bd, bg=bg2)  # quit和destroy区别
        btn_back = tk.Button(self.face1, command=self.back, text='Back', font=ft_bd, bg=bg2)

        c1, c2 = 1, 2  # 第一个按钮的列下标, 第0列留个canvas
        self.canvas.get_tk_widget().grid(sticky=N + S + W + E, row=0, column=0, rowspan=15, columnspan=c1)
        btn0.grid(sticky=N + S + W + E, row=0, column=c1)
        btn1.grid(sticky=N + S + W + E, row=0, column=c2)
        btn2.grid(sticky=N + S + W + E, row=1, column=c1)
        btn3.grid(sticky=N + S + W + E, row=1, column=c2)
        btn4.grid(sticky=N + S + W + E, row=2, column=c1)
        btn5.grid(sticky=N + S + W + E, row=2, column=c2)
        btn6.grid(sticky=N + S + W + E, row=3, column=c1)
        btn7.grid(sticky=N + S + W + E, row=3, column=c2)
        btn8.grid(sticky=N + S + W + E, row=4, column=c1)
        btn9.grid(sticky=N + S + W + E, row=4, column=c2)
        btn10.grid(sticky=N + S + W + E, row=5, column=c1)
        btn11.grid(sticky=N + S + W + E, row=5, column=c2)
        btn12.grid(sticky=N + S + W + E, row=6, column=c1)
        btn13.grid(sticky=N + S + W + E, row=6, column=c2)
        btn14.grid(sticky=N + S + W + E, row=7, column=c1)
        btn15.grid(sticky=N + S + W + E, row=7, column=c2)
        btn16.grid(sticky=N + S + W + E, row=8, column=c1)
        btn17.grid(sticky=N + S + W + E, row=8, column=c2)
        btn18.grid(sticky=N + S + W + E, row=9, column=c1)
        btn19.grid(sticky=N + S + W + E, row=9, column=c2)
        btn20.grid(sticky=N + S + W + E, row=10, column=c1)
        btn21.grid(sticky=N + S + W + E, row=10, column=c2)
        btn22.grid(sticky=N + S + W + E, row=11, column=c1)
        btn23.grid(sticky=N + S + W + E, row=11, column=c2)
        btn24.grid(sticky=N + S + W + E, row=12, column=c1)
        btn25.grid(sticky=N + S + W + E, row=12, column=c2)
        btn26.grid(sticky=N + S + W + E, row=13, column=c1)
        btn27.grid(sticky=N + S + W + E, row=13, column=c2)
        # btn_exit.grid(sticky=N + S + W + E, row=14, column=c1, columnspan=2)
        btn_back.grid(sticky=N + S + W + E, row=14, column=c1)
        btn_exit.grid(sticky=N + S + W + E, row=14, column=c2)

    def execute(self, func_id):
        """使用OptGAN_fit_gradient_tkinter"""
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from OptGAN_fit_gradient_tkinter import OptGAN_fit_gradient_tkinter
        # import OptGAN_fit_gradient_tkinter
        from cec import CEC
        from utils import plot_line, Log

        test_set = '13'  # 2013年函数集
        all_info = ''
        test_dim = 2
        noise_dim = 10
        alllog = Log('./output/all_log_cec%s' % (test_set))

        if test_dim != 2:
            raise ValueError('Test_dim must be 2 in this program.')

        test_id = func_id  # test_id、 func_id起始1
        log_file = './output/log_cec%s_%d' % (test_set, test_id)
        # 由CEC类创建实例
        fitness_func = CEC('20%s' % test_set, test_id, test_dim)
        fitness_func.print_info()
        func_dim = fitness_func.get_dim()
        optimal = fitness_func.get_optimal()
        gan_o = OptGAN_fit_gradient_tkinter(x_dim=func_dim, noise_dim=noise_dim, fitness_func=fitness_func,
                                            batch_size=32, g_layers=[64], d_layers=[64, 64], log_file=log_file)
        # gan_o.pre_train_D(10000)
        best, train_time = gan_o.train(epochs=20, eval_epochs=1, verbose=True,
                                       root=root, canvas=self.canvas, ax1=self.ax1, ax2=self.ax2)
        """
        val_res, val_time = gan_o.validate(epochs=10000, folds=val_folds)
        # best = plot_line(test_set, test_id)
        try:
            info = 'CEC20%s[%d] optimal %.8g best %.8g dif %.8g train_time %.2f ' \
                   'val_mean %.8g val_std %.8g val_min %.8g val_max %.8g val_median %.8g val_ave_time %.2f' % (
                       test_set, test_id, optimal, best, best - optimal, train_time, val_res.mean(), val_res.std(),
                       val_res.min(), val_res.max(), np.median(val_res), val_time / val_folds if val_folds != 0 else 0)
            print(info)
            all_info += (info + '\n')
        except Exception as e:
            print('print info error. ', e)

        alllog.write(all_info)
        print(all_info)
        """

    def back(self):
        self.face1.destroy()
        InitFace(self.master)


class Experiment:
    """自定义参数实验界面"""
    def __init__(self, master=None):
        self.master = master
        self.master.title('experiment')
        screenwidth = int(self.master.winfo_screenwidth())
        screenheight = int(self.master.winfo_screenheight())
        width, height = 900, 450
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, 0)
        self.master.geometry(alignstr)

        self.face2 = tk.Frame(self.master, )
        self.face2.pack(fill=BOTH, expand=True)
        self.face2.rowconfigure(list(range(4)), weight=1)
        self.face2.columnconfigure(list(range(24)), weight=1)
        self.arrange()

    def arrange(self):
        bg = "#DADADA"  # "#CFE0F7"  # "#B0C4DE"
        # 测试函数标签
        label_func = Label(self.face2, text='测试函数:', font=('华文宋体', 15, 'bold'), bg=bg, width=14)
        label_func.grid(row=0, column=0, columnspan=3, sticky=W+E)
        # 函数下拉列表
        profits = ("sphere", "ellips", "bent_cigar", "discus", 'dif_powers', 'rosenbrock',
                   'schaffer', 'ackley', 'weierstrass', 'griewank', 'rastrigin', 'rotated_rastrigin',
                   'step_rastrigin', 'schwefel', 'rotated_schwefel', 'katsuura', 'bi_rastrigin',
                   'rotated_bi_rastrigin', 'grie_rosen', 'escaffer6', 'cf1', 'cf2', 'cf3', 'cf4',
                   'cf5', 'cf6', 'cf7', 'cf8')
        dicty = dict(zip(profits, range(1, 29)))
        var_name = StringVar()  # 窗体自带的文本，新建一个值
        funclist = ttk.Combobox(self.face2, textvariable=var_name, font=('Times new Roman', 15))
        funclist["values"] = profits
        funclist["state"] = "readonly"  # 只读状态
        funclist.current(0)  # 默认显示第一个
        funclist.grid(row=0, column=3, columnspan=3, sticky=W+E)
        # funclist.get()

        # 维度标签
        label_dim = Label(self.face2, text='测试维度:', font=('华文宋体', 15, 'bold'), bg=bg, width=14)
        label_dim.grid(row=0, column=6, columnspan=3, sticky=W+E)
        # 维度下拉列表
        var_dim = StringVar()
        combobox_dim = ttk.Combobox(self.face2, textvariable=var_dim, font=('Times new Roman', 15))
        combobox_dim["values"] = ("2", "5", "10", "20", "30", "40", "50", "70", "80", "90", "100")
        combobox_dim["state"] = "readonly"
        combobox_dim.grid(row=0, column=9, columnspan=3, sticky=W+E)

        # # 迭代次数标签
        label_epochs = Label(self.face2, text='迭代次数:', font=('华文宋体', 15, 'bold'), bg=bg, width=14)
        label_epochs.grid(row=0, column=12, columnspan=3, sticky=W+E)
        # 迭代次数文本框
        var_epochs = StringVar()
        entry_epochs = Entry(self.face2, textvariable=var_epochs, font=('Times new Roman', 15))
        entry_epochs.grid(row=0, column=15, columnspan=3, sticky=W+E)

        # 运行次数标签
        label_runs = Label(self.face2, text='运行次数:', bg=bg, font=('华文宋体', 15, 'bold'), width=14)
        label_runs.grid(row=0, column=18, columnspan=3, sticky=W+E)
        # 运行次数文本框
        var_runs = StringVar()
        entry_runs = Entry(self.face2, textvariable=var_runs, font=('Times new Roman', 15))
        entry_runs.grid(row=0, column=21, columnspan=3, sticky=W+E)
        # -----------------------------------------------------------------------------

        # 生成解标签
        label_gen = Label(self.face2, text='生成解个数:', bg=bg, font=('华文宋体', 15, 'bold'), width=20)
        label_gen.grid(row=1, column=0, columnspan=4, sticky=W+E)
        # 生成解下拉列表
        var_gen = StringVar()
        combobox_gen = ttk.Combobox(self.face2, textvariable=var_gen, font=('Times new Roman', 15), width=20)
        combobox_gen["values"] = ("0", "5", "10", "15", "20", "25", "30")
        combobox_gen["state"] = "readonly"
        combobox_gen.grid(row=1, column=4, columnspan=4, sticky=W+E)

        # 局部随机解标签
        label_local = Label(self.face2, text='局部随机解个数:', bg=bg, font=('华文宋体', 15, 'bold'), width=20)
        label_local.grid(row=1, column=8, columnspan=4, sticky=W+E)
        # 局部随机解下拉列表
        var_local = StringVar()
        combobox_local = ttk.Combobox(self.face2, textvariable=var_local, font=('Times new Roman', 15, 'bold'), width=20)
        combobox_local["values"] = ("0", "5", "10", "15", "20", "25", "30")
        combobox_local["state"] = "readonly"
        combobox_local.grid(row=1, column=12, columnspan=4, sticky=W+E)

        # 全局随机解
        label_global = Label(self.face2, text='全局随机解个数:', bg=bg, font=('华文宋体', 15, 'bold'), width=20)
        label_global.grid(row=1, column=16, columnspan=4, sticky=W+E)
        # 全局随机解下拉列表
        var_global = StringVar()
        combobox_global = ttk.Combobox(self.face2, textvariable=var_global, font=('Times new Roman', 15), width=20)
        combobox_global["values"] = ("0", "5", "10", "15", "20", "25", "30")
        combobox_global["state"] = "readonly"
        combobox_global.grid(row=1, column=20, columnspan=4, sticky=W+E)
        # -------------------------------------------------------------------------------

        # 当前次数标签
        label_cur_run = Label(self.face2, text="当前次数:", bg=bg, font=('华文宋体', 15, 'bold'), width=20)
        label_cur_run.grid(row=2, column=0, columnspan=4, sticky=W+E)
        var_cur_run_val = StringVar()
        label_cur_epoch_val = Label(self.face2, textvariable=var_cur_run_val, font=('Times new Roman', 15, 'bold'), width=15,
                                    bg="White", fg="red")
        label_cur_epoch_val.grid(row=2, column=4, columnspan=4, sticky=W+E)

        # 均值标签
        label_mean = Label(self.face2, text="均值:", bg=bg, font=('华文宋体', 15, 'bold'), width=20)
        label_mean.grid(row=2, column=8, columnspan=4, sticky=W+E)
        var_mean_val = StringVar()
        label_mean_val = Label(self.face2, textvariable=var_mean_val, font=('Times new Roman', 15, 'bold'), width=15,
                               bg="White", fg="red")
        label_mean_val.grid(row=2, column=12, columnspan=4, sticky=W+E)

        # 标准差标签
        label_std = Label(self.face2, text="标准差：", bg=bg, font=('华文宋体', 15, 'bold'), width=20)
        label_std.grid(row=2, column=16, columnspan=4, sticky=W+E)
        var_std_val = StringVar()
        label_std_val = Label(self.face2, textvariable=var_std_val, font=('Times new Roman', 15, 'bold'), width=15,
                              bg="White", fg="red")
        label_std_val.grid(row=2, column=20, columnspan=4, sticky=W+E)

        # 执行按钮
        btn_run = Button(self.face2, text="运行", font=('华文宋体', 20, 'bold'), bg=bg, height=-5,
                         command=lambda: self.execute(res_tuple=(var_cur_run_val, var_mean_val, var_std_val),
                                                      func_id=dicty[funclist.get()],
                                                      dim=int(var_dim.get()),
                                                      epochs=int(var_epochs.get()),
                                                      runs=int(var_runs.get()),
                                                      g_sample_times=int(var_gen.get()),
                                                      local_random_sample_times=int(var_local.get()),
                                                      gloabal_sample_times=int(var_global.get())))
        btn_back = Button(self.face2, text='Back', font=('Times new Roman', 20), bg=bg, command=self.back)
        btn_exit = Button(self.face2, text='Exit', font=('Times new Roman', 20), bg=bg, command=on_closing)
        btn_run.grid(row=3, columnspan=24)
        btn_back.grid(row=4, column=0, columnspan=12, sticky=W+E)
        btn_exit.grid(row=4, column=12, columnspan=12, sticky=W+E)

    def execute(self, res_tuple, func_id=1, dim=2, epochs=10, runs=51,
                g_sample_times=15, local_random_sample_times=10, gloabal_sample_times=5):
        """
        使用OptGAN_fit_gradient_tkinter_ui
        待传参数，函数id、维度、epochs、生成解个数、局部随机解个数、全局随机解个数
        """
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from OptGAN_fit_gradient_ui import OptGAN_fit_gradient_ui
        from cec import CEC
        from utils import plot_line, Log

        test_set = '13'  # 2013年函数集
        all_info = ''
        noise_dim = 10
        alllog = Log('./output/all_log_cec%s' % test_set)

        if dim < 1:
            raise ValueError('Dimension must be a positive integer(维度必须是正整数)')

        test_id = func_id  # test_id、 func_id起始1
        log_file = './output/log_cec%s_%d' % (test_set, test_id)
        # 由CEC类创建实例
        fitness_func = CEC('20%s' % test_set, test_id, dim)
        fitness_func.print_info()
        print('epochs:%d  runs:%d  gen_samples:%d  local_samples:%d  global_sample:%d'
              % (epochs, runs, g_sample_times/5, local_random_sample_times/5, gloabal_sample_times/5))
        func_dim = fitness_func.get_dim()
        optimal = fitness_func.get_optimal()
        gan_o = OptGAN_fit_gradient_ui(x_dim=func_dim, noise_dim=noise_dim, fitness_func=fitness_func,
                                       batch_size=32, g_layers=[64], d_layers=[64, 64], log_file=log_file,
                                       g_sample_times=int(g_sample_times / 5),
                                       local_random_sample_times=int(local_random_sample_times / 5),
                                       gloabal_sample_times=int(gloabal_sample_times / 5),
                                       root=self.face2)  # 使其在训练中能够强制刷新
        # gan_o.pre_train_D(10000)
        # best, train_time = gan_o.train(epochs=epochs, eval_epochs=1, verbose=True)
        val_res, val_time = gan_o.validate(res_tuple, folds=runs, epochs=epochs, eval_epochs=1)
        print('-----------------------------------------------------')
        print(val_res.mean(), val_res.std())
        return val_res.mean(), val_res.std()

    def back(self):
        self.face2.destroy()
        InitFace(self.master)


if __name__ == '__main__':
    root = Tk()
    InitFace(root)
    root.protocol('WM_DELETE_WINDOW', on_closing)
    root.mainloop()
