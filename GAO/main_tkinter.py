from tkinter import *
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import threading


def execute(func_id):
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

    test_id = func_id     # test_id、 func_id起始1
    log_file = './output/log_cec%s_%d' % (test_set, test_id)
    # 由CEC类创建实例
    fitness_func = CEC('20%s' % test_set, test_id, test_dim)
    fitness_func.print_info()
    func_dim = fitness_func.get_dim()
    optimal = fitness_func.get_optimal()
    gan_o = OptGAN_fit_gradient_tkinter(x_dim=func_dim, noise_dim=noise_dim, fitness_func=fitness_func, batch_size=32,
                                        g_layers=[64], d_layers=[64, 64], log_file=log_file)
    # gan_o.pre_train_D(10000)
    best, train_time = gan_o.train(epochs=20, eval_epochs=1, verbose=True,
                                   root=root, canvas=canvas, ax1=ax1, ax2=ax2)
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


def thread_it(func, *args):
    """将函数打包进线程"""
    # 创建
    t = threading.Thread(target=func, args=args)
    # 守护 !!!
    t.setDaemon(True)
    # 启动
    t.start()
    # 阻塞--卡死界面！
    # t.join()


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()


root = Tk()
root.title('test')
screenwidth = int(root.winfo_screenwidth())
screenheight = int(root.winfo_screenheight())
width, height = 900, 450
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, 0)
root.geometry(alignstr)
root.rowconfigure(list(range(15)), weight=1)
root.columnconfigure(list(range(3)), weight=1)

c1, c2 = 1, 2     # 第一个按钮的列下标, 第0列留个canvas
# plt.close("all")
# 创建画布
fig = plt.figure()
plt.ion()  # 切换交互模式
gs = gridspec.GridSpec(1, 5)
ax1 = plt.subplot(gs[0, 0:2])
ax2 = plt.subplot(gs[0, 2:], projection='3d')
ax1.grid()
# plt.tight_layout()
plt.subplots_adjust(left=0.08, bottom=0.09, right=0.95, top=0.96, wspace=0.01, hspace=0.1)
plt.ioff()
canvas = FigureCanvasTkAgg(fig, master=root)     # 只需要设置宽度，由于sticky=N+S+W+E，高度会自动对齐
canvas.draw()


# 包装定位canvas
# canvas.grid(sticky=N+S+W+E, row=0, column=0, rowspan=15, columnspan=c1)
canvas.get_tk_widget().grid(sticky=N+S+W+E, row=0, column=0, rowspan=15, columnspan=c1)
"""
# 将matplotlib的左下角导航工具栏显示出来
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
# canvas.get_tk_widget().grid()
"""

# 创建按钮，anchor设置控件内文字在控件内的位置
btn0 = Button(root, width=20, command=lambda: execute(1), text='sphere')
btn1 = Button(root, width=20, command=lambda: execute(2), text='ellips')
btn2 = Button(root, command=lambda: execute(3), text='bent_cigar')
btn3 = Button(root, command=lambda: execute(4), text='discus')
btn4 = Button(root, command=lambda: execute(5), text='dif_powers')
btn5 = Button(root, command=lambda: execute(6), text='rosenbrock')
btn6 = Button(root, command=lambda: execute(7), text='schaffer')
btn7 = Button(root, command=lambda: execute(8), text='ackley')
btn8 = Button(root, command=lambda: execute(9), text='weierstrass')
btn9 = Button(root, command=lambda: execute(10), text='griewank')
btn10 = Button(root, command=lambda: execute(11), text='rastrigin')
btn11 = Button(root, command=lambda: execute(12), text='rotated_rastrigin')
btn12 = Button(root, command=lambda: execute(13), text='step_rastrigin')
btn13 = Button(root, command=lambda: execute(14), text='schwefel')
btn14 = Button(root, command=lambda: execute(15), text='rotated_schwefel')
btn15 = Button(root, command=lambda: execute(16), text='katsuura')
btn16 = Button(root, command=lambda: execute(17), text='bi_rastrigin')
btn17 = Button(root, command=lambda: execute(18), text='rotated_bi_rastrigin')
btn18 = Button(root, command=lambda: execute(19), text='grie_rosen')
btn19 = Button(root, command=lambda: execute(20), text='escaffer6')
btn20 = Button(root, command=lambda: execute(21), text='cf1')
btn21 = Button(root, command=lambda: execute(22), text='cf2')
btn22 = Button(root, command=lambda: execute(23), text='cf3')
btn23 = Button(root, command=lambda: execute(24), text='cf4')
btn24 = Button(root, command=lambda: execute(25), text='cf5')
btn25 = Button(root, command=lambda: execute(26), text='cf6')
btn26 = Button(root, command=lambda: execute(27), text='cf7')
btn27 = Button(root, command=lambda: execute(28), text='cf8')
btn_exit = Button(root, command=on_closing, text='Exit')     # quit和destroy区别

# 按钮包装定位，sticky设置控件的对齐方式
btn0.grid(sticky=N+S+W+E, row=0, column=c1)
btn1.grid(sticky=N+S+W+E, row=0, column=c2)
btn2.grid(sticky=N+S+W+E, row=1, column=c1)
btn3.grid(sticky=N+S+W+E, row=1, column=c2)
btn4.grid(sticky=N+S+W+E, row=2, column=c1)
btn5.grid(sticky=N+S+W+E, row=2, column=c2)
btn6.grid(sticky=N+S+W+E, row=3, column=c1)
btn7.grid(sticky=N+S+W+E, row=3, column=c2)
btn8.grid(sticky=N+S+W+E, row=4, column=c1)
btn9.grid(sticky=N+S+W+E, row=4, column=c2)
btn10.grid(sticky=N+S+W+E, row=5, column=c1)
btn11.grid(sticky=N+S+W+E, row=5, column=c2)
btn12.grid(sticky=N+S+W+E, row=6, column=c1)
btn13.grid(sticky=N+S+W+E, row=6, column=c2)
btn14.grid(sticky=N+S+W+E, row=7, column=c1)
btn15.grid(sticky=N+S+W+E, row=7, column=c2)
btn16.grid(sticky=N+S+W+E, row=8, column=c1)
btn17.grid(sticky=N+S+W+E, row=8, column=c2)
btn18.grid(sticky=N+S+W+E, row=9, column=c1)
btn19.grid(sticky=N+S+W+E, row=9, column=c2)
btn20.grid(sticky=N+S+W+E, row=10, column=c1)
btn21.grid(sticky=N+S+W+E, row=10, column=c2)
btn22.grid(sticky=N+S+W+E, row=11, column=c1)
btn23.grid(sticky=N+S+W+E, row=11, column=c2)
btn24.grid(sticky=N+S+W+E, row=12, column=c1)
btn25.grid(sticky=N+S+W+E, row=12, column=c2)
btn26.grid(sticky=N+S+W+E, row=13, column=c1)
btn27.grid(sticky=N+S+W+E, row=13, column=c2)
btn_exit.grid(sticky=N+S+W+E, row=14, column=c1, columnspan=2)

root.protocol('WM_DELETE_WINDOW', on_closing)
root.mainloop()
