from keras.layers import Input, Dense, Dropout, Lambda, Concatenate, Subtract, Add, Multiply, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import time
from utils import Log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

MAX_STEP_LEN = 50
MIN_STEP_LEN = 0
MID_STEP_LEN = 1e-2
# for power: 1e-500
# for other: 1e-20
EPSILON = 1e-8


class OptGAN_fit_gradient_matplotlib(object):
    """
    define GAN class for optimization
    """

    def __init__(self, x_dim, noise_dim, fitness_func, batch_size, g_layers, d_layers, log_file):
        self.x_dim = x_dim
        self.noise_dim = noise_dim
        self.fitness_func = fitness_func
        self.func_name = self.fitness_func.get_func_name()
        self.func_name_title = self.fitness_func.get_func_name_title()  # 补充
        self.optimal = self.fitness_func.get_optimal()
        self.lb = self.fitness_func.get_lb()
        self.ub = self.fitness_func.get_ub()
        self.batch_size = batch_size
        self.best_sample_num = 5  # 类似种群规模
        self.best_samples = None  # 补充，否则会有warning，所有属性在init中初始化
        self.best_v = None  # 补充，否则会有warning，所有属性在init中初始化
        self.g_sample_times = 3  # 生成样本数
        self.local_random_sample_times = 2  # 局部随机样本数
        self.global_sample_times = 1  # 全局随机样本数
        self.g_sample_times_val = 3
        self.local_random_sample_times_val = 2
        self.global_sample_times_val = 1
        self.log = Log(log_file)
        self.mesh_xyz = None  # 补充，在train中由_init_mesh_xyz初始化
        self.linsize = 11  # 补充，meshgrid划分网格数目， 100/n+1：21 41 51 101

        # step_len params
        # choices: [ ratio, power, exponent, combine, self_adapt ]
        self.step_reduce_method = 'power'  # 指定方法为幂函数
        self.step_len = None  # 引导向量长度，由函数_init_step_len初始化
        self.step_epoch_cnt = 0
        self.step_reduce_dif = None
        self.step_max_epochs = None
        self.step_last_best = None
        self.step_nochange_epochs = 0
        self.step_std_ratio = 0.0
        # for ratio method
        self.step_reduce_ratio = 0.6
        self.step_reduce_epochs = 100
        # for power method
        self.step_reduce_grade = 4.5
        # for self_adapt mothod
        self.step_adapt_ratio = 0.5
        self.step_adapt_epochs = 50

        optimizer = Adam(0.001)

        # build discriminator
        self.D = self._build_discriminator(x_dim, d_layers)
        self.D.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
        # print('discriminator.summary:')
        # self.D.summary()

        # build generator
        self.G = self._build_generator(x_dim, noise_dim, g_layers)
        self.G.compile(loss='binary_crossentropy', optimizer=optimizer)  # 编译模型，指定损失函数、优化器
        # print('generator.summary:')
        # self.G.summary()  # 打印模型概况

        # build combined
        self.D.trainable = False
        x = Input(shape=(x_dim,))
        z = Input(shape=(noise_dim,))
        step_len = Input(shape=(1,))
        motion = self.G([x, z, step_len])
        x_gen = Add()([x, motion])
        x_gen = Lambda(lambda x: K.clip(x, self.lb, self.ub))(x_gen)
        prediction = self.D([x, x_gen])
        self.combined = Model([x, z, step_len], prediction)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
        # print('combined.summary:')
        # self.combined.summary()

    def _build_generator(self, x_dim, noise_dim, g_layers):
        x = Input(shape=(x_dim,))
        z = Input(shape=(noise_dim,))
        step_len = Input(shape=(1,))
        input = Concatenate()([x, z])
        if len(g_layers) <= 0:  # 无隐藏层，输入x+noise，输出grad
            output = Dense(x_dim, input_shape=(x_dim + noise_dim,), activation='tanh')(input)
        else:
            output = Dense(g_layers[0], input_shape=(x_dim + noise_dim,), activation='relu')(input)
            for layer in g_layers[1:]:
                output = Dense(layer, activation='relu')(output)
            output = Dense(x_dim, activation='tanh')(output)
        # output = Lambda(lambda x: K.clip(x, -1, 1))(output)
        motion = Multiply()([output, step_len])  # 引导向量=方向向量*长度
        return Model(inputs=[x, z, step_len], outputs=motion)

    def _build_discriminator(self, x_dim, d_layers):
        x1 = Input(shape=(x_dim,))
        x2 = Input(shape=(x_dim,))
        model = Sequential()  # 序贯模型, FC1
        if len(d_layers) <= 0:
            model.add(Dense(10, input_shape=(x_dim,), activation='relu'))
        else:
            model.add(Dense(d_layers[0], input_shape=(x_dim,), activation='relu'))
            for layer in d_layers[1:]:
                model.add(Dense(layer, activation='relu'))
            model.add(Dense(10, activation='relu'))  # FC1输出层，10
        # model.summary()
        x1_fitness = model(x1)
        x2_fitness = model(x2)
        output = Subtract()([x1_fitness, x2_fitness])
        output = Dense(10, activation='relu')(output)
        output = Dense(1, activation='sigmoid')(output)
        return Model(inputs=[x1, x2], outputs=output)

    def pre_train_D(self, sample_num):
        lb = self.lb
        ub = self.ub
        # 均匀分布的随机数
        x = lb + (ub - lb) * np.random.random_sample((sample_num, self.x_dim))
        grad = (np.random.random_sample((sample_num, self.x_dim)) - 0.5) * 5
        gen_x = np.add(x, grad)  # 生成解
        gen_x = np.clip(gen_x, self.lb, self.ub)  # 截取
        x_fitness = self.fitness_func.get_fitness(x)
        gen_x_fitness = self.fitness_func.get_fitness(gen_x)
        labels = np.asarray(x_fitness > gen_x_fitness, dtype=int)
        # for i in range(len(x)):
        #     print('x:%s - fit=%f | gen_x:%s - fit=%f | labels=%d' % (
        #         x[i, :], x_fitness[i], gen_x[i, :], gen_x_fitness[i], labels[i]))

        self.D.fit(x=[x, gen_x], y=labels, verbose=1, shuffle=True)

    def train(self, epochs, eval_epochs, verbose):
        topk = 5
        start_time = time.time()
        self.best_samples = self.global_random_sample(1)  # 下文的自定义函数
        self.best_samples = self.best_samples[self.best_samples[:, -1].argsort()].copy()  # 按函数值升序排序并存储
        self.best_v = self.best_samples[0, -1]
        self._init_step_len(epochs=epochs)
        self.mesh_xyz = self._init_mesh_xyz()

        vbest_old = None
        ymax = None
        plt.close("all")
        plt.ion()
        """
        ax1 = plt.subplot(1, 1, 1)     # 隐式创建画布，并创建子图
        fig2 = plt.figure(2)  # 创建画布画布
        ax2 = Axes3D(fig2)  # 创建Axes3D对象，子图
        # ax2 = fig2.add_subplot(1, 1, 1, projection="3d")
        """
        """
        plt.figure(1)
        ax1 = plt.subplot(1, 1, 1)
        plt.figure(2)
        ax2 = plt.subplot(1, 1, 1, projection="3d")
        """
        """
        # 子图，1:1
        plt.figure(num=2, figsize=(14, 14))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2, projection="3d")
        """
        # 自定义子图位置
        fig = plt.figure(num=2, figsize=(5, 4))
        gs = gridspec.GridSpec(1, 5)
        ax1 = plt.subplot(gs[0, 0:2])
        ax2 = plt.subplot(gs[0, 2:], projection='3d')
        plt.tight_layout()
        # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.1)
        for epoch in range(epochs + 1):  # +1使得画图能到10000
            # ################### sample from G and normal_distribution ####################
            g_samples = self.g_sample(self.g_sample_times)
            rand_samples = self.local_random_sample(self.local_random_sample_times)
            all_samples = np.concatenate((g_samples, rand_samples), 0)  # 前n+1个元素为当前解和函数值

            # ################### train D ####################
            labels = np.asarray(all_samples[:, self.x_dim] > all_samples[:, self.x_dim * 2 + 1], dtype=int)
            d_his = self.D.fit(x=[all_samples[:, :self.x_dim], all_samples[:, self.x_dim + 1:self.x_dim * 2 + 1]],
                               y=labels, verbose=0, shuffle=True).history

            # ################### retain best_sample ####################
            all_samples = all_samples[:, self.x_dim + 1:].copy()  # 只取后一个
            global_samples = self.global_random_sample(self.global_sample_times)
            all_samples = np.concatenate((all_samples, global_samples), 0)
            all_samples = np.concatenate((self.best_samples, all_samples), 0)
            all_samples = all_samples[all_samples[:, -1].argsort()]
            self.best_samples = all_samples[:self.best_sample_num, :].copy()  # 总是选最好的前五个，且第一个为五个中的最优

            # ################### train G ####################
            for i in range(1):
                z = np.random.normal(0, 1, (self.best_sample_num, self.noise_dim))
                step_lens = np.random.normal(self.step_len, self.step_len * self.step_std_ratio,
                                             size=(self.best_sample_num, 1))
                step_lens = np.clip(step_lens, 0, 100)
                labels = np.ones((self.best_sample_num, 1))  # 全一
                g_his = self.combined.fit(x=[self.best_samples[:, :-1], z, step_lens], y=labels, verbose=0,
                                          shuffle=True).history
                # print('train_G: acc=%f  loss=%f' % (g_his['binary_accuracy'][0], g_his['loss'][0]))

            # ################### write log and show best ####################
            if verbose and epoch % eval_epochs == 0:
                info = '\ttrain epoch:%4d func_name:%s optimal=%.8g best_v=%.8g best_dif=%.8g time=%.2f step_len=%.5g' % (
                    epoch, self.func_name, self.optimal, self.best_samples[0, -1],
                    self.best_samples[0, -1] - self.optimal, time.time() - start_time, self.step_len
                )
                self.log.write(info)
                print(info)

                # 训练过程，绘图，横坐标为epoch，纵坐标best_v
                # plt.figure(1)
                if 0 == epoch:
                    vbest_old = self.best_samples[0, -1]
                    ymax = vbest_old
                    zmax = self.best_samples[self.best_sample_num - 1, -1]
                else:
                    vbest_new = self.best_samples[0, -1]
                    ax1.plot([epoch - eval_epochs, epoch], [vbest_old, vbest_new], 'r-')
                    # plt.xlim((0, epochs))
                    # plt.ylim((self.optimal, ymax))
                    ax1.axis([0, epochs, self.optimal, ymax])
                    vbest_old = vbest_new
                    # plt.pause(0.01)  # 不然画得太快会卡住，例如某一图还没来得及画又画下一图

                # 第二个图的更新部分
                ax2.cla()
                countour = ax2.contour(self.mesh_xyz[0], self.mesh_xyz[1], self.mesh_xyz[2],
                                       levels=15, zdir='z', offset=0, cmap='rainbow', alpha=1)
                # ax2.clabel(countour, levels=15, inline=True, fontsize=10)
                # plt.colorbar(countour, shrink=0.5, aspect=5)
                ax2.plot_wireframe(self.mesh_xyz[0], self.mesh_xyz[1], self.mesh_xyz[2], alpha=1,
                                   rcount=self.linsize, ccount=self.linsize, color='dodgerblue')

                # surf = ax2.plot_surface(self.mesh_xyz[0], self.mesh_xyz[1], self.mesh_xyz[2], cmap='rainbow')
                # shrink伸缩比例0-1， aspect颜色条宽度（反比例， 一个颜色条宽度为1/aspect）
                # fig2.colorbar(surf, shrink=0.5, aspect=5)
                ax2.scatter(self.best_samples[:, 0], self.best_samples[:, 1], self.best_samples[:, 2],
                            color='r', alpha=1)
                ax2.view_init(50, -60)
                ax2.set_xlabel('x1')
                ax2.set_ylabel('x2')
                ax2.set_title(self.func_name_title)
                ax2.axis([-100, 100, -100, 100])
                # ax2.set_zlim(self.optimal, zmax)
                plt.pause(0.01)  # 不然画得太快会卡住，例如某一图还没来得及画又画下一图

                # print('top %d from best_samples:' % topk)
                # self._print_topk(self.best_samples, topk)
                # print('top %d from g_samples before training G:' % topk)
                # self._print_topk(g_samples, topk)
                # new_g_sample = self.g_sample(batch_num=1, fitness=fitness_method)
                # print('top %d from g_samples after training G:' % topk)
                # self._print_topk(new_g_sample, topk)

                # print('top %d from all random_samples:' % topk)
                # self._print_topk(ranom_samples, topk)

            # ################### save best_v and judge whether to break ####################
            self.best_v = self.best_samples[0, -1]
            if self.best_v - self.optimal < EPSILON or self.step_len < MIN_STEP_LEN:
                break

            # ################### reduce step_len ####################
            self._recuce_step_len()  # 更新(减小)引导向量长度step_len
        plt.ioff()
        plt.show()
        info = 'Finish train func_name:%s optimal=%.8g best_v=%.8g best_dif=%.8g time=%.2f' % (
            self.func_name, self.optimal, self.best_v, self.best_v - self.optimal, time.time() - start_time)
        self.log.write(info)
        print(info)
        return self.best_v, time.time() - start_time

    def validate(self, epochs, folds):
        res = np.zeros(folds)
        total_time = time.time()

        for k in range(folds):  # folds=51
            start_time = time.time()
            self.best_samples = self.global_random_sample(1)
            self.best_samples = self.best_samples[self.best_samples[:, -1].argsort()].copy()
            self.best_v = self.best_samples[0, -1]
            self._init_step_len(epochs=epochs)
            for epoch in range(epochs):
                #################### sample new x ####################
                g_samples = self.g_sample(self.g_sample_times_val)
                rand_samples = self.local_random_sample(self.local_random_sample_times_val)
                all_samples = np.concatenate((g_samples, rand_samples), 0)
                all_samples = all_samples[:, self.x_dim + 1:].copy()
                global_samples = self.global_random_sample(self.global_sample_times_val)
                all_samples = np.concatenate((all_samples, global_samples), 0)

                #################### retain best_sample ####################
                all_samples = np.concatenate((self.best_samples, all_samples), 0)
                all_samples = all_samples[all_samples[:, -1].argsort()]
                self.best_samples = all_samples[:self.best_sample_num, :].copy()

                ################### write log and show best ####################
                if epoch % 200 == 0:  # 验证时200次评估一次
                    info = '\tvalidate epoch:%4d func_name:%s optimal=%.8g best_v=%.8g best_dif=%.8g time=%.2f step_len=%.5g' % (
                        epoch, self.func_name, self.optimal, self.best_samples[0, -1],
                        self.best_samples[0, -1] - self.optimal, time.time() - start_time, self.step_len
                    )
                    self.log.write(info)
                    print(info)

                #################### save best_v and judge whether to break ####################
                self.best_v = self.best_samples[0, -1]
                if self.best_v - self.optimal < EPSILON or self.step_len < MIN_STEP_LEN:
                    break

                #################### reduce step_len ####################
                self._recuce_step_len()
            res[k] = self.best_v - self.optimal
            info = 'fold:%d func_name:%s optimal=%.8g best_v=%.8g best_dif=%.8g time=%.2f' % (
                k, self.func_name, self.optimal, self.best_v, self.best_v - self.optimal, time.time() - start_time)
            self.log.write(info)
            print(info)
        info = 'Finish validate mean %.8g std %.8g ave_time %.2f' % (
            res.mean(), res.std(), (time.time() - total_time) / folds if folds != 0 else 0)
        self.log.write(info)
        print(info)
        return res, time.time() - total_time

    def g_sample(self, times):
        """
        生成解，times = 3
        """
        res = np.zeros(shape=(0, (self.x_dim + 1) * 2))
        for _ in range(times):
            z = np.random.normal(0, 1, (self.best_sample_num, self.noise_dim))
            step_lens = np.random.normal(self.step_len, self.step_len * self.step_std_ratio,
                                         size=(self.best_sample_num, 1))
            step_lens = np.clip(step_lens, 0, 100)
            # print('max=%f  min=%f' % (np.max(step_lens), np.min(step_lens)))
            grad = self.G.predict([self.best_samples[:, :-1], z, step_lens])  # 生成网络产生引导向量
            gen_x = np.add(self.best_samples[:, :-1], grad)
            gen_x = np.clip(gen_x, self.lb, self.ub)
            gen_x_fitness = self.fitness_func.get_fitness(gen_x)
            if res is None:
                res = np.c_[self.best_samples, gen_x, gen_x_fitness].copy()
            else:
                res = np.r_[res, np.c_[self.best_samples, gen_x, gen_x_fitness]].copy()
        return res

    def local_random_sample(self, times):
        """局部随机样本，解的范围[-steplen, +],    方向向量随机，个数times=2"""
        res = np.zeros(shape=(0, (self.x_dim + 1) * 2))
        for _ in range(times):
            grad = (np.random.random_sample((self.best_sample_num, self.x_dim)) * 2 - 1)  # 方向向量随机[-1,1]
            step_lens = np.random.normal(self.step_len, self.step_len * self.step_std_ratio,
                                         size=(self.best_sample_num, 1))
            step_lens = np.clip(step_lens, 0, 100)
            # print('max=%f  min=%f' % (np.max(step_lens), np.min(step_lens)))
            grad = np.multiply(grad, step_lens)
            gen_x = np.add(self.best_samples[:, :-1], grad)
            gen_x = np.clip(gen_x, self.lb, self.ub)
            gen_x_fitness = self.fitness_func.get_fitness(gen_x)
            if res is None:
                res = np.c_[self.best_samples, gen_x, gen_x_fitness].copy()
            else:
                res = np.r_[res, np.c_[self.best_samples, gen_x, gen_x_fitness]].copy()
        return res

    def global_random_sample(self, times):
        """全局随机样本，解的范围[lb, ub]，方向向量随机， times=1"""
        lb = self.lb
        ub = self.ub
        res = np.zeros(shape=(0, self.x_dim + 1))  # 解+函数值
        for _ in range(times):
            gen_x = lb + (ub - lb) * np.random.random_sample((self.best_sample_num, self.x_dim))
            gen_x = np.clip(gen_x, self.lb, self.ub)
            gen_x_fitness = self.fitness_func.get_fitness(gen_x)
            if res is None:
                res = np.c_[gen_x, gen_x_fitness].copy()  # 两矩阵按行合并
            else:
                res = np.r_[res, np.c_[gen_x, gen_x_fitness]].copy()  # 两矩阵按列合并
        return res

    def _init_step_len(self, epochs):
        """初始化引导向量长度的相关信息"""
        self.step_len = MAX_STEP_LEN
        if self.step_reduce_method == 'power':
            self.step_reduce_dif = (np.power(MAX_STEP_LEN, 1 / self.step_reduce_grade) -
                                    np.power(MIN_STEP_LEN, 1 / self.step_reduce_grade)) / epochs
        elif self.step_reduce_method == 'ratio':
            self.step_epoch_cnt = 0
        elif self.step_reduce_method == 'exponent':
            self.step_reduce_dif = (np.log(MAX_STEP_LEN) - np.log(MIN_STEP_LEN)) / epochs
        elif self.step_reduce_method == 'combine':
            self.step_epoch_cnt = 0
            self.step_max_epochs = epochs
            self.step_reduce_dif = (np.power(MAX_STEP_LEN, 1 / self.step_reduce_grade) -
                                    np.power(MID_STEP_LEN, 1 / self.step_reduce_grade)) / (epochs / 2)
        elif self.step_reduce_method == 'self_adapt':
            self.step_nochange_epochs = 0
            self.step_last_best = self.best_v.copy()
        else:
            raise Exception('wrong param self.step_reduce_method.')

    def _recuce_step_len(self):
        """更新(减小)引导向量长度"""
        if self.step_reduce_method == 'power':
            # numpy中power或**运算时，底数为负数时幂指数不能是小数。在这里当epochs较小时，相减会产生负数
            self.step_len = np.power(np.fabs(np.power(self.step_len, 1 / self.step_reduce_grade) - self.step_reduce_dif),
                                     self.step_reduce_grade)  # 幂次
            # self.step_len = 10
        elif self.step_reduce_method == 'ratio':
            if self.step_epoch_cnt % self.step_reduce_epochs == 0:
                self.step_len = self.step_len * self.step_reduce_ratio
            self.step_epoch_cnt += 1
        elif self.step_reduce_method == 'exponent':
            self.step_len = np.exp(np.log(self.step_len) - self.step_reduce_dif)
        elif self.step_reduce_method == 'combine':
            if self.step_epoch_cnt == (self.step_max_epochs / 2):
                self.step_reduce_dif = (np.log(10 * MID_STEP_LEN) - np.log(MIN_STEP_LEN)) / (self.step_max_epochs / 2)
                self.step_len = 10 * MID_STEP_LEN
            elif self.step_epoch_cnt < (self.step_max_epochs / 2):
                self.step_len = np.power(np.power(self.step_len, 1 / self.step_reduce_grade) - self.step_reduce_dif,
                                         self.step_reduce_grade)
            else:
                self.step_len = np.exp(np.log(self.step_len) - self.step_reduce_dif)
            self.step_epoch_cnt += 1
        elif self.step_reduce_method == 'self_adapt':
            if abs(self.best_v - self.step_last_best) < EPSILON:
                self.step_nochange_epochs += 1
                if self.step_nochange_epochs > self.step_adapt_epochs:
                    self.step_len *= self.step_adapt_ratio
                    self.step_nochange_epochs = 0
            else:
                self.step_nochange_epochs = 0
            self.step_last_best = self.best_v.copy()
        else:
            raise Exception('wrong param self.step_reduce_method.')

    def _print_topk(self, data, k):
        data = data.copy()
        data = data[data[:, -1].argsort()]
        for i in range(k):
            x = data[i, :-1]
            y = data[i, -1]
            if x.shape[0] > 5:
                x = x[:5]
            print("\tbest %d: x=%s  fitness=%f" % (i, x, y))

    def _init_mesh_xyz(self):
        x = np.linspace(self.lb, self.ub, int(self.ub - self.lb + 1))
        x, y = np.meshgrid(x, x)
        (sizem, sizen) = x.shape
        z = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1), y.reshape(y.shape[0] * y.shape[1], 1)))
        z = self.fitness_func.get_fitness(z)
        z = z.reshape(sizem, sizen)
        return x, y, z
