from OptGAN_fit_gradient import OptGAN_fit_gradient
from cec import CEC
from utils import plot_line, Log
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

test_set = '13'  # 2013年函数集
all_info = ''
test_dim = 30
noise_dim = 10
val_folds = 30  # 验证次数51
alllog = Log('./output/all_log_cec%s' % (test_set))

for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27]:
# for i in range(28):
    test_id = i + 1
    log_file = './output/log_cec%s_%d' % (test_set, test_id)
    # 由CEC类创建实例
    fitness_func = CEC('20%s' % test_set, test_id, test_dim)
    # fitness_func.print_info()
    func_dim = fitness_func.get_dim()
    optimal = fitness_func.get_optimal()
    gan_o = OptGAN_fit_gradient(x_dim=func_dim, noise_dim=noise_dim, fitness_func=fitness_func, batch_size=32,
                                g_layers=[64], d_layers=[64, 64], log_file=log_file)
    # gan_o.pre_train_D(10000)    # 先让判别器达到最优
    # best, train_time = gan_o.train(epochs=10000, eval_epochs=50, verbose=True)

    val_res, val_time = gan_o.validate(epochs=10000, folds=val_folds)
    """
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