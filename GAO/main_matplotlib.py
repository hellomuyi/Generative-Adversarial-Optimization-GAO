import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from OptGAN_fit_gradient_matplotlib import OptGAN_fit_gradient_matplotlib
from cec import CEC
from utils import plot_line, Log

test_set = '13'  # 2013年函数集
all_info = ''
test_dim = 2
noise_dim = 10
val_folds = 1  # 验证次数51
alllog = Log('./output/all_log_cec%s' % (test_set))

if test_dim != 2:
    raise ValueError('Test_dim must be 2 in this program.')

# for i in range(28):
func_id = 1     # 起始为1
for i in [func_id-1]:
    test_id = i + 1
    log_file = './output/log_cec%s_%d' % (test_set, test_id)
    # 由CEC类创建实例
    fitness_func = CEC('20%s' % test_set, test_id, test_dim)
    fitness_func.print_info()
    func_dim = fitness_func.get_dim()
    optimal = fitness_func.get_optimal()
    gan_o = OptGAN_fit_gradient_matplotlib(x_dim=func_dim, noise_dim=noise_dim, fitness_func=fitness_func, batch_size=32,
                                g_layers=[64], d_layers=[64, 64], log_file=log_file)
    # gan_o.pre_train_D(10000)
    best, train_time = gan_o.train(epochs=20, eval_epochs=1, verbose=True)
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