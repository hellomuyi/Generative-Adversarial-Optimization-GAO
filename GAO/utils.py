import keras.backend as K

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# def print_topk(data, k):
#     for i in range(k):
#         y = data[i, 0]
#         v = data[i, 1]
#         z = data[i, 2:2 + self.noise_dim]
#         x = data[i, -self.input_dim:]
#         print("best %d: y=%d  v=%.8f  z=%s -> x=%s" % (i + 1, y, v, z, x))



class Log():
    def __init__(self, file_name):
        self.file_name = file_name
        self.file = open(file_name, 'w')
        self.file.write('\n' + '#' * 100 + '\n')
        self.file.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.file.write('\n' + '#' * 100 + '\n')

    def write(self, msg):
        self.file.write(msg + '\n')

    def __del__(self):
        self.file.write('#' * 100 + '\n')
        self.file.close()


def my_tp(y_true, y_pred):
    return K.sum(K.cast(K.equal(1.0, K.round(y_pred), dtype='float16')))


def my_pn(y_true, y_pred):
    return K.sum(K.cast(K.equal(1.0, y_true), dtype='float16'))


def my_tpr(y_true, y_pred):
    return my_tp(y_true, y_pred) / my_pn(y_true, y_pred)


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def plot_line(plot_set, plot_id):
    filename = './output/log_cec%s_%d' % (plot_set, plot_id + 1)
    with open(filename, 'r') as f:
        lines = f.readlines()
    df = pd.DataFrame(columns=['epoch', 'best', 'optimal', 'time'])
    for line in lines:
        sp = line.split(' ')
        if len(sp) == 12:
            epoch_num = int(sp[1])
            best_v = float(sp[7].split('=')[1])
            time_v = float(sp[8].split('=')[1][:-1])
            df = df.append({'epoch': epoch_num, 'best': best_v, 'time': time_v, 'optimal': -1400 + 100 * plot_id},
                           ignore_index=True)
    if len(df) <= 0:
        return
    df.index = df['epoch']
    df['fixed'] = np.log10(df['best'] - df['optimal'] + 100)
    plt.figure()
    ax = df['fixed'].plot(kind='line')
    fig = ax.get_figure()
    fig.savefig(filename + '_line.png')
    # print('finish plot line of %s' % filename)
    return df['best'].iloc[-1]

# plot_line()
