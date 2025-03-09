import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_matrix(cm, t, title=None, thresh=0.8, axis_labels=None,tick_fontsize=16, text_fontsize=16):
    labels_name = ['N', 'FWC', 'FWE', 'RL', 'RO', 'EO', 'CF', 'NC']
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8,8), dpi=400)
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar().remove()

    if title is not None:
        plt.title(title)

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45, fontsize=tick_fontsize)
    plt.yticks(num_local, axis_labels, fontsize=tick_fontsize)
    plt.ylabel('True label', fontsize=tick_fontsize)
    plt.xlabel('Predicted label', fontsize=tick_fontsize)


    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            # if int(cm[i][j] * 100 + 0.5) > 0:
            plt.text(j, i, format(cm[i][j], '.2f'),
                    fontsize=text_fontsize,
                    ha="center",
                    va="center",
                    color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    plt.savefig('cm/use_semi/fig_{0}.png'.format(t, dpi=600))
    plt.show()



RP1 = np.load('cm/use_semi/cm_8_0.8662917271407837.npy')
RP2 = np.load('cm/use_semi/cm_2_0.9386934673366835.npy')
RP3 = np.load('cm/use_semi/cm_24_0.9566582914572864.npy')
HY1 = np.load('cm/use_semi/cm_3_0.8917085427135678.npy')
HY2 = np.load('cm/use_semi/cm_21_0.9659547738693467.npy')
HY3 = np.load('cm/use_semi/cm_20_0.967462311557789.npy')
# t2 = np.load('cm/50to40/cm_7_0.9404522613065327.npy')

if __name__ == '__main__':

    # plot_matrix(t2,'RP')
    plot_matrix(RP1, 'RP_1')
    plot_matrix(RP2, 'RP_2')
    plot_matrix(RP3, 'RP_3')
    plot_matrix(HY1, 'HY_1')
    plot_matrix(HY2, 'HY_2')
    plot_matrix(HY3, 'HY_3')

