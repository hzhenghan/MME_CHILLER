from __future__ import print_function
import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import time
# from FLOPS import FL
from model.LinearAverage import LinearAverage
from utils.loss import entropy, adentropy, entropy_c, adentropy_c
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from torch.utils.data import DataLoader
from model.basenet import FeatureExtractor
from model.basenet import Predictor_deep, ResClassifier_MME
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn
import matplotlib.pyplot as plt
from t_SNE import visualizePerformance


# from thop import profile, clever_format
# source_loader, target_loader, target_loader_unl, target_loader_val, \
#     target_loader_test, class_list = return_dataset()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Training settings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=15000, metavar='N',
                    help='maximum number of iterations ''to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME', choices=['ST', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--T', type=float, default=0.05, metavar='T', help='temperature (default: 0.075)')
parser.add_argument('--lamda', type=float, default=0.15, metavar='LAM', help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False, help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda', help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging ''training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34')
parser.add_argument('--source', type=int, default=40, help='source domain')
parser.add_argument('--target', type=int, default=50, help='target domain')
parser.add_argument('--num', type=int, default=3, help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment ''before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=False, help='early stopping on validation or not')
parser.add_argument('--batch_size', default=4, help='batch size')
parser.add_argument('--UDA', default=False, help='Whether use labeled target data')
parser.add_argument('--dataset', default='multi', help='Whether use labeled target data')
parser.add_argument('--enc', type=bool, default=True, help='Whether use labeled target data')
args = parser.parse_args()
print(args)

source_dataset, target_dataset, target_dataset_val, target_dataset_unl, target_dataset_test = return_dataset(args.source, args.target,dataset='A')

source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                           drop_last=True)
target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                           drop_last=True)
target_loader_val = DataLoader(target_dataset_val,
                               batch_size=args.batch_size,
                               shuffle=True, drop_last=True)
target_loader_unl = DataLoader(target_dataset_unl,
                               batch_size=args.batch_size,
                               shuffle=True)
target_loader_test = DataLoader(target_dataset_test,
                                batch_size=args.batch_size,
                                shuffle=True, drop_last=True)

s_v_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True,
                            drop_last=True)

t_v_dataloader = DataLoader(target_dataset_unl,
                            batch_size=32,
                            shuffle=True, drop_last=True)

# use_gpu = torch.cuda.is_available()

record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s' %
                           (args.method, args.net, args.source,
                            args.target, args.num))

torch.manual_seed(args.seed)

ndata = target_loader_unl.__len__() * args.batch_size

lemniscate = LinearAverage(64, ndata, args.T, 0.0)

G = FeatureExtractor()
C1 = ResClassifier_MME(num_classes=8, input_size=64, temp=args.T)

lr = args.lr

G
C1
# fl_g = FL(G)
# fl_c = FL(C1)
# fl_m = FL(lemniscate)


# G = nn.DataParallel(G)
# C1 = nn.DataParallel(C1)

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)
index_t = torch.LongTensor(1)

im_data_s = im_data_s
im_data_t = im_data_t
im_data_tu = im_data_tu
gt_labels_s = gt_labels_s
gt_labels_t = gt_labels_t
sample_labels_t = sample_labels_t
sample_labels_s = sample_labels_s
index_t = index_t

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)
index_t = Variable(index_t)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def train():
    G.train()
    C1.train()
    criterion = nn.CrossEntropyLoss()
    optimizer_g = optim.SGD(list(G.parameters()), momentum=0.9, lr=0.01,
                            weight_decay=0.0005, nesterov=True)
    # optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
    #                         weight_decay=0.0005, nesterov=True)
    optimizer_c1 = optim.SGD(list(C1.parameters()), lr=1.0, momentum=0.9,
                             weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        # optimizer_f.zero_grad()
        optimizer_c1.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])

    # param_lr_f = []
    # for param_group in optimizer_f.param_groups:
    #     param_lr_f.append(param_group["lr"])

    param_lr_c1 = []
    for param_group in optimizer_c1.param_groups:
        param_lr_c1.append(param_group["lr"])

    all_step = args.steps

    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)

    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)

    best_acc = 0
    counter = 0
    fea_flag = 0
    df_result_loss = {'step': [], 'loss_S': [], 'loss_T': []}
    df_result_acc = {'step': [], 'val': [], 'test': []}

    for step in range(all_step):

        inv_lr_scheduler(param_lr_g, optimizer_g, step,
                         init_lr=args.lr)
        # optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
        #                                init_lr=args.lr)
        inv_lr_scheduler(param_lr_c1, optimizer_c1, step,
                         init_lr=args.lr)
        lr = optimizer_c1.param_groups[0]['lr']

        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)

        data_s = next(data_iter_s)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)

        im_data_s.resize_(data_s[0].size()).copy_(data_s[0])  # source_data
        gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])  # source_data_label
        im_data_t.resize_(data_t[0].size()).copy_(data_t[0])  # target_data
        gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])  # target_data_label
        im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])  # unlabeled_data
        # index_t.resize_(data_t_unl[2].size()).copy_(data_t_unl[2])
        index_t = data_t_unl[2]
        index_t = Variable(index_t)

        zero_grad_all()
        # C1.module.weight_norm()
        # C1.module.weight_norm()
        if args.UDA:
            data = im_data_s
            target = gt_labels_s
        else:
            data = torch.cat((im_data_s, im_data_t), 0)
            target = torch.cat((gt_labels_s, gt_labels_t), 0)

        if len(data) < args.batch_size:
            break
        if len(target) < args.batch_size:
            break

        output = G(data.unsqueeze(1))

        # flops, params = fl_g.flops(data.unsqueeze(1))
        # print(f"FLOPs of G: {flops}")
        # print(f"Parameters of G: {params}")
        # total_memory = fl_g.calculate_memory_access(G, data.unsqueeze(1))
        # print(f"Total memory access of G: {total_memory} Bytes")

        out1 = C1(output)

        # flops, params = fl_c.flops(output)
        # print(f"FLOPs of C: {flops}")
        # print(f"Parameters of C: {params}")
        # total_memory = fl_c.calculate_memory_access(C1, output)
        # print(f"Total memory access of C: {total_memory} Bytes")

        loss_s = criterion(out1, target.long())
        # no parametric feature memory bank
        feat_t = G(im_data_tu.unsqueeze(1))
        out_t = C1(feat_t)
        feat_t = F.normalize(feat_t)


        feat_mat = lemniscate(feat_t, index_t)
        feat_mat[:, index_t] = -1 / args.T
        feat_mat2 = torch.matmul(feat_t, feat_t.t()) / args.T
        mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool()
        feat_mat2.masked_fill_(mask, -1 / args.T)

        loss_nc = args.lamda * entropy_c(torch.cat([out_t, feat_mat, feat_mat2], 1))
        # loss_nc = 0.05 * adentropy_c(torch.cat([out_t, feat_mat, feat_mat2], 1))

        loss = loss_nc + loss_s
        log_train = ('S {} T {} Train Ep: {} lr{} \t ' \
                    'Loss S: {:.6f}  \t' \
                    'Loss All: {:.6f} \t'\
                    'Loss T {:.6f} \t' \
                    'Method {}\n'.format(args.source, args.target,
                                         step, lr, loss_s.data, loss.data,
                                         loss_nc.data, args.method))
        loss.backward()
        optimizer_g.step()
        optimizer_c1.step()
        optimizer_g.zero_grad()
        optimizer_c1.zero_grad()
        # update  memory bank
        lemniscate.update_weight(feat_t, index_t)

        if step % args.log_interval == 0:
            print(log_train)

        if step % args.save_interval == 0 and step > 0:

            loss_val, acc_val,_ = test(target_loader_val)

            loss_test, acc_test, cm = test(target_loader_test)

            # plot_confusion_matrix(cm)

            plot_matrix(cm, flag=fea_flag, acc=acc_test, labels_name=['N', 'FWC', 'FWE', 'RL', 'RO', 'EO', 'CF', 'NC'])
            # print(f"Execution time: {execution_time} seconds")
            df_result_acc['step'].append(step)
            df_result_acc['val'].append(acc_val)
            df_result_acc['test'].append(acc_test)

            file_path = 'log_acc.csv'

            if os.path.exists(file_path):
                os.remove(file_path)

            # pd.DataFrame(df_result_acc).to_csv('log_acc.csv')
            #
            # visualizePerformance(G, C1, s_v_dataloader,
            #                      t_v_dataloader,
            #                      num_of_samples=3000, flag = fea_flag,
            #                      title='UDA of S {} T {} step {}'.format(args.source, args.target, step))
            fea_flag += 1

            G.train()
            C1.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    break
            print('best acc test %f best acc val %f' % (best_acc_test,
                                                        acc_val))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d best %f final %f \n' % (step, best_acc_test, acc_val))
            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))
                torch.save(C1.state_dict(),
                           os.path.join(args.checkpath,
                                        "F1_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))


def plot_confusion_matrix(conf_matrix):

    # 绘制混淆矩阵
    labels = [0, 1, 2, 3, 4, 5, 6, 7]  # 每种类别的标签

    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    row_sums = conf_matrix.sum(axis=1)
    accuracies = conf_matrix.diagonal() / row_sums

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(8):
        for y in range(8):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(8), labels)
    plt.xticks(range(8), labels)  # X轴字体倾斜45°
    plt.show()

def plot_matrix(cm, flag, acc, labels_name, title=None, thresh=0.8, axis_labels=None,tick_fontsize=15, text_fontsize=14):
# 利用sklearn中的函数生成混淆矩阵并归一化
#     cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    np.save('cm/{2}to{3}/cm_{0}_{1}'.format(flag, acc, args.source, args.target), cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.figure(figsize=(8,8), dpi=400)
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    # plt.colorbar()  # 绘制图例
    plt.colorbar().remove()
# 图像标题
    if title is not None:
        plt.title(title)
# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45, fontsize=tick_fontsize)
    plt.yticks(num_local, axis_labels, fontsize=tick_fontsize)
    plt.ylabel('True label', fontsize=tick_fontsize)
    plt.xlabel('Predicted label', fontsize=tick_fontsize)

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            # if int(cm[i][j] * 100 + 0.5) > 0:
            plt.text(j, i, format(cm[i][j], '.2f'),
                    fontsize=text_fontsize,
                    ha="center",
                    va="center",
                    color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行


    plt.savefig('cm/{2}to{3}/fig/fig_{0}_{1}.png'.format(flag, acc,args.source, args.target), dpi=400)
    plt.show()

def test(loader):
    G.eval()
    C1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = 8
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss()
    confusion_matrix = torch.zeros(num_class, num_class)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t.unsqueeze(1))
            output1 = C1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]

            all_preds.extend(pred1.cpu().numpy())
            all_labels.extend(gt_labels_t.cpu().numpy())

            # for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
            #     confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    cm = confusion_matrix_sklearn(all_labels, all_preds)




    # class_accuracy = class_TP / (class_TP + class_FN)
    # class_precision = class_TP / (class_TP + class_FP)
    # class_recall = class_TP / (class_TP + class    # flops_G, params_G = fl_g.flops(im_data_t.unsqueeze(1))
    # print(f"FLOPs of G: {flops_G}")
    # print(f"Parameters of G: {params_G}")
    # total_memory_G = fl_g.calculate_memory_access(G, im_data_t.unsqueeze(1))
    # print(f"Total memory access of G: {total_memory_G} Bytes")

    #     # flops_C, params_C = fl_c.flops(feat)
    #     # print(f"FLOPs of C: {flops_C}")
    #     # print(f"Parameters of C: {params_C}")
    #     # total_memory_C = fl_c.calculate_memory_access(C1, feat)
    #     # print(f"Total memory access of C: {total_memory_C} Bytes")_FN)
    # class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)

    # Print or use class-wise metrics as needed
    # for i in range(num_classes):
    #     print(f"Class {i}: Accuracy = {class_accuracy[i]:.2f}, Precision = {class_precision[i]:.2f}, Recall = {class_recall[i]:.2f}, F1-Score = {class_f1[i]:.2f}")

    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.6f})\n'.
          format(test_loss, correct, size,
                 correct / size))


    return test_loss.data, float(correct) / size, cm

if __name__ == '__main__':
    train()
