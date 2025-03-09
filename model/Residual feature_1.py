import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
import math
import warnings
import scipy.io as io

from utils.return_dataset_res import return_dataset_res

warnings.filterwarnings("ignore")

train_data_normal, test_data_normal, fault_data_fault = return_dataset_res()

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=700, help="size of the batches")
parser.add_argument("--test_batch_size", type=int, default=1600, help="size of the batches")
parser.add_argument("--d_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_classes", type=int, default=3, help="number of classes for dataset")
parser.add_argument("--variable_num", type=int, default=40, help="train data second dimension")
parser.add_argument('--manualSeed', type=int, default=36, help='manual seed')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

dataset = torch.load("G:/论文/代码/零样本/data工况/工况1故障状态训练样本700个(带标签)).npy")
dataset_normal = torch.load("G:/论文/代码/零样本/data工况/工况1正常状态训练样本100个(带标签)).npy")
dataset_test = torch.load("G:/论文/代码/零样本/data工况/工况1故障状态测试样本210个(带标签)).npy")
dataset_test_normal = torch.load("G:/论文/代码/零样本/data工况/工况1正常状态测试样本30个(带标签)).npy")

MyFAGAN_normal_data_train = torch.utils.data.DataLoader(dataset_normal, batch_size=100, shuffle=True, num_workers=0)
MyFAGAN_data_train = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
fault_test = torch.utils.data.DataLoader(dataset_test, batch_size=210, shuffle=False, num_workers=0)
normal_test = torch.utils.data.DataLoader(dataset_test_normal, batch_size=30, shuffle=False, num_workers=0)

class Domain_Generator(nn.Module):
    def __init__(self):
        super(Domain_Generator, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('e_conv1', nn.Linear(64, 128))
        self.encoder.add_module('e_relu1', nn.ReLU(True))
        self.encoder.add_module('e_conv2', nn.Linear(128, 256))
        self.encoder.add_module('e_relu2', nn.ReLU(True))
        self.encoder.add_module('e_conv3', nn.Linear(256, 128))
        self.encoder.add_module('e_relu3', nn.ReLU(True))
        self.encoder.add_module('e_conv4', nn.Linear(128, 64))
        self.encoder.add_module('e_relu4', nn.Sigmoid())

    def forward(self, input_data):
        output = self.encoder(input_data)
        return output

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(0.5)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

generator = Domain_Generator().cuda()
optimizer = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
loss_MSE = torch.nn.MSELoss().cuda()
generator.apply(weights_init)

for epoch in range(opt.n_epochs):
    for i, (inputs, labels) in enumerate(MyFAGAN_normal_data_train, 0):
        print(labels)
        optimizer.zero_grad()
        input_sample = torch.FloatTensor(opt.batch_size, opt.variable_num)
        class_label = torch.LongTensor(opt.batch_size)

        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            input_sample = input_sample.cuda()
            class_label = class_label.cuda()

        input_sample.resize_as_(inputs).copy_(inputs)
        class_label.resize_as_(labels).copy_(labels)

        out = generator(input_sample)
        loss = loss_MSE(out, input_sample)

        loss.backward()
        optimizer.step()

        print(loss)

#故障状态训练样本
#####################
for i, (samples, label) in enumerate(MyFAGAN_data_train):   #利用训练样本正式训练模型
    label = label[:, 0]
    input_sample = torch.FloatTensor(opt.batch_size, opt.variable_num)
    class_label = torch.LongTensor(opt.batch_size)
    if cuda:
        samples = samples.cuda()
        label = label.cuda()
        input_sample = input_sample.cuda()
        class_label = class_label.cuda()

    input_sample.resize_as_(samples).copy_(samples)
    class_label.resize_as_(label).copy_(label)

    out = generator(input_sample)

# torch.save(generator.state_dict(), "G:/论文/代码/零样本/残差特征网络参数.pkl")
residual = input_sample - out
class_label = class_label.view(-1, 1)
out1 = torch.cat((class_label.float(), residual), 1)
out1 = np.array(out1.cpu().detach())
# print(out1.shape)
# io.savemat('G:/论文/代码/data/1027/残差特征(400个).mat', {'result': out1})
train_data = torch.tensor(residual)
train_label = torch.tensor(class_label)
train_data = torch.cat((inputs[0:100,:], train_data), 0)
train_label = torch.cat((labels[0:100, 1], train_label), 0)
print(train_data.shape)
print(train_label)

#计算属性
# att = torch.zeros(train_data.shape[0], train_data.shape[1]).cuda()
# att_ = torch.zeros(train_data.shape[0], train_data.shape[1]).cuda()
# for i in range(train_data.shape[0]):
#     for j in range(train_data.shape[1]):
#         if train_data[i, j] > 0:
#             att[i, j] = 1
#         else:
#             att[i, j] = 0
#
# for i in range(train_data.shape[0]):
#     for j in range(train_data.shape[1]):
#         if train_data[i, j] >= 0.5:
#             att_[i, j] = 1
#         elif train_data[i, j] <= -0.5:
#             att_[i, j] = 0
#         else:
#             att_[i, j] = 0.5
#
# t_SNE(train_data.detach().cpu(), train_label.detach().cpu())
#
# train_label = torch.cat((train_label.float(), att), 1)
# train_label = torch.cat((train_label, att_), 1)

# train_data_all = torch.cat((train_data, train_label), 1)
train_data_all = np.array(train_label.detach().cpu())
io.savemat('G:/论文/代码/零样本/data工况/测试的故障状态残差特征工况1（800个）.mat', {'train_gk1': train_data_all})

train_sample = TensorDataset(train_data, train_label)
torch.save(train_sample, "G:/论文/代码/零样本/data工况/工况1残差训练（800个）.npy")
#####################

#故障状态测试样本
#####################
for i, (samples, labels) in enumerate(fault_test):   #利用训练样本正式训练模型
    input_sample = torch.FloatTensor(opt.batch_size, opt.variable_num)
    class_label = torch.LongTensor(opt.batch_size)
    if cuda:
        samples = samples.cuda()
        labels = labels.cuda()
        input_sample = input_sample.cuda()
        class_label = class_label.cuda()

    input_sample.resize_as_(samples).copy_(samples)
    class_label.resize_as_(labels).copy_(labels)

    out_test = generator(input_sample)

residual_test = input_sample - out_test
test_data = torch.tensor(residual_test)
test_label = torch.tensor(class_label)
#####################

#正常状态测试样本
#####################
for i, (samples, labels) in enumerate(normal_test):
    input_sample = torch.FloatTensor(opt.batch_size, opt.variable_num)
    class_label = torch.LongTensor(opt.batch_size)
    if cuda:
        samples = samples.cuda()
        labels = labels.cuda()
        input_sample = input_sample.cuda()
        class_label = class_label.cuda()

    input_sample.resize_as_(samples).copy_(samples)
    class_label.resize_as_(labels).copy_(labels)

class_label = class_label.view(-1)
# print(class_label.shape)
test_all = torch.cat((input_sample, test_data), 0)
test_lab_all = torch.cat((class_label, test_label.view(-1)), 0)
test_all_ = torch.cat((test_lab_all.view(-1,1).float(), test_all), 1)
test_all_np = np.array(test_all_.cpu())
io.savemat('G:/论文/代码/零样本/data工况/测试的残差特征工况1（240个）.mat', {'result_gk1': test_all_np})
# print(test_all.shape)
# print(test_lab_all)
# t_SNE(test_all.detach().cpu(), test_lab_all.detach().cpu())
test_sample = TensorDataset(test_all, test_lab_all)
torch.save(test_sample, "G:/论文/代码/零样本/data工况/工况1残差测试（240个）.npy")