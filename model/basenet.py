import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Optional, Tuple
import torch
from model.VIT import VisionTransformer
#
# class GradReverse(Function):
#     def __init__(self, lambd):
#         self.lambd = lambd
#
#     def forward(self, x):
#         return x.view_as(x)
#
#     def backward(self, grad_output):
#         return (grad_output * -self.lambd)
#
#
# def grad_reverse(x, lambd=1.0):
#     return GradReverse(lambd)(x)



#
# class AlexNetBase(nn.Module):
#     def __init__(self, pret=True):
#         super(AlexNetBase, self).__init__()
#         model_alexnet = models.alexnet(pretrained=pret)
#         self.features = nn.Sequential(*list(model_alexnet.
#                                             features._modules.values())[:])
#         self.classifier = nn.Sequential()
#         for i in range(6):
#             self.classifier.add_module("classifier" + str(i),
#                                        model_alexnet.classifier[i])
#         self.__in_features = model_alexnet.classifier[6].in_features
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.classifier(x)
#         return x
#
#     def output_num(self):
#         return self.__in_features
#
#
# class VGGBase(nn.Module):
#     def __init__(self, pret=True, no_pool=False):
#         super(VGGBase, self).__init__()
#         vgg16 = models.vgg16(pretrained=pret)
#         self.classifier = nn.Sequential(*list(vgg16.classifier.
#                                               _modules.values())[:-1])
#         self.features = nn.Sequential(*list(vgg16.features.
#                                             _modules.values())[:])
#         self.s = nn.Parameter(torch.FloatTensor([10]))
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 7 * 7 * 512)
#         x = self.classifier(x)
#         return x


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


def grad_reverse(x, coeff=1.0):
    return GradReverse.apply(x, coeff)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output

class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out


class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.0015):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 64)
        self.fc2 = nn.Linear(64, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) \
                 / self.temp
        return x_out

class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=8, input_size=64, temp=0.05):
        super(ResClassifier_MME, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if return_feat:
            return x
        x = F.normalize(x)
        x = self.fc(x) / self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self, m):
        m.weight.data.normal_(0.0, 0.1)


class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 1)

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x_out = F.sigmoid(self.fc3_1(x))
        return x_out

        # self.Vit = VisionTransformer(img_size=64,
        #                   patch_size=4,
        #                   embed_dim=N * S,
        #                   depth=1,
        #                   num_heads=4,
        #                   representation_size=None,
        #                   num_classes=8)



# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#
#         self.f_conv1 = nn.Conv1d(1, 16, 4, 2, 1, bias=False)  # (1, 64) -> (16, 32)
#         self.f_relu1 = nn.LeakyReLU(True)
#         self.f_conv2 = nn.Conv1d(16, 32, 4, 2, 1, bias=False)  # (16, 32) -> (32, 16)
#         self.f_bn2 = nn.BatchNorm1d(32)
#         self.f_relu2 = nn.LeakyReLU(True)
#         self.f_pool1 = nn.MaxPool1d(2)  # (32, 16) -> (32, 8)
#         self.f_relu3 = nn.LeakyReLU(True)
#         self.f_conv3 = nn.Conv1d(32, 32, 4, 2, 1, bias=False)  # (32, 8) -> (32, 4)
#         self.f_bn3 = nn.BatchNorm1d(32)
#         self.f_relu4 = nn.LeakyReLU(True)
#         self.f_conv4 = nn.Conv1d(32, 32, 4, 2, 0, bias=False)  # (32, 4) -> (32, 1)
#         self.f_bn4 = nn.BatchNorm1d(32)
#         self.f_relu5 = nn.LeakyReLU(True)
#         self.flatten = nn.Flatten()
#         # No fully connected layer needed
#
#     def forward(self, x):
#         x = self.f_conv1(x)
#         x = self.f_relu1(x)
#         x = self.f_conv2(x)
#         x = self.f_bn2(x)
#         x = self.f_relu2(x)
#         x = self.f_pool1(x)
#         x = self.f_relu3(x)
#         x = self.f_conv3(x)
#         x = self.f_bn3(x)
#         x = self.f_relu4(x)
#         x = self.f_conv4(x)
#         x = self.f_bn4(x)
#         x = self.f_relu5(x)
#         x = self.flatten(x)
#         return x




class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        S = 16
        N = 4

        self.f_conv1 = nn.Conv1d(1, 16, 4, 2, 1, bias=False)
        self.f_relu1 = nn.LeakyReLU(True)
        self.f_conv2 = nn.Conv1d(16, 32, 4, 2, 1, bias=False)
        self.f_bn2 = nn.BatchNorm1d(32)
        self.f_relu2 = nn.LeakyReLU(True)
        self.f_pool1 = nn.MaxPool1d(2)
        self.f_relu3 = nn.LeakyReLU(True)
        self.f_conv3 = nn.Conv1d(32, 32, 4, 2, 1, bias=False)
        self.f_bn3 = nn.BatchNorm1d(32)
        self.f_relu4 = nn.LeakyReLU(True)
        self.f_conv4 = nn.Conv1d(32, 32, 4, 2, 1, bias=False)
        self.f_bn4 = nn.BatchNorm1d(32)
        self.f_relu5 = nn.LeakyReLU(True)
        self.f_pool2 = nn.MaxPool1d(2)
        self.f_relu6 = nn.LeakyReLU(True)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(32, 64)

    def forward(self, x):
        # print('shape of input is {}'.format(x.shape))
        # x = self.Vit(x)
        x = self.f_conv1(x)
        x = self.f_relu1(x)
        x = self.f_conv2(x)
        x = self.f_bn2(x)
        x = self.f_relu2(x)
        x = self.f_pool1(x)
        x = self.f_relu3(x)
        x = self.f_conv3(x)
        x = self.f_bn3(x)
        x = self.f_relu4(x)
        x = self.f_conv4(x)
        x = self.f_bn4(x)
        x = self.f_relu6(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)

        # x = self.fc(x)
        return x