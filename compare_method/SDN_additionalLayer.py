import torch
import torch.nn as nn
import logging
import pdb
from My_Nets.Net_Models import DNN_MNIST, DNN_FashionMNIST, HLIE_Net
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from tqdm import trange
import matplotlib
from scipy.stats import wasserstein_distance
import joypy
from matplotlib.pyplot import MultipleLocator
import math
import Utils.model as umodel
import pandas as pd
from torch.utils.data import random_split
from scipy.spatial import ConvexHull
from Utils import data
import os
import argparse
from tqdm import tqdm
from scipy.stats import wasserstein_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_image(img, label, name):
    fig = plt.figure()
    plt.tight_layout()
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    # plt.title("{}:{}".format(name,label.item()))
    plt.show()


class AdditionalLayerTemplet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdditionalLayerTemplet, self).__init__()
        #transformer把隐空间维度改成512
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()
        linear_params = list(self.fc1.parameters())
        linear_params[0].data.normal_(0, 0.01)
        linear_params[0].data.fill_(0)
        linear_params_2 = list(self.fc2.parameters())
        linear_params_2[0].data.normal_(0, 0.01)
        linear_params_2[0].data.fill_(0)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # return self.sigmoid(x)
        return x


class InternalClassifier_DNN(nn.Module):
    # def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
    def __init__(self, input_size, num_classes, alpha=0.5):
        super(InternalClassifier_DNN, self).__init__()
        # red_kernel_size = -1 # to test the effects of the feature reduction
        # red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        # self.output_channels = output_channels

        # if red_kernel_size == -1:
        self.linear = nn.Linear(input_size, num_classes)
        self.forward = self.forward_wo_pooling
        # else:
        #     red_input_size = int(input_size/red_kernel_size)
        #     self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
        #     self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
        #     self.alpha = nn.Parameter(torch.rand(1))
        #     self.linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
        #     self.forward = self.forward_w_pooling

    # def forward_w_pooling(self, x):
    #     avgp = self.alpha*self.max_pool(x)
    #     maxp = (1 - self.alpha)*self.avg_pool(x)
    #     mixed = avgp + maxp
    #     return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        # return self.linear(x.view(x.size(0), -1))
        return self.linear(x)


# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size / 4)
    else:
        return -1


class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(InternalClassifier, self).__init__()
        # red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size)  # get the pooling size
        self.output_channels = output_channels
        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels * input_size * input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size / red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear = nn.Linear(output_channels * red_input_size * red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        avgp = self.alpha * self.max_pool(x)
        maxp = (1 - self.alpha) * self.avg_pool(x)
        mixed = avgp + maxp
        return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))
        return self.linear(x)


def load_dataset(dataname):
    if dataname == "MNIST":
        data_tf = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(
            root='G:/代码/李家旺实验/第二章算法程序/data/',
            train=True,
            transform=data_tf,
            download=True
        )

        test_data = datasets.MNIST(
            root='G:/代码/李家旺实验/第二章算法程序/data/',
            train=False,
            transform=data_tf,
            download=True
        )

        return train_data, test_data

    if dataname == "FashionMNIST":
        data_tf = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.FashionMNIST(
            root='G:/代码/李家旺实验/第二章算法程序/data/',
            train=True,
            transform=data_tf,
            download=True
        )
        test_data = datasets.FashionMNIST(
            root='G:/代码/李家旺实验/第二章算法程序/data/',
            train=False,
            transform=data_tf,
            download=True
        )
        return train_data, test_data

    if dataname == "EMNIST":
        data_tf = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.EMNIST(
            root='G:/代码/李家旺实验/第二章算法程序/data/',
            train=True,
            transform=data_tf,
            download=True,
            split='letters'
        )
        test_data = datasets.EMNIST(
            root='G:/代码/李家旺实验/第二章算法程序/data/',
            train=False,
            transform=data_tf,
            download=True,
            split='letters'
        )
        # EMNIST的原始label是[1,26]，将其转为 [0,25]
        train_data.targets = train_data.targets - 1
        test_data.targets = test_data.targets - 1
        return train_data, test_data

    if dataname == "CIFAR-10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = datasets.CIFAR10(root='G:/代码/李家旺实验/第二章算法程序/data/', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='G:/代码/李家旺实验/第二章算法程序/data/', train=False, download=True, transform=transform)

        return train_data, test_data

    if dataname == "CIFAR-100":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = datasets.CIFAR100(root='G:/代码/李家旺实验/第二章算法程序/data/', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root='G:/代码/李家旺实验/第二章算法程序/data/', train=False, download=True, transform=transform)

        return train_data, test_data

    if dataname == "ImageNet":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        root = './data/ImageNet'
        train_data = get_imagenet(root, train=False, transform=data_transforms['val'], target_transform=None)
        class_names = train_data.classes
        return train_data


features_out_hook = []


def hook(module, fea_in, fea_out):
    features_out_hook.append(fea_out.clone().detach())


def get_imagenet(root, train=True, transform=None, target_transform=None):
    if train:
        root = os.path.join(root, 'train')
    else:
        root = os.path.join(root, 'val')
    return datasets.ImageFolder(root=root,
                                transform=transform,
                                target_transform=target_transform)


def encircle(x, y, ax=None, **kw):
    if not ax: ax = plt.gca()
    p = np.c_[x, y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kw)
    ax.add_patch(poly)


def plot_probs_of_layers(img, x, xtricks, sample_target, path=None):
    # 设置matplotlib正常显示中文和负号
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 设置图框线粗细
    bwith = 2  # 边框宽度设置为2

    plt.rcParams['ytick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(55, 3))
    if len(x.shape) == 1:
        plt.xlabel("classes")
        plt.ylabel("probability")
        plt.bar(xtricks, height=x, width=0.5)
        plt.ylim([0, 0.5])
    else:
        plt.subplot(1, x.shape[0] + 1, 1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        for i in range(1, x.shape[0] + 1):
            # 创建小图
            plt.subplot(1, x.shape[0] + 1, i + 1)
            # plt.xlabel("classes", fontsize=40)
            if i == 1:
                plt.ylabel("probability", fontsize=35)
                plt.yticks([0, 0.5, 1])
            else:
                plt.yticks([])
            TK = plt.gca()  # 获取边框
            TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
            TK.spines['left'].set_linewidth(bwith)  # 图框左边
            TK.spines['top'].set_linewidth(bwith)  # 图框上边
            TK.spines['right'].set_linewidth(bwith)  # 图框右边
            plt.ylim([0, 1])
            # 备选颜色 mediumblue
            plt.bar(xtricks, height=x[i - 1], width=0.5, color='dodgerblue')
            # 把该样本类别的概率换位另一种颜色
            plt.bar(sample_target, height=x[i - 1][sample_target], width=0.5, color='orange')
            # plt.xticks(np.linspace(0, 20, 4)) #设置步长
            plt.tick_params(labelsize=40)

            if i == x.shape[0]:
                plt.legend(["out".format(i)], loc=1, prop={'size': 30})
            else:
                plt.legend(["layer{0}".format(i)], loc=1, prop={'size': 30})
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
    if path != None:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def IC_train(model, data_loader, layer_inx, epochs, nn_model=None):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    loss_fun = nn.CrossEntropyLoss()
    pbar = tqdm(range(epochs), mininterval=1, ncols=100)
    for j in pbar:
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.view(-1, 784)
            data, target = data.to(device), target.to(device)
            features_out_hook.clear()
            #CNN用这个
            for name, module in nn_model.named_children():
                if name == f"layer{layer_inx + 1}":
                    handle = module.register_forward_hook(hook=hook)
                    y = nn_model(data)
                    handle.remove()
                    break
            #transformer用这个
            # for name, module in nn_model.named_children():
            #     if name == "transformer":
            #         handle = module.layers[layer_inx][1].register_forward_hook(hook=hook)
            #         y = nn_model(data)
            #         handle.remove()
            #         break
            # vit class token机制，在0位置加入了永用于分类的token，因此只需要用第0个特征进行预测
            # layer_features = features_out_hook[0][:,0]

            layer_features = features_out_hook[0]

            classes = model(layer_features.detach())
            loss = loss_fun(classes, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_postfix_str("loss={:0.6f}".format(loss.item()))
    return model


def AdditionalLayerTemplet_train(model, data_loader, layer_inx, epochs, nn_model=None):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    loss_fun = nn.KLDivLoss(reduction="batchmean")
    pbar = tqdm(range(epochs), mininterval=1, ncols=100)
    for j in pbar:
        for batch_idx, (data, target) in enumerate(data_loader):
            data=data.view(-1, 784)
            data, target = data.to(device), target.to(device)
            features_out_hook.clear()
            #CNN用这个
            for name, module in nn_model.named_children():
                if name == f"layer{layer_inx + 1}":
                    handle = module.register_forward_hook(hook=hook)
                    y = nn_model(data)
                    handle.remove()
                    break
            layer_features = features_out_hook[0].view(features_out_hook[0].size(0), -1)
            # transformer用这个
            # for name, module in nn_model.named_children():
            #     if name == "transformer":
            #         handle = module.layers[layer_inx][1].register_forward_hook(hook=hook)
            #         y = nn_model(data)
            #         handle.remove()
            #         break
            # layer_features = features_out_hook[0][:,0].view(features_out_hook[0].size(0), -1)

            output = model(layer_features.detach())
            # output_normal = output / torch.sum(output, dim=1, keepdim=True)
            # y_normal = y / torch.sum(y, dim=1, keepdim=True)

            loss = loss_fun(F.log_softmax(output), F.softmax(y).detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_postfix_str("loss={:0.6f}".format(loss.item()))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LPI args")
    parser.add_argument("--dataset", default="MNIST", type=str)
    parser.add_argument("--model", default="DNN_MNIST", type=str)
    parser.add_argument("--model_path", default="../models/black_box", type=str)
    # parser.add_argument("--layer_inx", default=0, type=int)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-step", type=int, default=30)
    parser.add_argument("--sample_inx", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # 加载模型
    model = torch.load(r"../models/black_box/{0}.pkl".format(args.model)).to(device)
    print(model)
    # 加载数据集
    train_data, test_data = load_dataset(args.dataset)
    background_data, _ = random_split(train_data, [30000, len(train_data) - 30000])
    # train_data = load_dataset(args.dataset)
    dataloader = DataLoader(background_data, batch_size=128, shuffle=True, num_workers=2)
    # ------------------DNN----------------------------------------------
    # background_data, _ = random_split(train_data, [30000, len(train_data) - 30000])
    # dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    # model = torch.load(r"./models/black_box/{0}.pkl".format(args.model)).to(device)
    # test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # umodel.DNN_test(model,test_data, args.batch_size, device)

    # for name, module in model.named_children():
    #     handle = module.register_forward_hook(hook=hook)
    #     y = model(train_X[0].to(torch.float32).view(-1,784).to(device))
    #     handle.remove()
    # #
    # layer_outs = features_out_hook
    #DNN参数
    input_size = [300,300,300,300,300,300]
    # ResNet18的参数
    # input_size = [32, 16, 8, 4]
    # output_channels = [64, 128, 256, 512]

    # ResNet50的参数
    # output_channels = [256, 512, 1024, 2048]
    # input_size = [32, 16, 8, 4]

    # VGG16的参数
    # input_size = [16, 8, 4, 2, 1]
    # output_channels = [64, 128, 256, 512, 512]

    # ViT的参数
    # input_size = [1, 1, 1, 1, 1, 1]
    # output_channels = [1024, 1024, 1024, 1024, 1024, 1024]

    for layer_inx in trange(6):
        IC_model = InternalClassifier_DNN(input_size[layer_inx], args.class_num).to(device)
        IC_train(IC_model, data_loader=dataloader, layer_inx=layer_inx, epochs=10, nn_model=model)
        torch.save(IC_model, "../models/compare_model/IC_{0}_layer{1}_lzt.pkl".format(args.model, layer_inx + 1))
        # AdditionalLayer_model = AdditionalLayerTemplet(input_size[layer_inx], args.class_num).to(device)
        # AdditionalLayerTemplet_train(AdditionalLayer_model, data_loader=dataloader, layer_inx=layer_inx, epochs=10, nn_model=model)
        # torch.save(AdditionalLayer_model,
        #            "../models/compare_model/AdditionalLayer_{0}_layer{1}_lzt.pkl".format(args.model, layer_inx + 1))



    # IC, AdditionalLayer
    # compare_model = "IC"
    # for batch_idx, (sample, target) in enumerate(test_dataloader):
    #     layers_prediction = []
    #     sample = sample.view(-1, 784).to(device)
    #     for layer_inx in range(4):
    #         explain_model = torch.load("./models/compare_model/{0}_{1}_layer{2}.pkl".format(compare_model, args.model, layer_inx + 1))
    #         for name, module in model.named_children():
    #             if name == f"layer{layer_inx+1}":
    #                 handle = module.register_forward_hook(hook=hook)
    #                 features_out_hook.clear()
    #                 y = model(sample.to(device))
    #                 layer_features = features_out_hook[0].detach()
    #                 break
    #         handle.remove()
    #         layer_prediction = F.softmax(explain_model(layer_features))
    #         layers_prediction.append(layer_prediction.cpu().detach().numpy().squeeze())
    #     final_prediction = F.softmax(model(sample))
    #     layers_prediction.append(final_prediction.cpu().detach().numpy().squeeze())
    #
    #     _, pred = torch.max(final_prediction, 1)
    #
    #     sample_show = sample.cpu().numpy().reshape(28,28)
    #
    #     if pred.cpu() == target:
    #         plot_probs_of_layers(sample_show, np.array(layers_prediction), [str(c) for c in range(len(train_data.classes))], target,
    #                              path=r"C:\Users\Phenix\Desktop\小论文\小论文实验\新增实验\4-一致性分析\compaer_method\{0}\{1}\预测正确\{2}.png".format(
    #                                  compare_model, args.model, batch_idx))
    #     else:
    #         plot_probs_of_layers(sample_show, np.array(layers_prediction), [str(c) for c in range(len(train_data.classes))], target,
    #                              path=r"C:\Users\Phenix\Desktop\小论文\小论文实验\新增实验\4-一致性分析\compaer_method\{0}\{1}\预测错误\{2}.png".format(
    #                                  compare_model, args.model, batch_idx))



