import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch

import argparse
import os
import os.path as osp
from Utils import data
from LPI_model.layer_wise_prediction_DNN import LayerwisePrediction_DNN
import matplotlib
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split
from torch import nn
import pandas as pd


def plot_probs_of_layers(img, x, xtricks, sample_target, path=None):
    #
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 设置图框线粗细
    bwith = 2  # 边框宽度设置为2

    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(55, 3))
    if len(x.shape) == 1:
        plt.xlabel("classes")
        plt.ylabel("probability")
        plt.bar(xtricks, height=x, width=0.5)
        plt.ylim([0, 0.5])
    else:
        plt.subplot(1, x.shape[0]+1, 1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        for i in range(1, x.shape[0]+1):
            # 创建小图
            plt.subplot(1, x.shape[0]+1, i + 1)
            # plt.xlabel("classes", fontsize=40)
            if i == 1:
                plt.ylabel("probability", fontsize=35)
                plt.yticks([0, 0.5, 1])
            else:
                plt.yticks([])
            TK = plt.gca()
            TK.spines['bottom'].set_linewidth(bwith)
            TK.spines['left'].set_linewidth(bwith)
            TK.spines['top'].set_linewidth(bwith)
            TK.spines['right'].set_linewidth(bwith)
            plt.ylim([0, 1])

            plt.bar(xtricks, height=x[i-1], width=0.5, color='dodgerblue')

            plt.bar(sample_target, height=x[i-1][sample_target], width=0.5, color='orange')

            plt.tick_params(labelsize=40)

            if i == x.shape[0]:
                plt.legend(["out".format(i)], loc=1, prop={'size':30})
            else:
                plt.legend(["layer{0}".format(i)], loc=1, prop={'size':30})
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
    if path != None:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def get_imagenet(root, train = True, transform = None, target_transform = None):
    if train:
        root = os.path.join(root, 'train')
    else:
        root = os.path.join(root, 'val')
    return datasets.ImageFolder(root = root,
                               transform = transform,
                               target_transform = target_transform)
def load_dataset(dataname):
    if dataname == "MNIST":
        data_tf = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(
            root='',
            train=True,
            transform=data_tf,
            download=True
        )

        test_data = datasets.MNIST(
            root='',
            train=False,
            transform=data_tf,
            download=True
        )

        return train_data, test_data

    if dataname == "FashionMNIST":
        data_tf = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.FashionMNIST(
            root='',
            train=True,
            transform=data_tf,
            download=True
        )
        test_data = datasets.FashionMNIST(
            root='',
            train=False,
            transform=data_tf,
            download=True
        )
        return train_data, test_data

    if dataname == "EMNIST":
        data_tf = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.EMNIST(
            root='',
            train=True,
            transform=data_tf,
            download=True,
            split='letters'
        )
        test_data = datasets.EMNIST(
            root='',
            train=False,
            transform=data_tf,
            download=True,
            split='letters'
        )
        #EMNIST的原始label是[1,26]，将其转为 [0,25]
        train_data.targets = train_data.targets - 1
        test_data.targets = test_data.targets - 1
        return train_data, test_data

    if dataname == "CIFAR-10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = datasets.CIFAR10(root='', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='', train=False, download=True, transform=transform)

        return train_data, test_data

    if dataname == "CIFAR-100":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = datasets.CIFAR100(root='', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root='', train=False, download=True, transform=transform)

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

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="LPI args")
    parser.add_argument("--dataset", default="MNIST", type=str)
    parser.add_argument("--model", default="DNN_MNIST", type=str)
    parser.add_argument("--model_path", default="../models/black_box", type=str)
    # parser.add_argument("--layer_inx", default=0, type=int)
    parser.add_argument("--latent-dim", type=int, default=300)
    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-step", type=int, default=500)
    parser.add_argument("--sample_inx", type=int, default=1)
    args = parser.parse_args()

    #加载模型
    model = torch.load(r"../models/black_box/{0}.pkl".format(args.model)).to(device)

    #load data
    train_data, test_data = load_dataset(args.dataset)





    background_data, _ = random_split(train_data, [30000, len(train_data) - 30000])
    dataloader = DataLoader(background_data, batch_size=128, shuffle=True, num_workers=4)
    # test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)

    #------------------DNN----------------------------------------------

    train_X, train_y, test_X, test_y = data.load_dataset(args.dataset)
    #使用部分数据训练LPI
    rand_list = np.random.choice(train_X.shape[0], 30000, replace=False)
    background_X = train_X[rand_list]
    background_y = train_y[rand_list]

    explain_sample = test_X[args.sample_inx] 
    sample_target = test_y[args.sample_inx]

    sample_layer_out = model(explain_sample.to(torch.float32).view(-1, 28 * 28).to(device)) #DNN维度是28*28


    layer_probs = []
    for layer_inx in range(6):
        lpi = LayerwisePrediction_DNN(model, args.latent_dim, args.class_num)
        lpi.fit(
            data={"sample_features": background_X, "sample_targets": background_y},
            params={"layer_inx":layer_inx, "lr": args.lr, "n_step": args.n_step, "momentum": 0.9, "wd": 5e-4, },
        )
        torch.save(lpi, "../models/LPI_model/LPI_{0}_layer{1}.pkl".format(args.model, layer_inx+1))
        prediction, _ = lpi(sample_layer_out[layer_inx].detach())
        layer_probs.append(F.softmax(prediction).squeeze().cpu().detach().numpy())

    layer_probs.append(F.softmax(sample_layer_out[-1]).squeeze().cpu().detach().numpy())

