
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from models.Net_Models.Net_Models import DNN_MNIST, DNN_FashionMNIST, DNN_EMNIST, ResNet50, Vgg16, ResNet18
import argparse
import os
import os.path as osp
from Utils import data
from LPI_model.layer_wise_prediction_DNN import LayerwisePrediction_DNN

import Utils.model as umodel

from torch.utils.data import random_split
from torch.utils.data import DataLoader
import math
from torch import nn, optim
from tqdm import tqdm
import matplotlib
from Utils.data import load_dataset
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use('TkAgg')



def entropy(probability_dis):
    result=-1
    if(len(probability_dis)>0):
        result=0

    for x in probability_dis:
        if x == 0:
            continue
        else:
            result+=(-x)*math.log(x,2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LPI args")
    parser.add_argument("--dataset", default="MNIST", type=str)
    parser.add_argument("--model", default="DNN_MNIST.pkl", type=str)
    parser.add_argument("--model_path", default="../models/black_box", type=str)
    # parser.add_argument("--layer_inx", default=0, type=int)
    parser.add_argument("--latent-dim", type=int, default=300)
    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_step", type=int, default=100)
    parser.add_argument("--sample_inx", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()


    #Image
    train_data,test_data= load_dataset(args.dataset)


    kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=100)
    plt.rc('font', family='Times New Roman')
    plt.rcParams.update({"font.size": 20})

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plot_data = []
    markers = ['o', 'v', 's', 'p', '*', 'D']

    #------------------------------------DNN------------------------------------------
    model = torch.load(osp.join(args.model_path, args.model)).to(device)
    model.eval()

    train_X, train_y, test_X, test_y = data.load_dataset(args.dataset)
#MNIST 60000 EMNIST 124800
    rand_list = np.random.choice(train_X.shape[0], 60000, replace=False)
    background_X = train_X[rand_list]
    background_y = train_y[rand_list]
    # epochs = [10, 30, 60]
    epochs = [100]
    dataset_size = [10000, 20000, 40000, 60000]
    for layer_inx in range(6):
        lpi = LayerwisePrediction_DNN(model, args.latent_dim, args.class_num)
        lpi.fit(
            data={"sample_features": background_X, "sample_targets": background_y},
            params={"layer_inx": layer_inx, "lr": args.lr, "n_step": args.n_step, "momentum": 0.9, "wd": 5e-4, },
        )
        torch.save(lpi,"../models/LPI_model/LPI_DNN_MNIST_layer{}_lzt.pkl".format(layer_inx+1))


    for epoch in epochs:
        for split_size in dataset_size:
            model = DNN_MNIST(28*28, 10).to(device)

            split_data, _ = random_split(train_data, [split_size, len(train_data)-split_size])
            umodel.DNN_train(model, split_data, epoch, args.batch_size, args.lr, device)
            acc = umodel.DNN_test(model, test_data, args.batch_size, device)
            x = plt.title('accuracy:' + str(round(float(acc * 100), 2)) + '%', fontsize=28)
            #hidden behavior
            layer_outs = model(background_X.view(-1, 28 * 28).to(torch.float32).to(device))
            for layer_inx in range(6):
                lpi = LayerwisePrediction_DNN(model, args.latent_dim, args.class_num)
                lpi.fit(
                    data={"sample_features": background_X, "sample_targets": background_y},
                    params={"layer_inx": layer_inx, "lr": args.lr, "n_step": args.n_step, "momentum": 0.9, "wd": 5e-4, },
                )
                layer_distribution = F.softmax(lpi(layer_outs[layer_inx].detach())[0])
                layer_entropy = []
                #
                for dis in layer_distribution:
                    layer_entropy.append(np.asscalar(entropy(dis).cpu().detach().numpy()))
                plot_data.append(layer_entropy)
                num, bins_limit, patches = plt.hist(layer_entropy, **kwargs, label="layer{}".format(layer_inx+1))

            plt.savefig(r"".format(split_size, epoch))
            plt.close()