import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch

import os

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _parse_data(data, device):
    # Extract features and targets from data dictionary and move to specified device
    features = data["sample_features"].to(device)
    targets = data["sample_targets"].to(device)
    return features, targets

def _parse_params(params):
    # Extract training parameters from params dictionary
    layer_inx = params["layer_inx"]
    lr = params["lr"]
    n_step = params["n_step"]
    momentum = params["momentum"]
    wd = params["wd"]


    return layer_inx, lr, n_step, momentum, wd


class Truncated_DNN(nn.Module):
    def __init__(self, model, layer_inx):
        super(Truncated_DNN, self).__init__()
        self.model = model
        self.layer_inx = layer_inx

    # Execute modules starting from the specified layer index
    def forward(self, x):
        layer = 0
        for module in self.model.children():
            if layer > self.layer_inx:
                x = module(x)
            layer += 1
        return x

features_out_hook = []
def hook(module, fea_in, fea_out):
    # Hook function to capture and store feature outputs during forward pass

    features_out_hook.append(fea_out.clone().detach())


class LayerwisePrediction_DNN(nn.Module):
    def __init__(self, model, input_size, class_size, device="cuda"):
        '''
        :param input_size: 隐藏层特征的维度
        :param class_size: 样本类别的维度
        :param DNN: 被解释的DNN模型
        '''
        super(LayerwisePrediction_DNN, self).__init__()

        # Initialize with the model to be interpreted, requiring model predictions and structure
        self.model = model
        self.device = device
        # Extractor network: maps input features to class predictions (feature -> 200 -> 100 -> class_size)
        self.extractor = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.Linear(200, 100),
            nn.Linear(100, class_size)
        )
        # Reconstructor network: maps predictions back to input feature space (class_size -> 100 -> 200 -> input_size)
        self.reconstructor = nn.Sequential(
            nn.Linear(class_size, 100),
            nn.Linear(100, 200),
            nn.Linear(200, input_size)
        )

    def forward(self, x):
        prediction = self.extractor(x)
        x_ = self.reconstructor(prediction)
        return prediction, x_

    def fit(self, data, params):

        features, targets = _parse_data(data, self.device)
        layer_inx, lr, n_step, momentum, wd = _parse_params(params)
        features_out_hook.clear()
        # Register hook to capture intermediate layer features at specified layer
        for name, module in self.model.named_children():
            if name == f"layer{layer_inx + 1}":
                handle = module.register_forward_hook(hook=hook)
                y = self.model(features.view(-1, 784))
                handle.remove()
                break
        layer_features = features_out_hook[0]


        # Initialize loss functions and optimizers
        loss_fun = nn.CrossEntropyLoss()
        loss_kl = nn.KLDivLoss()
        BCE_loss = nn.BCELoss()

        self.extractor.to(self.device)
        self.reconstructor.to(self.device)
        extractor_optimizer = optim.SGD(self.extractor.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        reconstructor_optimizer = optim.SGD(self.reconstructor.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        pbar = tqdm(range(n_step), mininterval=1, ncols=100)
        loss_dict = {"extractor_loss": [], "reconstructor_loss": []}

        for j in pbar:
            prediction, x_ = self.forward(layer_features.detach())
            truncted_model = Truncated_DNN(self.model, layer_inx)
            origin_result = F.softmax(truncted_model(layer_features), dim=1)
            con_result = F.log_softmax(truncted_model(x_), dim=1)

            # Compute reconstructor loss: KL divergence for prediction consistency + BCE for feature reconstruction
            # Note: For KLDivLoss, first parameter is predicted log-probabilities, second is target probabilities

            loss_reconstructor = loss_kl(con_result, origin_result) + BCE_loss(F.sigmoid(x_), F.sigmoid(layer_features))
            reconstructor_optimizer.zero_grad()
            extractor_optimizer.zero_grad()
            loss_reconstructor.backward(retain_graph=True)
            reconstructor_optimizer.step()
            extractor_optimizer.step()
            # Compute extractor loss: KL divergence for prediction consistency + cross-entropy for classification

            prediction_, x_ = self.forward(layer_features.detach())
            loss_extractor = loss_kl(F.log_softmax(prediction.detach(), dim=1), F.softmax(prediction_, dim=1)) + loss_fun(prediction_, targets.long())
            extractor_optimizer.zero_grad()
            loss_extractor.backward()
            extractor_optimizer.step()
            # Record losses for monitoring training progress
            loss_dict["extractor_loss"].append(loss_extractor.item())
            loss_dict["reconstructor_loss"].append(loss_reconstructor.item())
            pbar.set_postfix_str("loss={:0.6f}".format(loss_extractor.item()+loss_reconstructor.item()))

