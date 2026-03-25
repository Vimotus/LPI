import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch

import os


import math
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _parse_data(data, device):
    features = data["sample_features"].to(device)
    targets = data["sample_targets"].to(device)
    return features, targets

def _parse_params(params):
    layer_inx = params["layer_inx"]
    lr = params["lr"]
    n_step = params["n_step"]
    momentum = params["momentum"]
    wd = params["wd"]
    # loss_type = params["loss_type"]

    return layer_inx, lr, n_step, momentum, wd


class Truncated_VGG(nn.Module):
    def __init__(self, model, layer_inx):
        super(Truncated_VGG, self).__init__()
        self.model = model
        self.layer_inx = layer_inx
    def forward(self, x):
        #original model
        # for name, module in self.model.named_children():
        #     if name == "features":
        #         for i in range(len(module)):
        #             if i > self.layer_inx:
        #                 x = module[i](x)
        #     if name == "avgpool":
        #         x = F.adaptive_avg_pool2d(x, output_size=(7, 7))
        #         # x = nn.AdaptiveAvgPool2d(x, (7, 7))
        #     if name == "classifier":
        #         x = torch.flatten(x, 1)
        #         x = module(x)
        layer = 0
        for name, module in self.model.named_children():
            if layer > self.layer_inx:
                if name == "fc":
                    # x = F.avg_pool2d(x, 4)
                    # resnet18 ImageNet用这个
                    # x = F.adaptive_avg_pool2d(x, output_size=(1,1))
                    x = x.view(x.size(0), -1)
                x = module(x)
            layer += 1
        return x



features_out_hook = []
def hook(module, fea_in, fea_out):
    features_out_hook.append(fea_out.clone().detach())

class LayerwisePrediction_VGG(nn.Module):
    '''
    for 64x64 face generation. The hidden dimensions can be tuned.
    '''
    def __init__(self, model, hiddens=[16, 32, 64, 128, 256], latent_dim=100, prev_channels=3, img_size=32 ,device=None) -> None:
        super().__init__()
        # encoder
        self.model = model
        self.original_channels = prev_channels
        self.prev_channels = prev_channels
        modules = []
        self.img_length = img_size
        self.device = device
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(self.prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)))
            self.prev_channels = cur_channels

            self.img_length = math.ceil(self.img_length / 2)
        self.extractor = nn.Sequential(*modules)

        self.class_liner = nn.Linear(self.prev_channels * self.img_length * self.img_length,
                                     latent_dim)

        # decoder
        modules = []
        self.decoder_projection = nn.Linear(latent_dim, self.prev_channels * self.img_length * self.img_length)

        self.decoder_input_chw = (self.prev_channels, self.img_length, self.img_length)
        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]), nn.ReLU()))
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]), nn.ReLU(),
                nn.Conv2d(hiddens[0], self.original_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()))
        self.reconstructor = nn.Sequential(*modules)

    def forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        prediction = self.class_liner(x)

        x = self.decoder_projection(prediction)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        x_ = self.reconstructor(x)
        return prediction, x_

    def fit(self, data_loader, params):

        layer_inx, lr, n_step, momentum, wd = _parse_params(params)

        loss_fun = nn.CrossEntropyLoss()
        loss_kl = nn.KLDivLoss()
        BCE_loss = nn.BCELoss()

        self.extractor.to(self.device)
        self.class_liner.to(device)
        self.decoder_projection.to(device)
        self.reconstructor.to(self.device)

        extractor_optimizer = optim.SGD(self.extractor.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        reconstructor_optimizer = optim.SGD(self.reconstructor.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        pbar = tqdm(range(n_step), mininterval=1, ncols=100)


        for j in pbar:
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                features_out_hook.clear()
                #
                # for name, module in self.model.named_children():
                #     if name == "features":
                #         handle = module[layer_inx].register_forward_hook(hook=hook)
                #         y = self.model(data)
                #         handle.remove()
                #         break
                #original model using
                for name, module in self.model.named_children():
                    if name == f"layer{layer_inx + 1}":
                        handle = module.register_forward_hook(hook=hook)
                        y = self.model(data)
                        handle.remove()
                        break
                layer_features = features_out_hook[0]

                prediction, x_ = self.forward(layer_features.detach())

                # The output size after deconvolution may not match the expected size, so we use AdaptiveAvgPool2d to adjust it.
                if layer_features.shape != x_.shape:
                    pooling = nn.AdaptiveAvgPool2d((layer_features.shape[-2], layer_features.shape[-1]))
                    x_ = pooling(x_)
                truncted_model = Truncated_VGG(self.model, layer_inx)
                origin_result = F.softmax(truncted_model(layer_features), dim=1)
                con_result = F.log_softmax(truncted_model(x_), dim=1)

                loss_reconstructor = loss_kl(con_result.detach(), origin_result.detach()) + BCE_loss(F.sigmoid(x_), F.sigmoid(layer_features))

                reconstructor_optimizer.zero_grad()
                extractor_optimizer.zero_grad()
                loss_reconstructor.backward(retain_graph=True)
                reconstructor_optimizer.step()
                extractor_optimizer.step()

                prediction_, x_ = self.forward(layer_features.detach())
                loss_extractor = loss_kl(F.log_softmax(prediction.detach()), F.softmax(prediction_)) + loss_fun(prediction_,target.long())

                extractor_optimizer.zero_grad()
                loss_extractor.backward()
                extractor_optimizer.step()

            pbar.set_postfix_str("loss={:0.6f}".format(loss_extractor.item() + loss_reconstructor.item()))
