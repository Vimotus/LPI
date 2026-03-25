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


class Truncated_ResNet(nn.Module):
    def __init__(self, model, layer_inx):
        super(Truncated_ResNet, self).__init__()
        self.model = model
        self.layer_inx = layer_inx
    def forward(self, x):
        # Execute modules starting from the specified layer index
        # Note: ResNet has convolution and BN layers before residual blocks, so add +2
        layer = 0
        for name, module in self.model.named_children():
            # Because residual blocks have a convolutional layer and BN layer at the beginning, need to add +2

            if layer > self.layer_inx + 2:

                if name == "linear":
                    # For ResNet18 ImageNet, use adaptive average pooling

                    # x = F.avg_pool2d(x, 4)
                    #ResNet18 ImageNet use this
                    # x = F.adaptive_avg_pool2d(x, output_size=(1,1))
                    x = x.view(x.size(0), -1)

                x = module(x)
            layer += 1
        return x

features_out_hook = []
def hook(module, fea_in, fea_out):
    features_out_hook.append(fea_out.clone().detach())
class LayerwisePrediction_ResNet18(nn.Module):
    '''
    VAE for 64x64 face generation. The hidden dimensions can be tuned.
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
        # Build encoder layers with progressive downsampling
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(self.prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)))
            self.prev_channels = cur_channels
            # self.img_length //= 2
            # Round up to nearest integer
            self.img_length = math.ceil(self.img_length / 2)
        self.extractor = nn.Sequential(*modules)
        # Classification layer: maps flattened features to class predictions

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
                                       output_padding=1)))
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
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
        # layer_features, targets = _parse_data(data, self.device)
        # Parse training parameters
        layer_inx, lr, n_step, momentum, wd = _parse_params(params)

        loss_fun = nn.CrossEntropyLoss()
        loss_kl = nn.KLDivLoss()
        BCE_loss = nn.BCELoss()


        self.extractor.to(self.device)
        self.class_liner.to(device)
        self.decoder_projection.to(device)
        self.reconstructor.to(self.device)
        # Initialize optimizers for extractor and reconstructor
        extractor_optimizer = optim.SGD(self.extractor.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        reconstructor_optimizer = optim.SGD(self.reconstructor.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        pbar = tqdm(range(n_step), mininterval=1, ncols=100)


        for j in pbar:
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                features_out_hook.clear()
                for name, module in self.model.named_children():
                    if name == f"layer{layer_inx+1}":
                        handle = module.register_forward_hook(hook=hook)
                        y = self.model(data)
                        handle.remove()
                        break

                layer_features = features_out_hook[0]

                prediction, x_ = self.forward(layer_features.detach())
                truncted_model = Truncated_ResNet(self.model, layer_inx)
                origin_result = self.model(layer_features)
                con_result = truncted_model(x_)
                # Compute reconstructor loss: KL divergence for prediction consistency + BCE for feature reconstruction
                # Note: For KLDivLoss, first parameter is predicted log-probabilities, second is target probabilities

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
