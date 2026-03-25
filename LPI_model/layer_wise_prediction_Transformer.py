import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch

import os

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


features_out_hook = []
def hook(module, fea_in, fea_out):
    features_out_hook.append(fea_out.clone().detach())


class Truncated_Transformer(nn.Module):
    def __init__(self, model, layer_inx):
        super(Truncated_Transformer, self).__init__()
        self.model = model
        self.layer_inx = layer_inx
    def forward(self, x):
        for name, module in self.model.named_children():
            if name == "transformer":
                # Extract features from the model to be interpreted
                for i in range(len(module.layers)):
                    if i > self.layer_inx:
                        # Apply attention layer (index 0) and feed-forward layer (index 1)
                        x = module.layers[i][0](x)
                        x = module.layers[i][1](x)

        # Extract class token from sequence (first position)
        x = x[:, 0]
        # Project to latent space and apply MLP head for classification
        x = self.model.to_latent(x)
        x = self.model.mlp_head(x)
        return x


class LayerwisePrediction_Transformer(nn.Module):
    def __init__(self, model, input_size, class_size, device="cuda"):
        '''
        :param input_size: Dimension of hidden layer features
        :param class_size: Dimension of sample classes (number of output categories)
        :param model: The Transformer model to be interpreted/explained
        '''
        super(LayerwisePrediction_Transformer, self).__init__()

        # Initialize with the model to be interpreted, requiring model predictions and structure
        self.model = model
        self.device = device
        # Extractor network: maps input features to class predictions (feature -> 512 -> 256 -> class_size)
        self.extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Linear(512, 256),
            nn.Linear(256, class_size)
        )
        # Reconstructor network: maps predictions back to input feature space (class_size -> 256 -> 512 -> input_size)
        self.reconstructor = nn.Sequential(
            nn.Linear(class_size, 256),
            nn.Linear(256, 512),
            nn.Linear(512, input_size)
        )

    def forward(self, x):
        # Forward pass: extract predictions and reconstruct input
        prediction = self.extractor(x)
        x_ = self.reconstructor(prediction)
        return prediction, x_

    def fit(self, data_loader, params):
        # Parse training parameters
        layer_inx, lr, n_step, momentum, wd = _parse_params(params)
        # Initialize loss functions
        loss_fun = nn.CrossEntropyLoss()
        loss_kl = nn.KLDivLoss()
        BCE_loss = nn.BCELoss()

        self.extractor.to(self.device)
        self.reconstructor.to(self.device)
        extractor_optimizer = optim.Adam(self.extractor.parameters(), lr=lr,  weight_decay=wd)
        reconstructor_optimizer = optim.Adam(self.reconstructor.parameters(), lr=lr,  weight_decay=wd)
        pbar = tqdm(range(n_step), mininterval=1, ncols=100)


        for j in pbar:
            # Iterate through batches in data loader
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                features_out_hook.clear()
                # Register hook to capture intermediate layer features at specified transformer layer
                for name, module in self.model.named_children():
                    if name == "transformer":
                        handle = module.layers[layer_inx][1].register_forward_hook(hook=hook)
                        y = self.model(data)
                        handle.remove()
                        break
                # The second dimension of hidden layer features is the number of image patches
                layer_features = features_out_hook[0]
                # ViT class token mechanism: a token for classification is added at position 0,
                # so we only need to use the first feature (class token) for prediction
                cls_features = layer_features[:,0]
                prediction, cls_features_ = self.forward(cls_features.detach())

                con_layer_features = layer_features.clone()
                con_layer_features[:,0] = cls_features_

                truncted_model = Truncated_Transformer(self.model, layer_inx)
                origin_result = F.softmax(truncted_model(layer_features), dim=1)
                con_result = F.log_softmax(truncted_model(con_layer_features), dim=1)

                reconstructor_optimizer.zero_grad()
                extractor_optimizer.zero_grad()

                loss_reconstructor = loss_kl(con_result.detach(), origin_result.detach()) + BCE_loss(torch.sigmoid(cls_features_),torch.sigmoid(cls_features))


                loss_reconstructor.backward(retain_graph=True)
                reconstructor_optimizer.step()
                extractor_optimizer.step()

                prediction_, _ = self.forward(cls_features.detach())
                loss_extractor = loss_kl(F.log_softmax(prediction.detach(), dim=1), F.softmax(prediction_, dim=1)) + loss_fun(prediction_, target.long())

                extractor_optimizer.zero_grad()
                loss_extractor.backward()
                extractor_optimizer.step()

            pbar.set_postfix_str("loss={:0.6f}".format(loss_extractor.item(), loss_reconstructor.item()))
