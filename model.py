import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # mean and std with shape [C x 1 x 1] can directly work with image Tensor of shape [B x C x H x W].
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class ContentLossLayer(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()  # as a non-variable

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)  # save tmp result while forwarding
        return x  # do not modify anything


class StyleLossLayer(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = self._gram_matrix(target).detach()

    def forward(self, x):
        mat = self._gram_matrix(x)
        self.loss = F.mse_loss(self.target, mat)
        return x

    @staticmethod
    def _gram_matrix(x):
        n, c, h, w = x.size()  # n = 1
        x = x.view(n * c, -1)
        return torch.mm(x, x.t()) / (n * c * h * w)


class TransferNet(nn.Module):
    def __init__(self, content_img, style_img, device, config):
        super().__init__()
        self._content_img = content_img.to(device)
        self._style_img = style_img.to(device)

        self._vgg = models.vgg19(pretrained=True).to(device).features.eval()
        vgg_input_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        vgg_input_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
        self._model = nn.Sequential(NormalizationLayer(vgg_input_mean, vgg_input_std))

        self.content_losses = []
        self.style_losses = []
        self._construct_losses(config)

    def _construct_losses(self, config):
        conv_cnt = 0
        for i, layer in enumerate(self._vgg.children()):
            name = 'layer_{}_{}'.format(i, layer.__class__.__name__)
            if isinstance(layer, nn.ReLU):
                # replace the in-place version of relu, otherwise runtime error will raise, but why?
                layer = nn.ReLU(inplace=False)
            self._model.add_module(name, layer)
            if isinstance(layer, nn.Conv2d):
                conv_cnt += 1
                if conv_cnt in config.content_layers:
                    target = self._model(self._content_img)
                    content_loss = ContentLossLayer(target)
                    self._model.add_module('content_loss_{}'.format(conv_cnt), content_loss)
                    self.content_losses.append(content_loss)
                if conv_cnt in config.style_layers:
                    target = self._model(self._style_img)
                    style_loss = StyleLossLayer(target)
                    self._model.add_module('style_loss_{}'.format(conv_cnt), style_loss)
                    self.style_losses.append(style_loss)
                if conv_cnt >= max(config.content_layers + config.content_layers):
                    break

    def forward(self, x):
        self._model(x)
        c_loss = sum(loss_layer.loss for loss_layer in self.content_losses)
        s_loss = sum(loss_layer.loss for loss_layer in self.style_losses)
        return c_loss, s_loss

