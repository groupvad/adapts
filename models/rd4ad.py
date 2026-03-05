import copy

import torch
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
import numpy as np

from models.R4AD.resnet import resnet18, wide_resnet50_2
from models.R4AD.de_resnet import de_resnet18, de_wide_resnet50_2

class RD4AD(torch.nn.Module):

    def __init__(self, model_name, device, input_size = (224, 224)):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.input_size = input_size

        # for actual task
        # self.encoder, self.bn = resnet18(pretrained=True)
        # self.decoder = de_resnet18(pretrained=False)

        if model_name == "wide_resnet50_2":
            self.encoder, self.bn = wide_resnet50_2(pretrained=True)
            self.decoder = de_wide_resnet50_2(pretrained=False)
        elif model_name == "resnet18":
            self.encoder, self.bn = resnet18(pretrained=True)
            self.decoder = de_resnet18(pretrained=False)
            
        # for old tasks
        self.old_task_decoder = None
        self.old_task_bn = None

    def to(self, device: torch.device):
        self.encoder.to(device)
        self.bn.to(device)
        self.decoder.to(device)
        if self.old_task_decoder:
            self.old_task_decoder.to(device)
            self.old_task_bn.to(device)

    def train(self, *args, **kwargs):
        self.encoder.eval()
        self.bn.train()
        self.decoder.train()
        if self.old_task_decoder:
            self.old_task_decoder.eval()
            self.old_task_bn.eval()
        return super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.encoder.eval()
        self.bn.eval()
        self.decoder.eval()
        if self.old_task_decoder:
            self.old_task_decoder.eval()
            self.old_task_bn.eval()
        return super().eval(*args, **kwargs)

    def forward(self, batch: torch.Tensor, category=None):
        """
        Output tensors
        List[torch.Tensor] of len (n_layers)
        every tensor shape is (B C H W)
        """
        enc_batch = self.encoder(batch)
        bn_batch = self.bn(enc_batch)
        dec_batch = self.decoder(bn_batch)

        if self.training:
            if self.old_task_decoder:
                with torch.no_grad():
                    old_bn_batch = self.old_task_bn(enc_batch)
                    old_dec_batch = self.old_task_decoder(old_bn_batch)
                return enc_batch, bn_batch, dec_batch, old_dec_batch, old_bn_batch
            else:
                return enc_batch, bn_batch, dec_batch, None, None
        else:
            return self.post_process(enc_batch, dec_batch)

    def __call__(self, batch, category=None):
        return self.forward(batch, category)

    def post_process(self, enc_batch, dec_batch) -> torch.Tensor:
        anomaly_map = None
        blur = GaussianBlur(1, sigma = 4)

        #iterate over the feature extraction layers batches
        for i in range(len(enc_batch)):
            fs = dec_batch[i]
            ft = enc_batch[i]

            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=self.input_size, mode='bilinear', align_corners=True)

            if anomaly_map is None:
                anomaly_map = a_map
            else:
                anomaly_map += a_map

        anomaly_map = blur(anomaly_map)
        return anomaly_map, torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim = 1)[0]
