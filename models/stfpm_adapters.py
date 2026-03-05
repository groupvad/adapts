from typing import List, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import wide_resnet50_2, mobilenet_v2, resnet18

import copy
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
from torchvision.transforms import GaussianBlur
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.observer import MinMaxObserver

from datasets.mvtec_dataset import MVTecDataset
from datasets.visa_dataset import VISA_CATEGORIES


# -----------------------------
# Helper: cosine similarity
# -----------------------------
def cosine_similarity(x, y):
    """
    Compute cosine similarity between x and y.
    x: [B, D] or [D]
    y: [N, D]
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    return torch.mm(x_norm, y_norm.T)  # [B, N]

LAYERS_CHANNELS = {
    "mobilenet_v2": [24, 64, 160],
    "wide_resnet50_2": [256, 512, 1024],
    "resnet18": [64,128,256],
}

class LinearAdapter(nn.Module):
    def __init__(self, in_channels: int):
        super(LinearAdapter, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.activation = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # torch.nn.init.xavier_uniform(self.conv1.weight)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
        nn.init.orthogonal_(self.conv1.weight, gain=1.0)
        nn.init.orthogonal_(self.conv2.weight, gain=1.0)

        if self.conv1.bias is not None: nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None: nn.init.zeros_(self.conv2.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if self.training:
        #     x = x + torch.randn_like(x) * 0.001

        return self.conv2(self.activation(self.bn(self.conv1(x)))) + x

    def save(self, path: str):
        torch.save(self.state_dict(), path)

class LinearAdapterExpansion(nn.Module):
    def __init__(self, in_channels: int, expansion_factor = 2):
        super(LinearAdapterExpansion, self).__init__()
        mid_channels = int(in_channels * expansion_factor)
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(mid_channels)
        self.activation = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(mid_channels, in_channels, kernel_size=1)

        # torch.nn.init.xavier_uniform(self.conv1.weight)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
        nn.init.orthogonal_(self.conv1.weight, gain=1.0)
        nn.init.orthogonal_(self.conv2.weight, gain=1.0)

        if self.conv1.bias is not None: nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None: nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if self.training:
        #     x = x + torch.randn_like(x) * 0.1

        return self.conv2(self.activation(self.bn(self.conv1(x)))) + x

    def save(self, path: str):
        torch.save(self.state_dict(), path)

class BottleneckAdapter(nn.Module):
    def __init__(self, in_channels: int, reduction_factor: float = 0.25):
        """
        Adapter bottleneck con conv1x1 → ReLU → conv1x1.

        Args:
            in_channels (int): Numero di canali in input/output.
            reduction_factor (float): Fattore di riduzione dei canali (default=0.25 → ¼ dei canali).
        """
        super(BottleneckAdapter, self).__init__()
        reduced_channels = max(1, int(in_channels * reduction_factor))

        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.activation = torch.nn.ReLU()
        self.conv2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        # torch.nn.init.xavier_uniform(self.conv1.weight)
        # torch.nn.init.xavier_uniform(self.conv2.weight)

        nn.init.orthogonal_(self.conv1.weight, gain=1.0)
        nn.init.orthogonal_(self.conv2.weight, gain=1.0)

        if self.conv1.bias is not None: nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None: nn.init.zeros_(self.conv2.bias)


    def forward(self, x):
        # Residual connection: original features + adapted features
        return x + self.bn2(self.conv2(self.activation(self.bn1(self.conv1(x)))))

    def save(self, path: str):
        torch.save(self.state_dict(), path)

class STFPMAdapters(torch.nn.Module):

    def __init__ (
        self,
        model_name: str,
        ad_layers_idx: List[str],
        adapter_class: nn.Module,
        mask_size: Tuple[int, int],
        device: torch.device,
        use_cosine_loss=False,
    ):
        super().__init__()
        self.model_name = model_name
        self.ad_layers_idx = ad_layers_idx
        self.mask_size = mask_size

        self.build_teacher()
        self.build_student()

        self.device = device
        self.adapter_class = adapter_class

        self.layers_idx = [int(layer.replace("layer", "")) for layer in ad_layers_idx]
        self.n_features = sum([LAYERS_CHANNELS[model_name][layer-1] for layer in self.layers_idx])

        # freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        # freeze student parameters
        for param in self.student.parameters():
            param.requires_grad = False

        # define adapters
        self.build_adapters()

        self.is_eval_during_training = False

        # save the prototypes of the classes
        self.class_prototypes = None
        self.loadeded_adapters_ids = []

        self.use_cosine_loss = use_cosine_loss

    def build_student(self):
        if self.model_name == "mobilenet_v2":
            self.student = mobilenet_v2(weights='IMAGENET1K_V1')
        elif self.model_name == "wide_resnet50_2":
            self.student = wide_resnet50_2(weights='IMAGENET1K_V1')
        elif self.model_name == "resnet18":
            self.student = resnet18(weights='IMAGENET1K_V1')

    def build_teacher(self):
        if self.model_name == "mobilenet_v2":
            self.teacher = STFPMAdapters.get_feature_extractor(self.model_name, self.ad_layers_idx + ["features.18"])
        elif self.model_name == "wide_resnet50_2":
            self.teacher = STFPMAdapters.get_feature_extractor(self.model_name, self.ad_layers_idx + ["avgpool"])
        elif self.model_name == "resnet18":
            self.teacher = STFPMAdapters.get_feature_extractor(self.model_name, self.ad_layers_idx + ["avgpool"])

    def build_adapters(self):
        for layer in self.layers_idx:
            setattr(self, f'adapter_{layer}', self.adapter_class(in_channels=LAYERS_CHANNELS[self.model_name][layer-1]))
    
    def reset_adapters(self):
        for layer in self.layers_idx:
            setattr(self, f'adapter_{layer}', self.adapter_class(in_channels=LAYERS_CHANNELS[self.model_name][layer-1]))

    def prepare_adapters_quantization(self):
        feature_dims = [
            (1, 256,  56, 56),
            (1, 512,  28, 28),
            (1, 1024, 14, 14),
        ]

        # fuse the conv, bn, activation layers in the adapters
        for i in self.layers_idx:
            adapter = getattr(self, f"adapter_{i}")
            fused_adapter = tq.fuse_modules(
                adapter,
                [['conv1', 'bn', 'activation']],
            )
            setattr(self, f"adapter_{i}", fused_adapter)

        # crea qconfig che usa per-tensor per pesi
        qconfig = tq.QConfig(
            activation=tq.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
        )

        # prepare the adapters for quantization
        for i in self.layers_idx:
            dummy_input = torch.randn(feature_dims[i-1])
            adapter = getattr(self, f"adapter_{i}")
            setattr(self, f"adapter_{i}", prepare_fx(adapter, {"": qconfig}, example_inputs=dummy_input))

    def quantize_adapters(self, calibration_loader):

        self.prepare_adapters_quantization()

        for batch in tqdm(calibration_loader, desc=f"Calibrating adapters:"):
            self.student_forward(batch.to(self.device))

        for i in self.layers_idx:
            prepared = getattr(self, f"adapter_{i}")
            adapter_int8 = convert_fx(prepared)
            setattr(self, f"adapter_{i}", adapter_int8)

    def student_forward_wideresnet(self, batch: torch.Tensor):
        """
        Forward pass for the student model.

        Parameters:
        ----------
            - batch: input batch of data

        Returns:
        --------
            - student features
        """

        if len(batch.shape) == 3:
            batch = batch.unsqueeze(0)

        out_features = []
        out = batch
        for name, module in self.student.named_children():
            if name in self.ad_layers_idx:
                layer_id = int(name.replace("layer", ""))
                out = module(out)
                out = getattr(self, f"adapter_{layer_id}")(out)
                out_features.append(out)
                out = getattr(self, f"adapter_{layer_id}")(out)
            else:
                out = module(out)
            if name == self.ad_layers_idx[-1]:
                break

        return out_features

    def student_forward_mobilenet(self, batch: torch.Tensor):
        """
        Forward pass for the student model.

        Parameters:
        ----------
            - batch: input batch of data

        Returns:
        --------
            - student features
        """

        if len(batch.shape) == 3:
            batch = batch.unsqueeze(0)

        layer_count = 1
        out_features = []
        out = batch
        for name, module in self.student.features._modules.items():
            print(f"features.{name} shape: {out.shape}")
            if f"features.{name}" in self.ad_layers_idx:
                out = module(out)
                out = getattr(self, f"adapter_{layer_count}")(out)
                out_features.append(out)
                out = getattr(self, f"adapter_{layer_count}")(out)
                layer_count += 1
            else:
                out = module(out)
            if f"features.{name}" == self.ad_layers_idx[-1]:
                break

        return out_features

    def student_forward(self, batch: torch.Tensor):
        if self.model_name == "mobilenet_v2":
            return self.student_forward_mobilenet(batch)
        else:
            return self.student_forward_wideresnet(batch)

    def to(self, device: torch.device):
        self.teacher.to(device)
        self.student.to(device)
        for layer_id in self.layers_idx:
            getattr(self, f'adapter_{layer_id}').to(device)

    def forward(self, batch: torch.Tensor, category: str = None):

        if self.training:
            teacher_features, student_features = None, None
            with torch.no_grad():
                teacher_features = self.teacher(batch)
                if self.model_name == "mobilenet_v2":
                    class_vectors = teacher_features["features.18"].flatten(1).clone()
                else:
                    class_vectors = teacher_features["avgpool"].flatten(1).clone()
                teacher_features = [teacher_features[l] for l in self.ad_layers_idx]

            student_features = self.student_forward(batch)

            return class_vectors, teacher_features, student_features

        else:
            teacher_features = self.teacher(batch)

            if self.model_name == "mobilenet_v2":
                class_vectors = teacher_features["features.18"].flatten(1)
            else:
                class_vectors = teacher_features["avgpool"].flatten(1)
            teacher_features = [teacher_features[l] for l in self.ad_layers_idx]

            # prototypes_norm = F.normalize(self.class_prototypes, dim=1)  # [N, 2048]
            # vectors_norm = F.normalize(class_vectors, dim=1) # [B, 2048]

            # cosine_sim = vectors_norm @ prototypes_norm.T
            # indices = torch.argmax(cosine_sim, dim=1) # [B]

            if category is None:
                proto_norms = (self.class_prototypes ** 2).sum(dim=1).unsqueeze(0)  # [1, N]
                vec_norms = (class_vectors ** 2).sum(dim=1).unsqueeze(1)        # [B, 1]
                dists = vec_norms + proto_norms - 2 * class_vectors @ self.class_prototypes.T  # [B, N]
                indices = torch.argmin(dists, dim=1)  # [B]
                self.loadeded_adapters_ids = indices.cpu().numpy()
            else:
                indices = torch.tensor([VISA_CATEGORIES.index(category)] * batch.size(0)).to(self.device)

            anomaly_scores = []
            anomaly_maps = []

            # process each sample individually
            for i in range(batch.size(0)):

                # check if we are in eval during training so we don't load adapters every time
                if not self.is_eval_during_training:
                    self.load_adapter_from_index(indices[i].item())
                student_features = self.student_forward(batch[i])
                teacher_features_i = [features[i].unsqueeze(0) for features in teacher_features]
                anomaly_map, anomaly_score = self.post_process(teacher_features_i, student_features)

                anomaly_scores.append(anomaly_score)
                anomaly_maps.append(anomaly_map)

            anomaly_scores = torch.stack(anomaly_scores, dim=0)
            anomaly_maps = torch.cat(anomaly_maps, dim=0)

            return anomaly_maps, anomaly_scores

    def __call__(self, batch: torch.Tensor, category: str = None):
        return self.forward(batch, category)

    def train(self):
        self.training = True
        self.teacher.eval()
        self.student.eval()
        for layer_id in self.layers_idx:
            getattr(self, f'adapter_{layer_id}').train()

    def eval(self):
        self.training = False
        self.teacher.eval()
        self.student.eval()
        for layer_id in self.layers_idx:
            getattr(self, f'adapter_{layer_id}').eval()

    def save_adapters(self, category: str):

        adapters = {}

        for layer_id in self.layers_idx:
            adapter = getattr(self, f'adapter_{layer_id}')
            adapters[f'adapter_{layer_id}'] = adapter.state_dict()

        torch.save(adapters, os.path.join(self.adapters_save_path, category + ".pth"))

    def save_prototypes(self):
        torch.save(self.class_prototypes, os.path.join(self.adapters_save_path, "class_prototypes.pth"))

    def load_adapters_from_path(self, category: str):
        adapters = torch.load(os.path.join(self.adapters_save_path, category + ".pth"), map_location=self.device)

        for layer_id in self.layers_idx:
            getattr(self, f"adapter_{layer_id}").load_state_dict(adapters[f"adapter_{layer_id}"])

    def load_quantized_adapters_from_path(self, category: str):
        """
        Carica un dizionario di pesi quantizzati creando una struttura pulita ogni volta.
        """
        adapters_path = os.path.join(self.adapters_save_path, category + ".pth")
        saved_state_dicts = torch.load(adapters_path, map_location='cpu')

        # (batch, channels, height, width)
        feature_dims = [
            (1, 256,  56, 56),
            (1, 512,  28, 28),
            (1, 1024, 14, 14),
        ]

        qconfig = tq.QConfig(
            activation=tq.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
        )

        for i in self.layers_idx:
            # 1. Crea un adapter NUOVO e PULITO in FP32
            in_channels = feature_dims[i-1][1] # Prende i channels: 256, 512, o 1024
            fresh_adapter = LinearAdapter(in_channels=in_channels).to(self.device)
            fresh_adapter.eval() # IMPORTANTE: i layer BatchNorm devono essere in eval() per il fuse!

            # 2. FONDILO (Fuse)
            fused_adapter = tq.fuse_modules(
                fresh_adapter,
                [['conv1', 'bn', 'activation']],
            )

            # 3. PREPARALO per FX
            dummy_input = torch.randn(feature_dims[i-1]).to(self.device)
            prepared = prepare_fx(fused_adapter, {"": qconfig}, example_inputs=dummy_input)

            # 4. CONVERTILO in modulo quantizzato
            adapter_int8 = convert_fx(prepared)

            # 5. CARICA i pesi quantizzati
            adapter_int8.load_state_dict(saved_state_dicts[f"adapter_{i}"])

            # 6. Assegna il nuovo adapter al modello, sovrascrivendo quello vecchio
            setattr(self, f"adapter_{i}", adapter_int8)

    def load_adapter_from_index(self, class_idx: int):
        category = MVTecDataset.CATEGORIES[class_idx]
        #sprint(f"Loading adapters for category: {category}")
        #self.load_adapters_from_path(category)
        self.load_quantized_adapters_from_path(category)

    def load_prototypes(self, path: str):
        self.class_prototypes = torch.load(path, map_location=self.device)

    def post_process(self, t_feat, s_feat) -> torch.Tensor:

        """
        This method actually produces the anomaly maps for evalution purposes

        Parameters:
        ----------
            - t_feat: teacher features maps
            - s_feat: student features maps

        Returns:
        --------
            - anomaly maps

        """


        device = t_feat[0].device
        score_maps = torch.tensor([1.0], device=device)
        cosine_loss = torch.nn.CosineSimilarity()
        for j in range(len(t_feat)):
            # print(t_feat[j].shape)
            # print(s_feat[j].shape)
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            if self.use_cosine_loss:
                sm = 1 - cosine_loss(t_feat[j], s_feat[j]).unsqueeze(1)
            else:
                sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)

            sm = F.interpolate(
                sm, size=self.mask_size, mode="bilinear", align_corners=False
            )
            # aggregate score map by element-wise product
            score_maps = score_maps * sm

        anomaly_scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
        return score_maps, anomaly_scores

    def post_process_gem(self, t_feat, s_feat) -> torch.Tensor:
        """
        This method actually produces the anomaly maps for evalution purposes

        Parameters:
        ----------
            - t_feat: teacher features maps
            - s_feat: student features maps

        Returns:
        --------
            - anomaly maps, anomaly scores
        """

        device = t_feat[0].device
        # Inizializziamo a 1.0 per la moltiplicazione
        score_maps = torch.tensor([1.0], device=device)
        cosine_loss = torch.nn.CosineSimilarity()
        
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            
            if self.use_cosine_loss:
                sm = 1 - cosine_loss(t_feat[j], s_feat[j]).unsqueeze(1)
            else:
                sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)

            sm = F.interpolate(
                sm, size=self.mask_size, mode="bilinear", align_corners=False
            )
            
            # --- MODIFICA 1: NORMALIZZAZIONE MIN-MAX SPAZIALE PER OGNI LAYER ---
            # Troviamo il minimo e massimo per ogni immagine nel batch (B, 1, 1, 1)
            B = sm.shape[0]
            sm_flat = sm.view(B, -1)
            sm_min = sm_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            sm_max = sm_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            
            # Normalizziamo la singola mappa tra 0 e 1 (con epsilon per evitare divisioni per 0)
            sm_norm = (sm - sm_min) / (sm_max - sm_min + 1e-8)
            
            # Ora moltiplichiamo in modo sicuro: le mappe hanno tutte lo stesso "peso" (0 -> 1)
            score_maps = score_maps * sm_norm

        # --- MODIFICA 2: GAUSSIAN BLUR ---
        # Spalma i picchi rumorosi e unisce i cluster di pixel anomali.
        # Kernel 21 e sigma 4.0 sono lo standard industriale per maschere 256x256
        blur_filter = GaussianBlur(kernel_size=21, sigma=4.0)
        score_maps = blur_filter(score_maps)

        # Calcoliamo l'anomaly score globale dell'immagine sul picco della mappa sfocata.
        # Essendo sfocata, il max() ora rappresenta una vera regione anomala, non un pixel impazzito.
        anomaly_scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
        
        return score_maps, anomaly_scores

    def prototypes_to_tensor(self):
        self.class_prototypes = torch.stack(self.class_prototypes, dim=0)

    @staticmethod
    def get_feature_extractor(backbone: str, return_nodes, pretrained=True):
        """Get the feature extractor from the backbone CNN.

        Args:
            backbone (str): Backbone CNN network
            return_nodes (list[str]): A list of return nodes for the given backbone.

        Raises:
            NotImplementedError: When the backbone is efficientnet_b5
            ValueError: When the backbone is not supported

        Returns:
            GraphModule: Feature extractor.
        """
        model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        feature_extractor = create_feature_extractor(
            model=model, return_nodes=return_nodes
        )

        return feature_extractor
