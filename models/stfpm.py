from typing import List, Tuple
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
import copy

class STFPM(torch.nn.Module):

    def __init__ (
        self,
        model_name: str,
        ad_layers_idx: List[int],
        mask_size: Tuple[int, int]
    ):
        super().__init__()
        self.model_name = model_name
        self.ad_layers_idx = ad_layers_idx
        self.mask_size = mask_size
        self.teacher = STFPM.get_feature_extractor(model_name, ad_layers_idx)
        self.old_tasks_teacher = None
        self.student = STFPM.get_feature_extractor(model_name, ad_layers_idx, pretrained=False)

        # freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

    def reset_student(self, device):
        self.student = STFPM.get_feature_extractor(self.model_name, self.ad_layers_idx, pretrained=False)
        self.student.to(device)

    def set_old_tasks_teacher(self):
        self.old_tasks_teacher = STFPM.get_feature_extractor(self.model_name, self.ad_layers_idx)
        self.old_tasks_teacher.load_state_dict(copy.deepcopy(self.student.state_dict()))

    def to(self, device: torch.device):
        self.teacher.to(device)
        self.student.to(device)
        if self.old_tasks_teacher:
            self.old_tasks_teacher.to(device)

    def forward(self, batch: torch.Tensor, category = None):

        if self.training:
            teacher_features, student_features, previous_tasks_teacher_features = None, None, None
            with torch.no_grad():
                if self.old_tasks_teacher:
                    previous_tasks_teacher_features = list(self.old_tasks_teacher(batch).values())
                teacher_features = list(self.teacher(batch).values())
            student_features = list(self.student(batch).values())

            if self.old_tasks_teacher:
                return previous_tasks_teacher_features, teacher_features, student_features
            else:
                return teacher_features, student_features

        else:
            student_features = list(self.student(batch).values())
            teacher_features = list(self.teacher(batch).values())

            return self.post_process(teacher_features, student_features)

    def __call__(self, batch: torch.Tensor, category=None):
        return self.forward(batch)

    def train(self):
        self.training = True
        self.teacher.eval()
        self.student.train()

    def eval(self):
        self.training = False
        self.teacher.eval()
        self.student.eval()


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
            #sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = 1 - cosine_loss(t_feat[j], s_feat[j]).unsqueeze(1)
            sm = F.interpolate(
                sm, size=self.mask_size, mode="bilinear", align_corners=False
            )
            # aggregate score map by element-wise product
            score_maps = score_maps * sm

        anomaly_scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
        return score_maps, anomaly_scores


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
