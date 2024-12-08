from abc import ABC, abstractmethod
import torch.nn as nn
class FinetuneModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_backbone(self):
        """
        Phương thức trừu tượng để xây dựng phần backbone của mô hình.
        Các lớp con phải ghi đè phương thức này.
        """
        pass

    @abstractmethod
    def build_classification(self):
        """
        Phương thức trừu tượng để xây dựng phần classification của mô hình.
        Các lớp con phải ghi đè phương thức này.
        """
        pass
        
    def forward(self, x):
        x = self.build_backbone()(x)
        x = self.build_classification()(x)
        return x
    
import torch
class Dinov2(FinetuneModel):
    def __init__(self, type_model, num_classes, device):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', type_model).to(device)
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224).to(device)
            sample_output = self.backbone(sample_input)
        self.classification = nn.Linear(
            in_features=sample_output.shape[-1], out_features=num_classes, bias=True, device=device
        )

    def __str__(self):
        return "Dinov2"
        
    def build_backbone(self):
        return self.backbone

    def build_classification(self):
        return self.classification
