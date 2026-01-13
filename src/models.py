import torch
import torch.nn as nn
import timm

class DeepfakeClassifierDINOv3(nn.Module):
    def __init__(self, model_name='vit_huge_plus_patch16_dinov3.lvd1689m', num_classes=0, pretrained=False, checkpoint_path="./model/timm_dinov3.pth"):
        super(DeepfakeClassifierDINOv3, self).__init__()
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            num_classes=0,
            dynamic_img_size=True 
        )
        
        self.embed_dim = self.backbone.num_features 
        
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 1024), 
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights(self.head)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone.forward_features(x) 
        cls_token = features[:, 0, :]
        reg_tokens = features[:, 1:5, :]
        reg_avg = torch.mean(reg_tokens, dim=1)
        combined_features = torch.cat((cls_token, reg_avg), dim=1)
        output = self.head(combined_features)
        
        return output