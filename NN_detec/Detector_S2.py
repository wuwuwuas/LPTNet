import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroCoordAttention(nn.Module):
    def __init__(self, d_model, patch_size=7, K=3):
        super().__init__()
        self.K = K
        self.d_model = d_model
        
        ax = torch.linspace(-1, 1, patch_size)
        gy, gx = torch.meshgrid(ax, ax, indexing='ij')
        grid = torch.stack([gy.flatten(), gx.flatten()], dim=0) 
        self.register_buffer('grid', grid)
        
        self.coord_proj = nn.Linear(2, d_model)
        
        self.query_slots = nn.Parameter(torch.randn(K, d_model))
        nn.init.orthogonal_(self.query_slots)
        
        self.scale = d_model ** -0.5

    def forward(self, feat_flatten):
        B = feat_flatten.size(0)
        
        grid_embed = self.coord_proj(self.grid.t()).unsqueeze(0)
        
        pos_feat = feat_flatten + grid_embed
        
        queries = self.query_slots.unsqueeze(0).expand(B, -1, -1)

        attn_scores = torch.bmm(queries, pos_feat.transpose(1, 2)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1) 
        
        slot_features = torch.bmm(attn_probs, feat_flatten)
        
        return slot_features

class PatchRegressor_Slots(nn.Module):
    def __init__(self, in_ch=1, patch_size=7, d_model=64, K=3):
        super().__init__()
        self.K = K
        
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * patch_size * patch_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_model * patch_size * patch_size),
            nn.ReLU(inplace=True)
        )
        self.d_model = d_model
        self.patch_size = patch_size
        self.coord_attn_slots = MicroCoordAttention(d_model, patch_size, K)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5) 
        )
        
        self._initialize_biases()

    def _initialize_biases(self):
        nn.init.zeros_(self.regressor[-1].weight)
        
        bias = torch.zeros(5)
        bias[4] = -3.0 
        
        with torch.no_grad():
            self.regressor[-1].bias.copy_(bias)

    def forward(self, patch):
        B = patch.size(0)

        feat = self.backbone(patch)
        feat_flatten = feat.view(B, self.patch_size * self.patch_size, self.d_model)
        slot_features = self.coord_attn_slots(feat_flatten)

        preds = self.regressor(slot_features)
        
        y_off = torch.tanh(preds[..., 0:1]) / 2.0  
        x_off = torch.tanh(preds[..., 1:2]) / 2.0  
        logI = preds[..., 2:3]
        logd = preds[..., 3:4]
        conf = torch.sigmoid(preds[..., 4:5])

        out = torch.cat([y_off, x_off, logI, logd, conf], dim=-1)
        return out