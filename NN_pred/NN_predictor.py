import torch.nn as nn


class LagrangeGRUPredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=3):
        super(LagrangeGRUPredictor, self).__init__()
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),  
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        
        last_hidden = h_n[-1, :, :] # [Batch, hidden_dim]
        
        pred_disp = self.mlp(last_hidden) # [Batch, 3]
        
        return pred_disp

