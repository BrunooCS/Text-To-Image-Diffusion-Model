import torch
from torch import nn
from models.modules import Block
from models.time_encoder import SinusoidalPositionEmbeddings

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=64, out_channels=1, time_dim=256, embedding_size=384,num_classes=10):
        super().__init__()
        # Time embedding
        self.time_dim = time_dim
        self.embedding_size = embedding_size
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True),
        )
                
        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Input 128x128
        self.downs = nn.ModuleList([    
            Block(model_channels, model_channels*2, time_dim+embedding_size),    # 64x64
            Block(model_channels*2, model_channels*4, time_dim+embedding_size),  # 32x32
            Block(model_channels*4, model_channels*8, time_dim+embedding_size),  # 16x16
            Block(model_channels*8, model_channels*16, time_dim+embedding_size)  # 8x8
        ])
        
        # Middle - Standard convolution 
        self.middle_block = nn.Sequential(
            nn.Conv2d(model_channels*16, model_channels*16, 3, padding=1),
            nn.BatchNorm2d(model_channels*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_channels*16, model_channels*16, 3, padding=1),
            nn.BatchNorm2d(model_channels*16),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling
        self.ups = nn.ModuleList([
            Block(model_channels*16, model_channels*8, time_dim+embedding_size, up=True),
            Block(model_channels*8, model_channels*4, time_dim+embedding_size, up=True),
            Block(model_channels*4, model_channels*2, time_dim+embedding_size, up=True),
            Block(model_channels*2, model_channels, time_dim+embedding_size, up=True)
        ])
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(model_channels*2, model_channels, 3, padding=1),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )
        
    def forward(self, x, t, y_emb):
        # 1) Initial conv
        x = self.conv0(x)
        residual1 = x
        
        # 2) Time embedding
        t = self.time_mlp(t)
        # 3) Text embedding
        t = torch.cat((t, y_emb), dim=1)
        
        # 4) Down samples
        skips = []
        for layer in self.downs:
            x = layer(x, t)
            skips.append(x)
        
        # Middle
        x = self.middle_block(x)
        
        # 5) Up samples
        for i, layer in enumerate(self.ups):
            x = torch.cat([x, skips[-i-1]], dim=1)
            x = layer(x, t)
        
        # 6) Concatenate with residual connection
        x = torch.cat([x, residual1], dim=1)
        return self.final_conv(x)