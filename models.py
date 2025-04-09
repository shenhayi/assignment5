import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, device, num_classes=3):
        super(cls_model, self).__init__()
        self.num_classes = num_classes
        
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        ).to(device)
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        ).to(device)
        
        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        points = points.transpose(2, 1)  # (B, 3, N)
        x = self.encoder(points)  # (B, 128, N)
        x = torch.max(x, dim=2)[0]  # (B, 128)
        x = self.decoder(x)  # (B, num_classes)
        return x
        



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, device, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.num_seg_classes = num_seg_classes
        
        # Encoder with skip connections
        self.encoder1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        ).to(device)
        
        self.encoder2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        ).to(device)
        
        # Global feature extraction (simplified)
        self.global_feat = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        ).to(device)
        
        # Decoder with skip connections
        self.decoder1 = nn.Sequential(
            nn.Conv1d(256 + 128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        ).to(device)
        
        self.decoder2 = nn.Sequential(
            nn.Conv1d(128 + 64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        ).to(device)
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, num_seg_classes, 1),
        ).to(device)
        
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        points = points.transpose(2, 1)  # (B, 3, N)
        
        # Encoder path with skip connections
        feat1 = self.encoder1(points)  # (B, 64, N)
        feat2 = self.encoder2(feat1)   # (B, 128, N)
        
        # Global feature extraction
        global_feat = self.global_feat(feat2)  # (B, 256, N)
        global_feat = torch.max(global_feat, dim=2, keepdim=True)[0]  # (B, 256, 1)
        global_feat = global_feat.expand(-1, -1, points.size(2))  # (B, 256, N)
        
        # Decoder path with skip connections
        x = torch.cat([global_feat, feat2], dim=1)  # (B, 256+128, N)
        x = self.decoder1(x)  # (B, 128, N)
        
        x = torch.cat([x, feat1], dim=1)  # (B, 128+64, N)
        x = self.decoder2(x)  # (B, 64, N)
        
        # Final segmentation
        x = self.seg_head(x)  # (B, num_seg_classes, N)
        x = x.transpose(2, 1)  # (B, N, num_seg_classes)
        
        return x


