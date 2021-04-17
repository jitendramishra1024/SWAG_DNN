# ## DAVIDNET 


# uses this new ResNet Architecture (DAVIDNET)for Cifar10:

# 1.PrepLayer - 
    # Conv 3x3 s1, p1) >> BN >> RELU [64k]
# 2.Layer1 -
    # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    # Add(X, R1)
# 3.Layer 2 -
    # Conv 3x3 [256k]
    # MaxPooling2D
    # BN
    # ReLU
# 4.Layer 3 -
    # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    # Add(X, R2)
    
# MaxPooling with Kernel Size 4
# FC Layer 
# SoftMax

# #ONECYCLE POLICY 

# Uses One Cycle Policy such that:
# Total Epochs = 24
# Max at Epoch = 5
# LRMIN = FIND
# LRMAX = FIND
# NO Annihilation

# #AUGMENTATION 

# Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
# Batch size = 512
# Target Accuracy: 90%. 





##***************************************DAVIDNET****************************************##


import torch.nn as nn
import torch.nn.functional as F

class DAVIDNET(nn.Module):
    def __init__(self):
        super(DAVIDNET, self).__init__()
        
        
        # 1.PrepLayer - 
        # Conv 3x3 s1, p1) >> BN >> RELU [64k]
        
        self.preplayer = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3),stride=1, padding=1, bias=False), 
        nn.BatchNorm2d(64),
        nn.ReLU(),  
        )

        # 2.Layer1 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        # Add(X, R1)
        self.x1 = nn.Sequential(

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        )
        self.R1 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
        nn.BatchNorm2d(128),
        nn.ReLU(),

        )
        
        
        # 3.Layer 2 -
            # Conv 3x3 [256k]
            # MaxPooling2D
            # BN
            # ReLU            
        self.layer2 = nn.Sequential(

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False), 
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        )      
        
        # 4.Layer 3 -
            # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
            # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
            # Add(X, R2)
        self.x2 = nn.Sequential(

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        )

        self.R2 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
        nn.BatchNorm2d(512),
        nn.ReLU(),
        )
        # MaxPooling with Kernel Size 4
        
        # FC Layer 
        
        
        # SoftMax      
        self.pool = nn.MaxPool2d(4, 4)

        self.fc = nn.Linear(in_features = 512, out_features = 10, bias=False)
        
        
    def forward(self, x):

        preplayer = self.preplayer(x) 
        x1 = self.x1(preplayer)
        R1 = self.R1(x1)
        layer1 = x1+R1
        layer2 = self.layer2(layer1)
        x2 = self.x2(layer2)
        R2 = self.R2(x2)
        layer3 = R2+x2
        maxpool = self.pool(layer3)
        x = maxpool.view(maxpool.size(0),-1)
        fc = self.fc(x)

        return F.log_softmax(fc.view(-1,10), dim=-1)


