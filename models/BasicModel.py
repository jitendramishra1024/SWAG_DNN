
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        """ This function instantiates all the model layers """

        super(Net, self).__init__()
        dropout_rate=0
        self.convblock1 = nn.Sequential(
            # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            # Input: 32x32x32 | Output: 32x32x64 | RF: 5x5
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.transblock1 = nn.Sequential(
            # Input: 32x32x64 | Output: 16x16x64 | RF: 6x6
            nn.MaxPool2d(2, 2),  
            # Input: 16x16x64 | Output: 16x16x32 | RF: 6x6
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  
        )

        self.convblock2 = nn.Sequential(
            # Input: 16x16x32 | Output: 16x16x32 | RF: 10x10
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            # Input: 16x16x32 | Output: 16x16x64 | RF: 14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.transblock2 = nn.Sequential(
            # Input: 16x16x64 | Output: 8x8x64 | RF: 16x16
            nn.MaxPool2d(2, 2),  
            # Input: 8x8x64 | Output: 8x8x32 | RF: 16x16
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  
        )

        self.convblock3 = nn.Sequential(
             # Input: 8x8x32 | Output: 8x8x32 | RF: 24x24
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            # Depthwise separable convolution
            # Input: 8x8x32 | Output: 8x8x32 | RF: 32x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32, padding=1), 
             # Input: 8x8x32 | Output: 8x8x64 | RF: 32x32 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.transblock3 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # Input: 8x8x64 | Output: 4x4x64 | RF: 36x36
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  # Input: 4x4x64 | Output: 4x4x32 | RF: 36x36
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
              # Input: 4x4x32 | Output: 4x4x32 | RF: 52x52
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),

            # Dilated convolution
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, dilation=2),
            # Input: 4x4x32 | Output: 2x2x64 | RF: 68X68
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )

        self.gap = nn.Sequential(
            #Input 2X2X64 |Output  1X1X64 | RF :76X 76
            nn.AvgPool2d(kernel_size=2),
            #Input 1X1X64 | Output 1X1X10 | RF :76 X76
            nn.Conv2d(in_channels=64,out_channels=10,kernel_size=(1,1),padding = 0, bias = True)
            #nn.AdaptiveAvgPool2d(1)
           
        ) 


        
    
    def forward(self, x):
        """ This function defines the network structure """

        x = self.convblock1(x)
        x = self.transblock1(x)
        x = self.convblock2(x)
        x = self.transblock2(x)
        x = self.convblock3(x)
        x = self.transblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        #x = self.fc(x)
        return x
