import torch
import torch.nn as nn
import torch.nn.functional as F

# This file contains the model definition of unet architecture
# This class is invoked during training. 
class UNet(nn.Module):

    def __init__(self, kernel_size=3, padding=1):           #, use_sigmoid = True):
        #self.use_sigmoid = use_sigmoid
        super(UNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=kernel_size, padding=padding)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(p=0.35)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(p=0.25)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=kernel_size, padding=padding)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=padding)
        self.maxpool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(p=0.35)

        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size=kernel_size, padding=padding)
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=kernel_size, padding=padding)
        self.conv5_t = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        #self.dropout5 = nn.Dropout(p = 0.3)
        self.conv6_1 = nn.Conv2d(1024, 512, kernel_size=kernel_size, padding=padding)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=padding)
        self.conv6_t = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=kernel_size, padding=padding)
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding)
        self.conv7_t = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv8_1 = nn.Conv2d(256, 128, kernel_size=kernel_size, padding=padding)
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.conv8_t = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv9_1 = nn.Conv2d(128, 64, kernel_size=kernel_size, padding=padding)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)

        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [B, F=32, H=256, W=256]
        conv1 = F.relu(self.conv1_1(x))
        # [B, F=32, H=256, W=256]
        conv1 = F.relu(self.conv1_2(conv1))
        # [B, F=32, H=128, W=128]
        pool1 = self.maxpool1(conv1)
        drop1=self.dropout1(pool1)

        # [B, F=64, H=128, W=128]
        conv2 = F.relu(self.conv2_1(drop1))
        # [B, F=64, H=128, W=128]
        conv2 = F.relu(self.conv2_2(conv2))
        # [B, F=64, H=64, W=64]
        pool2 = self.maxpool2(conv2)
        drop2=self.dropout2(pool2)

        # [B, F=128, H=64, W=64]
        conv3 = F.relu(self.conv3_1(drop2))
        # [B, F=128, H=64, W=64]
        conv3 = F.relu(self.conv3_2(conv3))
        # [B, F=128, H=32, W=32]
        pool3 = self.maxpool3(conv3)
        drop3=self.dropout3(pool3)
        

        # [B, F=256, H=32, W=32]
        conv4 = F.relu(self.conv4_1(drop3))
        # [B, F=256, H=32, W=32]
        conv4 = F.relu(self.conv4_2(conv4))
        # [B, F=256, H=16, W=16]
        pool4 = self.maxpool4(conv4)
        drop4=self.dropout4(pool4)
        

        # [B, F=512, H=16, W=16]
        conv5 = F.relu(self.conv5_1(drop4))
        # [B, F=512, H=16, W=16]
        conv5 = F.relu(self.conv5_2(conv5))
        #drop5 = self.dropout5(conv5)

        # [B, F=256, H=32, W=32] ⊕ [B, F=256, H=32, W=32] => [B, F=512, H=32, W=32]
        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)
        # [B, F=256, H=32, W=32]
        conv6 = F.relu(self.conv6_1(up6))
        # [B, F=256, H=32, W=32]
        conv6 = F.relu(self.conv6_2(conv6))

        # [B, F=128, H=64, W=64] ⊕ [B, F=128, H=64, W=64] => [B, F=256, H=64, W=64]
        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)
        # [B, F=128, H=64, W=64]
        conv7 = F.relu(self.conv7_1(up7))
        # [B, F=128, H=64, W=64]
        conv7 = F.relu(self.conv7_2(conv7))

        # [B, F=64, H=128, W=128] ⊕ [B, F=64, H=128, W=128] => [B, F=128, H=128, W=128]
        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)
        # [B, F=64, H=128, W=128]
        conv8 = F.relu(self.conv8_1(up8))
        # [B, F=64, H=128, W=128]
        conv8 = F.relu(self.conv8_2(conv8))

        # [B, F=32, H=256, W=256] ⊕ [B, F=32, H=256, W=256] => [B, F=64, H=256, W=256]
        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)
        # [B, F=32, H=256, W=256]
        conv9 = F.relu(self.conv9_1(up9))
        # [B, F=32, H=256, W=256]
        conv9 = F.relu(self.conv9_2(conv9))
        
        #if self.use_sigmoid == True:
        return self.sigmoid(self.conv10(conv9))
        #else:
            #return self.conv10(conv9)
    
    
def test():
    x = torch.randn((3, 1, 512, 512))
    model = UNet()
    print(model)
    preds = model(x)
    print(x.shape,preds.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
    
