import torch
from torch import nn
from torch import quantization
from torch.nn.intrinsic import qat
import torch.nn.functional as F
import random

import numpy as np

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
#imgaug.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

################################## VGG Feature Extractor #######################################
    
###################################### Base VGG Extractor ########################################
class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel=1, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x24

    def forward(self, input):
        return self.ConvNet(input)

    
###################################### Quantized VGG ########################################
class VGGQuantized_FeatureExtractor(nn.Module):

    def __init__(self, input_channel=1, output_channel=512, qatconfig='fbgemm'):
        super(VGGQuantized_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        
        self.qconfig = quantization.get_default_qat_qconfig(qatconfig)
        
        self.ConvNet = nn.Sequential(
            qat.ConvReLU2d(input_channel, self.output_channel[0], 3, 1, 1, qconfig=self.qconfig),
            nn.MaxPool2d(2, 2),  # 64x16x50
            qat.ConvReLU2d(self.output_channel[0], self.output_channel[1], 3, 1, 1, qconfig=self.qconfig),
            nn.MaxPool2d(2, 2),  # 128x8x25
            qat.ConvReLU2d(self.output_channel[1], self.output_channel[2], 3, 1, 1, qconfig=self.qconfig),  # 256x8x25
            qat.ConvReLU2d(self.output_channel[2], self.output_channel[2], 3, 1, 1, qconfig=self.qconfig),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            qat.ConvBnReLU2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False, qconfig=self.qconfig),  # 512x4x25
            qat.ConvBnReLU2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False, qconfig=self.qconfig),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            qat.ConvReLU2d(self.output_channel[3], self.output_channel[3], 2, 1, 0, qconfig=self.qconfig))  # 512x1x24
        
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.ConvNet(x)
        x = self.dequant(x)
        return x 
    
    
###################### Teacher VGG with modified output shape ##################################
# class VGG_FeatureExtractor(nn.Module):

#     def __init__(self, input_channel=1, output_channel=512):
#         super(VGG_FeatureExtractor, self).__init__()
#         self.output_channel = [int(output_channel / 8), int(output_channel / 4),
#                                int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
#         self.ConvNet = nn.Sequential(
#             nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # 64x16x50
#             nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # 128x8x25
#             nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
#             nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
#             nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
#             nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
#             nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
#             nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
#             nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
#             nn.MaxPool2d(2, (2, 1)),  # 512x2x25
#             nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, (0, 1)), nn.ReLU(True))  # 512x1x24

#     def forward(self, input):
#         return self.ConvNet(input)
    
    
############################ Original Extractor from Paper ##########################################
# class VGG_FeatureExtractor(nn.Module):
#     """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

#     def __init__(self, input_channel=1, output_channel=512):
#         super(VGG_FeatureExtractor, self).__init__()
#         self.output_channel = [int(output_channel / 8), int(output_channel / 4),
#                                int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
#         self.ConvNet = nn.Sequential(
#             nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # 64x16x50
#             nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # 128x8x25
#             nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1, bias=False), 
#             nn.BatchNorm2d(self.output_channel[2]), nn.ReLU(True),  # 256x8x25
#             nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
#             nn.MaxPool2d(2, (2, 1)),  # 256x4x25
#             nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
#             nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
#             nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1), nn.ReLU(True),
#             nn.MaxPool2d(2, (2, 1)),  # 512x2x25
#             nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True),  # 512x1x24
#             nn.BatchNorm2d(self.output_channel[3]))

#     def forward(self, input):
#         return self.ConvNet(input)


################################## VGG Feature Extractor #######################################



########################BILSTM (SEquence Modelling)####################################


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, in_channels, hidden_size):
        super(BidirectionalLSTM, self).__init__()
        
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, bidirectional=True, num_layers=2)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1, "the height of conv must be 1"
        x = x.squeeze(axis=2)
        x = x.permute(2, 0, 1)  # (NTC)(width, batch, channels)
        x, _ = self.lstm(x)

        #print("Output size: " + str(x.shape))
        return x


########################BILSTM (SEquence Modelling)####################################

class CRNN(nn.Module):
    def __init__(self, nclass, hidden_size=256, qatconfig=None, bias=True):
        super(CRNN, self).__init__()
        
        #VGG backbone
        if qatconfig is None:
            self.backbone = VGG_FeatureExtractor()
        
        else:
            # Quantization aware training config
            self.backbone = VGGQuantized_FeatureExtractor(qatconfig=qatconfig)
            self.backbone = quantization.prepare_qat(self.backbone)
        
        #importing rnn
        self.rnn = BidirectionalLSTM(512, hidden_size)
        self.embedding = nn.Linear(hidden_size * 2, nclass, bias=bias)
        
        if qatconfig is None:
            self._initialize_weights(bias)
        else:
            self._initialize_weights_qat()


    def forward(self, x):

        #print(input.shape)
        x = x.float()
        
        # conv features
        x = self.backbone(x)        
        x = self.rnn(x)
        #print("RNN output shape: " + str(x.shape))
        
        T, b, h = x.size()
        x = x.view(T * b, h)
        x = self.embedding(x)  # [T * b, nOut]
        #print("Embedding output shape: " + str(x.shape))
        x = x.view(T, b, -1)
        #print("Output size: " + str(x.shape))
        
        return x
    
    def _initialize_weights(self, bias=True):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if name!="embedding" or (name=="embedding" and bias):
                    nn.init.zeros_(m.bias)
                
    def _initialize_weights_qat(self):
        for m in self.modules():
            if isinstance(m, (qat.ConvReLU2d, qat.ConvBnReLU2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
        

def get_crnn(n_class, qatconfig=None, bias=True):
    
    return CRNN(n_class, qatconfig=qatconfig, bias=bias)
