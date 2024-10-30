from torch import nn
import torch

# Model with concatenation
class CNNGeneratorBigConcat(nn.Module):
	def __init__(self):
		super().__init__()
		self.avgPool2 = nn.AvgPool1d(2, ceil_mode=True)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='linear')
		self.cnn1 = nn.Sequential(
                 	nn.Conv1d(1, 16, kernel_size=1, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnn2 = nn.Sequential(
                 	nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn2_ = nn.Sequential(
                 	nn.Conv1d(32, 32, kernel_size=2, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn4 = nn.Sequential(
                 	nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn4_ = nn.Sequential(
                 	nn.Conv1d(64, 64, kernel_size=4, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn8 = nn.Sequential(
                 	nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn8_ = nn.Sequential(
                 	nn.Conv1d(128, 128, kernel_size=8, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn16 = nn.Sequential(
                 	nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn16_ = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=16, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32_ = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn64 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge1 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge2 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge3 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans64 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans32 = nn.Sequential(
                 	nn.ConvTranspose1d(512, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans16 = nn.Sequential(
                 	nn.ConvTranspose1d(512, 128, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnnTrans8 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 64, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnnTrans4 = nn.Sequential(
                 	nn.ConvTranspose1d(128, 32, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnnTrans2 = nn.Sequential(
                 	nn.ConvTranspose1d(64, 16, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnnTransOut = nn.Sequential(
             	nn.ConvTranspose1d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            )

	def forward(self, z):
		res1 = self.cnn1(z)
		out = res1

		out = self.avgPool2(out)
		out = self.cnn2(out)
		res2 = self.cnn2_(out)
		out = res2

		out = self.avgPool2(out)
		out = self.cnn4(out)
		res4 = self.cnn4_(out)
		out = res4

		out = self.avgPool2(out)
		out = self.cnn8(out)
		res8 = self.cnn8_(out)
		out = res8

		out = self.avgPool2(out)
		out = self.cnn16(out)
		res16 = self.cnn16_(out)
		out = res16

		out = self.avgPool2(out)
		out = self.cnn32(out)
		res32 = self.cnn32_(out)
		out = res32

		out = self.avgPool2(out)
		out = self.cnn64(out)

		out = self.bridge1(out)
		out = self.bridge2(out)
		out = self.bridge3(out)

		out = self.cnnTrans64(out)
		out = self.upsample2(out)
		out = torch.cat((out[:,:,0:-1], res32), dim=1) 

		out = self.cnnTrans32(out)
		out = self.upsample2(out)
		out = torch.cat((out, res16), dim=1) 

		out = self.cnnTrans16(out)
		out = self.upsample2(out)
		out = torch.cat((out[:,:,0:-1], res8), dim=1) 

		out = self.cnnTrans8(out)
		out = self.upsample2(out)
		out = torch.cat((out, res4), dim=1) 

		out = self.cnnTrans4(out)
		out = self.upsample2(out)
		out = torch.cat((out[:,:,0:-1], res2), dim=1) 

		out = self.cnnTrans2(out)
		out = self.upsample2(out)
		out = torch.cat((out, res1), dim=1)

		out = self.cnnTransOut(out)
		return out

