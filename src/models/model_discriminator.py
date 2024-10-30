from torch import nn
import torch

class DiscriminatorStructures_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_output = nn.Sequential(
            
            nn.Linear(100, 64),
			nn.LeakyReLU(0.2),
			
            nn.Linear(64, 64),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2),

            nn.Linear(64, 64),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2),
			
			nn.Linear(64, 64),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2),

            nn.Linear(64, 64),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2),

            nn.Linear(64, 32),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(0.2),

			nn.Linear(32, 1),
            nn.Sigmoid()
		)
        
    def forward(self, x):
        return self.dense_output(x)

class DiscriminatorMultiNet16_4(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn16384 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 8, stride = 4, padding = 0, bias = False),
			nn.BatchNorm1d(4),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(4, 8, kernel_size = 8, stride = 4, padding = 0, bias = False),
			nn.BatchNorm1d(8),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(8, 16, kernel_size = 5, stride = 3, padding = 0, bias = False),
			nn.BatchNorm1d(16),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(16, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
			nn.BatchNorm1d(16),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(16, 32, kernel_size = 5, stride = 2, padding = 0, bias = False),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(32, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Flatten(),

			nn.Linear(64*9, 64),
			nn.LeakyReLU(0.2),

			nn.Linear(64, 1),
            nn.Sigmoid()
		)

        self.cnn8192 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 8, stride = 4, padding = 0, bias = False),
			nn.BatchNorm1d(4),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(4, 8, kernel_size = 8, stride = 4, padding = 0, bias = False),
			nn.BatchNorm1d(8),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(8, 16, kernel_size = 5, stride = 3, padding = 0, bias = False),
			nn.BatchNorm1d(16),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(16, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
			nn.BatchNorm1d(16),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(16, 32, kernel_size = 5, stride = 2, padding = 0, bias = False),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(32, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2, inplace=False),

			nn.Flatten(),

			nn.Linear(64*4, 64),
			nn.LeakyReLU(0.2),

			nn.Linear(64, 1),
            nn.Sigmoid()
		)

        self.cnn4096 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 8, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(64*9, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
		)

    def forward(self, x):

        p16384_0 = self.cnn16384(x[:,:,:16384])
        p16384_1 = self.cnn16384(x[:,:,16384:])

        p8192_0 = self.cnn8192(x[:,:,:8192])
        p8192_1 = self.cnn8192(x[:,:,8192:16384])
        p8192_2 = self.cnn8192(x[:,:,16384:24576])
        p8192_3 = self.cnn8192(x[:,:,24576:])

        p4096_0 = self.cnn4096(x[:,:,:4096])
        p4096_1 = self.cnn4096(x[:,:,4096:8192])
        p4096_2 = self.cnn4096(x[:,:,8192:12288])
        p4096_3 = self.cnn4096(x[:,:,12288:16384])
        p4096_4 = self.cnn4096(x[:,:,16384:20480])
        p4096_5 = self.cnn4096(x[:,:,20480:24576])
        p4096_6 = self.cnn4096(x[:,:,24576:28672])
        p4096_7 = self.cnn4096(x[:,:,28672:])

        out = torch.cat((p16384_0, p16384_1, p8192_0, p8192_1, p8192_2, p8192_3, p4096_0, p4096_1, p4096_2, p4096_3, p4096_4, p4096_5, p4096_6, p4096_7),dim=1)
        return out