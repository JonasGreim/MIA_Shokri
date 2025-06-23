import torch
import torch.nn as nn
import torch.nn.functional as F

# not tested yet
# idea to expand the SimpleCNN to a deeper architecture (here 15 conv.)


class DeepCnn(nn.Module):
    def __init__(self):
        super(DeepCnn, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 1
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 2
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 3
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 4
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 5
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 6
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 7
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 8
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 9
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 10
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 11
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 12
            nn.Tanh(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 13
            nn.Tanh(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 14
            nn.Tanh(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 15
            nn.Tanh(),
        )

        self.fc1 = nn.Linear(512 * 4 * 4, 256)  # Assuming input size is 64x64
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
