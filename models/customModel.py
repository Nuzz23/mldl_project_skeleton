import torch

class CustomNet(torch.nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2), # 112
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), # 56
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), # 28
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(4, 4), # Add a max pooling layer
            torch.nn.Flatten(),
            torch.nn.Linear(256*7**2, 200),  # 200 output classes
        )

    def forward(self, x):
        # Define forward pass
        return self.linear_relu_stack(x)