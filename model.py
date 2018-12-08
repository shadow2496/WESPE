import torch.nn as nn
import torch.optim as optim


class Flatten(nn.Module):
    def forward(self, x):
        n = x.shape[0]
        return x.view(n, 128 * 7 * 7)


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    def __init__(self, repeat_num=4):
        super(Generator, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(3, 64, kernel_size=9, padding=4, bias=True))
        layers.append(nn.ReLU())

        for _ in range(repeat_num):
            layers.append(ResidualBlock())

        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(64, 3, kernel_size=9, padding=4, bias=True))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x) * 0.58 + 0.5


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(channels, 48, kernel_size=11, padding=5, stride=4, bias=True))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(48, 128, kernel_size=5, padding=2, stride=2, bias=True))
        layers.append(nn.BatchNorm2d(128, momentum=0.1))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(192, momentum=0.1))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(192, momentum=0.1))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=2, bias=True))
        layers.append(nn.BatchNorm2d(128, momentum=0.1))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(Flatten())
        layers.append(nn.Linear(128 * 7 * 7, 1024))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Linear(1024, 2))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class WESPE:
    def __init__(self, config, device):
        self.gen_g = Generator()
        self.gen_f = Generator()
        self.gen_g.to(device)
        self.gen_f.to(device)

        if config.train:
            self.dis_c = Discriminator()
            self.dis_t = Discriminator(channels=1)
            self.dis_c.to(device)
            self.dis_t.to(device)

            self.g_optimizer = optim.Adam(self.gen_g.parameters(), lr=config.g_lr)
            self.f_optimizer = optim.Adam(self.gen_f.parameters(), lr=config.g_lr)
            self.c_optimizer = optim.Adam(self.dis_c.parameters(), lr=config.d_lr)
            self.t_optimizer = optim.Adam(self.dis_t.parameters(), lr=config.d_lr)
            self.criterion = nn.CrossEntropyLoss()
            self.criterion.to(device)
