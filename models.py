from torch import nn, optim


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):
    def __init__(self, channels=64, repeat_num=4):
        super(Generator, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(3, channels, kernel_size=9, padding=4, bias=True))
        layers.append(nn.ReLU())

        for _ in range(repeat_num):
            layers.append(ResidualBlock(channels))

        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(channels, 3, kernel_size=9, padding=4, bias=True))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) * 0.58 + 0.5


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
        self.conv_layers = nn.Sequential(*layers)

        layers = list()
        layers.append(nn.Linear(128 * 7 * 7, 1024))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Linear(1024, 2))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        h = self.conv_layers(x)
        h = h.view(h.size(0), 128 * 7 * 7)
        return self.fc_layers(h)


class WESPE(nn.Module):
    def __init__(self, config, device):
        super(WESPE, self).__init__()

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

    def forward(self, x):
        pass
