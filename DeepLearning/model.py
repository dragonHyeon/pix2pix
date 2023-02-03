import torch.nn as nn

from DeepLearning.layer import UNetUp, UNetDown


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """
        * 모델 구조 정의
        :param in_channels: in_channels 수
        :param out_channels: out_channels 수
        """

        super(GeneratorUNet, self).__init__()

        # (N, in_channels (3), H, W) -> (N, 64, H/2, W/2)
        self.down1 = UNetDown(in_channels=in_channels, out_channels=64, normalize=False)
        # (N, 64, H/2, W/2) -> (N, 128, H/4, W/4)
        self.down2 = UNetDown(in_channels=64, out_channels=128)
        # (N, 128, H/4, W/4) -> (N, 256, H/8, W/8)
        self.down3 = UNetDown(in_channels=128, out_channels=256)
        # (N, 256, H/8, W/8) -> (N, 512, H/16, W/16)
        self.down4 = UNetDown(in_channels=256, out_channels=512, dropout=0.5)
        # (N, 512, H/16, W/16) -> (N, 512, H/32, W/32)
        self.down5 = UNetDown(in_channels=512, out_channels=512, dropout=0.5)
        # (N, 512, H/32, W/32) -> (N, 512, H/64, W/64)
        self.down6 = UNetDown(in_channels=512, out_channels=512, dropout=0.5)
        # (N, 512, H/64, W/64) -> (N, 512, H/128, W/128)
        self.down7 = UNetDown(in_channels=512, out_channels=512, dropout=0.5)
        # (N, 512, H/128, W/128) -> (N, 512, H/256, W/256)
        self.down8 = UNetDown(in_channels=512, out_channels=512, normalize=False, dropout=0.5)

        # (N, 512, H/256, W/256) -> (N, 512, H/128, W/128)
        self.up1 = UNetUp(in_channels=512, out_channels=512, dropout=0.5)
        # (N, 1024, H/128, W/128) -> (N, 512, H/64, W/64)
        self.up2 = UNetUp(in_channels=1024, out_channels=512, dropout=0.5)
        # (N, 1024, H/64, W/64) -> (N, 512, H/32, W/32)
        self.up3 = UNetUp(in_channels=1024, out_channels=512, dropout=0.5)
        # (N, 1024, H/32, W/32) -> (N, 512, H/16, W/16)
        self.up4 = UNetUp(in_channels=1024, out_channels=512, dropout=0.5)
        # (N, 1024, H/16, W/16) -> (N, 256, H/8, W/8)
        self.up5 = UNetUp(in_channels=1024, out_channels=256)
        # (N, 512, H/8, W/8) -> (N, 128, H/4, W/4)
        self.up6 = UNetUp(in_channels=512, out_channels=128)
        # (N, 256, H/4, W/4) -> (N, 64, H/2, W/2)
        self.up7 = UNetUp(in_channels=256, out_channels=64)
        # (N, 128, H/2, W/2) -> (N, out_channels (3), H, W)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels (3), H, W)
        :return: 배치 개수 만큼의 변환된 이미지. (N, out_channels (3), H, W)
        """

        # (N, in_channels (3), H, W) -> (N, 64, H/2, W/2)
        d1 = self.down1(x)
        # (N, 64, H/2, W/2) -> (N, 128, H/4, W/4)
        d2 = self.down2(d1)
        # (N, 128, H/4, W/4) -> (N, 256, H/8, W/8)
        d3 = self.down3(d2)
        # (N, 256, H/8, W/8) -> (N, 512, H/16, W/16)
        d4 = self.down4(d3)
        # (N, 512, H/16, W/16) -> (N, 512, H/32, W/32)
        d5 = self.down5(d4)
        # (N, 512, H/32, W/32) -> (N, 512, H/64, W/64)
        d6 = self.down6(d5)
        # (N, 512, H/64, W/64) -> (N, 512, H/128, W/128)
        d7 = self.down7(d6)
        # (N, 512, H/128, W/128) -> (N, 512, H/256, W/256)
        d8 = self.down8(d7)

        # (N, 512, H/256, W/256) -> (N, 1024, H/128, W/128)
        u1 = self.up1(d8, d7)
        # (N, 1024, H/128, W/128) -> (N, 1024, H/64, W/64)
        u2 = self.up2(u1, d6)
        # (N, 1024, H/64, W/64) -> (N, 1024, H/32, W/32)
        u3 = self.up3(u2, d5)
        # (N, 1024, H/32, W/32) -> (N, 1024, H/16, W/16)
        u4 = self.up4(u3, d4)
        # (N, 1024, H/16, W/16) -> (N, 512, H/8, W/8)
        u5 = self.up5(u4, d3)
        # (N, 512, H/8, W/8) -> (N, 256, H/4, W/4)
        u6 = self.up6(u5, d2)
        # (N, 256, H/4, W/4) -> (N, 128, H/2, W/2)
        u7 = self.up7(u6, d1)
        # (N, 128, H/2, W/2) -> (N, out_channels (3), H, W)
        u8 = self.up8(u7)

        return u8


class Discriminator(nn.Module):
    def __init__(self):
        """
        * 모델 구조 정의
        """

        super(Discriminator, self).__init__()

        # 판별자
        self.main = nn.Sequential(
            # (N, 3, 64, 64) -> (N, 128, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (N, 128, 32, 32) -> (N, 256, 16, 16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (N, 256, 16, 16) -> (N, 512, 8, 8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (N, 512, 8, 8) -> (N, 1024, 4, 4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (N, 1024, 4, 4) -> (N, 1, 1, 1)
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, 3, 64, 64)
        :return: 배치 개수 만큼의 참 거짓 판별 결과. (N)
        """

        # (N, 3, 64, 64) -> (N, 1, 1, 1)
        x = self.main(x)
        # (N, 1, 1, 1) -> (N)
        x = x.view(-1)

        return x
