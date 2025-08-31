import torch
import torch.nn as nn


# Squeeze-and-Excitation bloki
class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        # Adaptive o'rtacha hovuzlash
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Neural tarmoq qismi
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x


# Stem blok (asosiy kirish bloki)
class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        # Birinchi konvolyutsiya bloki
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        # Shortcut (o'tish yo'li) konvolyutsiya
        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        # Squeeze-Excitation bloki yordamida e'tibor mexanizmi
        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)  # Shortcut va asosiy yo'ldan kelgan ma'lumotlarni yig'ish
        return y


# ResNet bloki (asosiy blok)
class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        # Konvolyutsiya va batch normalization
        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        # Shortcut (o'tish yo'li)
        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        # E'tibor mexanizmi
        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)  # Shortcut bilan ma'lumotlarni birlashtirish
        return y


# ASPP bloki (Atrous Spatial Pyramid Pooling)
class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        # Kengaytirilgan konvolyutsiyalar (dilation)
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_c)
        )

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)

    def forward(self, inputs):
        # Har xil darajadagi kengaytirilgan konvolyutsiya natijalari
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)  # Yig'ilgan natijalarni bitta qatlam orqali o'tkazish
        return y


# Diqqat bloki (Attention Block)
class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        # G ma'lumot uchun konvolyutsiya
        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )

        # X ma'lumot uchun konvolyutsiya
        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        # G va X ma'lumotlarni birlashtirish
        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y


# Dekoder bloki (Decoder Block)
class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # E'tibor bloki bilan dekodlash
        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")  # Upsampling
        self.r1 = ResNet_Block(in_c[0] + in_c[1], out_c, stride=1)  # ResNet bloki

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)  # Upsample qilingan ma'lumotlarni birlashtirish
        d = self.r1(d)
        return d


# ResUNet++ ni qurish
class build_resunetplusplus(nn.Module):
    def __init__(self):
        super().__init__()

        # Kodlash bosqichlari
        self.c1 = Stem_Block(3, 16, stride=1)
        self.c2 = ResNet_Block(16, 32, stride=2)
        self.c3 = ResNet_Block(32, 64, stride=2)
        self.c4 = ResNet_Block(64, 128, stride=2)

        # ASPP
        self.b1 = ASPP(128, 256)

        # Dekodlash bosqichlari
        self.d1 = Decoder_Block([64, 256], 128)
        self.d2 = Decoder_Block([32, 128], 64)
        self.d3 = Decoder_Block([16, 64], 32)

        # Yakuniy ASPP va chiqish konvolyutsiyasi
        self.aspp = ASPP(32, 16)
        self.output = nn.Conv2d(16, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        # Kodlash bosqichlari
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)

        # ASPP bloki
        b1 = self.b1(c4)

        # Dekodlash bosqichlari
        d1 = self.d1(c3, b1)
        d2 = self.d2(c2, d1)
        d3 = self.d3(c1, d2)

        # Yakuniy chiqish
        output = self.aspp(d3)
        output = self.output(output)

        return output


if __name__ == "__main__":
    model = build_resunetplusplus()

    # Model murakkabligi va parametrlari
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), as_strings=True,
                                              print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
