import torch
import torch.nn as nn

import utils


class MultiMagnification_Local_Affine_Network(nn.Module):
    def __init__(self, device):
        super(MultiMagnification_Local_Affine_Network, self).__init__()
        self.device = device

        self.encoder_1 = Conv_Block(in_channels=2, out_channels=32)
        self.encoder_2 = Conv_Block(in_channels=32, out_channels=64)
        self.encoder_3 = Conv_Block(in_channels=64, out_channels=128)
        self.encoder_4 = Conv_Block(in_channels=128, out_channels=256)

        self.max_pool = nn.MaxPool2d(2)

        self.conv_block_1 = Conv_Block(in_channels=128+192, out_channels=128+192)
        self.conv_block_2 = Conv_Block(in_channels=64+192, out_channels=64+192)
        self.conv_block_3 = Conv_Block(in_channels=32+144, out_channels=128-16)

        self.decoder_1 = Conv_Transpose(in_channels=256+128)
        self.decoder_2 = Conv_Transpose(in_channels=128+192+64)
        self.decoder_3 = Conv_Transpose(in_channels=64+192+32)

        self.magnification_layer = Magnification_Layer(cropping_rate=2, device=device)

        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regression_network = Regression_Network()

    def forward(self, source, target):
        # source: (n_samples, 2, 256, 256)
        # target: (n_samples, 2, 256, 256)

        source_level_1 = torch.unsqueeze(source[:, 0, :, :], dim=1)  # (n_samples, 1, 256, 256)
        source_level_2 = torch.unsqueeze(source[:, 1, :, :], dim=1)
        target_level_1 = torch.unsqueeze(target[:, 0, :, :], dim=1)
        target_level_2 = torch.unsqueeze(target[:, 1, :, :], dim=1)

        x_1 = torch.cat((source_level_1, target_level_1), dim=1)  # (n_samples, 2, 256, 256)
        x_2 = torch.cat((source_level_2, target_level_2), dim=1)

        # feature maps from the original image pairs
        x_1_1 = self.encoder_1(x_1)
        x_1_2 = self.encoder_2(self.max_pool(x_1_1))
        x_1_3 = self.encoder_3(self.max_pool(x_1_2))
        x_1_4 = self.encoder_4(self.max_pool(x_1_3))

        # feature maps from the first magnification layer
        d_2_1 = self.magnification_layer.feature_map_3(x_2)
        d_2_2 = self.magnification_layer.feature_map_2(x_2)
        d_2_3 = self.magnification_layer.feature_map_1(x_2)
        d_2_4 = self.magnification_layer(x_2)

        d_1_4 = self.decoder_1(torch.cat((x_1_4, d_2_4), dim=1))
        d_1_4 = self.conv_block_1(torch.cat((x_1_3, d_1_4), dim=1))

        d_1_3 = self.decoder_2(torch.cat((d_1_4, d_2_3), dim=1))
        d_1_3 = self.conv_block_2(torch.cat((x_1_2, d_1_3), dim=1))

        d_1_2 = self.decoder_3(torch.cat((d_1_3, d_2_2), dim=1))
        d_1_2 = self.conv_block_3(torch.cat((x_1_1, d_1_2), dim=1))

        output = self.final_conv(torch.cat((d_1_2, d_2_1), dim=1))
        output = output.view(output.size(0), -1)
        output = self.regression_network(output)

        return output


class Magnification_Layer(nn.Module):
    def __init__(self, cropping_rate, device):
        super(Magnification_Layer, self).__init__()
        self.device = device

        self.cropping_rate = cropping_rate
        self.num_tr = cropping_rate // 2

        self.encoder_1 = Conv_Block(2, 32)
        self.encoder_2 = Conv_Block(32, 64)
        self.encoder_3 = Conv_Block(64, 128)
        self.encoder_4 = Conv_Block(128, 256)

        self.conv_block_1 = Conv_Block(in_channels=128, out_channels=128)
        self.conv_block_2 = Conv_Block(in_channels=64, out_channels=64)
        self.conv_block_3 = Conv_Block(in_channels=32, out_channels=32)

        self.decoder_1 = nn.Sequential()
        for i in range(self.num_tr):
            self.decoder_1.add_module(f'conv_tr_{i + 1}', Conv_Transpose(256 // (i + 1)))
        self.decoder_2 = nn.Sequential()
        for i in range(self.num_tr):
            self.decoder_2.add_module(f'conv_tr_{i + 1}', Conv_Transpose(128 // (i + 1)))
        self.decoder_3 = nn.Sequential()
        for i in range(self.num_tr):
            self.decoder_3.add_module(f'conv_tr_{i + 1}', Conv_Transpose(64 // (i + 1)))
        self.decoder_4 = nn.Sequential()
        for i in range(self.num_tr):
            self.decoder_4.add_module(f'conv_tr_{i + 1}', Conv_Transpose(32 // (i + 1)))

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        # down-sample
        x = self.max_pool(self.encoder_1(x))
        x = self.max_pool(self.encoder_2(x))
        x = self.max_pool(self.encoder_3(x))
        x = self.encoder_4(x)

        # center-crop the feature map
        x = utils.center_crop_tensor(x, cropping_rate=self.cropping_rate, device=self.device)

        # up-sample
        x = self.decoder_1(x)
        return x

    def feature_map_1(self, x):
        x = self.max_pool(self.encoder_1(x))
        x = self.max_pool(self.encoder_2(x))
        x = self.encoder_3(x)
        x = utils.center_crop_tensor(x, cropping_rate=self.cropping_rate, device=self.device)
        x = self.decoder_2(self.conv_block_1(x))
        return x

    def feature_map_2(self, x):
        x = self.max_pool(self.encoder_1(x))
        x = self.encoder_2(x)
        x = utils.center_crop_tensor(x, cropping_rate=self.cropping_rate, device=self.device)
        x = self.decoder_3(self.conv_block_2(x))
        return x

    def feature_map_3(self, x):
        x = self.encoder_1(x)
        x = utils.center_crop_tensor(x, cropping_rate=self.cropping_rate, device=self.device)
        x = self.decoder_4(self.conv_block_3(x))
        return x


class Conv_Block(nn.Module):
    """
    Two sets of a convolutional layer with kernel size of 3 × 3 with padding of 1
    followed by a rectified linear unit (ReLU) activation function in series
    """
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels, out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels, out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.conv_block(x)


class Conv_Transpose(nn.Module):
    """
    A transposed convolutional layer with the up-sampling rate of 2 followed by the ReLU activation function
    """
    def __init__(self, in_channels):
        super(Conv_Transpose, self).__init__()

        self.conv_tr = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2),
            nn.GroupNorm(in_channels//2, in_channels//2),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.conv_tr(x)


class Regression_Network(nn.Module):
    def __init__(self):
        super(Regression_Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 6),
        )

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, 2, 3)


def load_network(device, path=None):
    model = MultiMagnification_Local_Affine_Network(device=device)
    model = model.to(device)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
    return model


def test_forward_pass():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_network(device=device)

    y_size = 256
    x_size = 256
    no_channels = 2
    batch_size = 3
    example_source = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)
    example_target = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)

    result = model(example_source, example_target)

    print(result.size())


if __name__ == "__main__":
    test_forward_pass()