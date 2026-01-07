import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from .component.encoder import Encoder


class SpatialSoftmax(nn.Module):
    def __init__(self, init_temp=1.0, output_normalized=True):
        super(SpatialSoftmax, self).__init__()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temp)))
        self.output_normalized = output_normalized

        # Meshgrid
        # pos_x, pos_y = torch.meshgrid(
        #     torch.linspace(-1, 1, height),
        #     torch.linspace(-1, 1, width),
        #     indexing='ij'
        # )

        # self.register_buffer('pos_x', pos_x.reshape(-1))
        # self.register_buffer('pos_y', pos_y.reshape(-1))

    def forward(self, feature_map):
        # feature_map shape: [Batch, Channels, Height, Width]

        n, c, h, w = feature_map.shape

        if self.output_normalized:
            y_t = torch.linspace(
                -1.0, 1.0, h, device=feature_map.device, dtype=feature_map.dtype
            )
            x_t = torch.linspace(
                -1.0, 1.0, w, device=feature_map.device, dtype=feature_map.dtype
            )
        else:
            y_t = torch.linspace(
                0, h - 1, h, device=feature_map.device, dtype=feature_map.dtype
            )
            x_t = torch.linspace(
                0, w - 1, w, device=feature_map.device, dtype=feature_map.dtype
            )

        pos_y, pos_x = torch.meshgrid(y_t, x_t, indexing="ij")
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        flat_map = feature_map.view(n, c, -1)

        temperature = torch.exp(self.log_temperature)
        flat_map = flat_map / temperature

        softmax_attention = F.softmax(flat_map, dim=2)

        # Expected X = sum(P_i * x_i)
        expected_x = torch.sum(pos_x * softmax_attention, dim=2, keepdim=True)
        expected_y = torch.sum(pos_y * softmax_attention, dim=2, keepdim=True)

        expected_coords = torch.cat([expected_x, expected_y], dim=2).reshape(n, -1)
        return expected_coords


class MobileNetAimBot(nn.Module):
    def __init__(self, defaultweight=True, en_weight=None):
        super().__init__()
        use_imagenet = defaultweight if en_weight is None else False
        self.encoder = Encoder(pretrained=use_imagenet, frozen=False)
        if en_weight is not None:
            self.load_pretrained_encoder(en_weight)

        in_channels = self.encoder.out_channels
        hidden_dim1 = 256
        hidden_dim2 = 64

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim1),
            nn.Hardswish(),
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim2),
            nn.Hardswish(),
            nn.Conv2d(hidden_dim2, 1, kernel_size=1),
        )

        self.spatial_softmax = SpatialSoftmax()

    def load_pretrained_encoder(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if "model_state_dict" in checkpoint:
                full_state_dict = checkpoint["model_state_dict"]
            else:
                full_state_dict = checkpoint

            # load encoder depend on diffrent situation
            encoder_dict = {}
            for k, v in full_state_dict.items():
                if k.startswith("encoder."):
                    new_key = k.replace("encoder.", "")
                    encoder_dict[new_key] = v

                elif k.startswith("features."):
                    encoder_dict[k] = v

            if len(encoder_dict) > 0:
                # strict=True check
                self.encoder.load_state_dict(encoder_dict, strict=True)
                print("DAE Encoder load sucessfully")
            else:
                print("can't find dae weights")

        except Exception as e:
            print(f"falied: {e}")
            print("use random weight continue")

    def forward(self, x):
        feature = self.encoder(x)
        x = self.head(feature)
        coords = self.spatial_softmax(x)
        return coords


if __name__ == "__main__":
    model = MobileNetAimBot()
    dummy_input = torch.randn(1, 3, 256, 256)
    summary(model, input_size=(1, 3, 320, 320), device="cpu")
    output = model(dummy_input)
    output = output.squeeze()
    print(output)
