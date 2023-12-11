import torch
from codebase import utils as ut
from torch import nn
from torch.nn import functional as F

#########################################################################################
class Encoder(nn.Module):
    def __init__(self, z_dim, y_dim=0, num_channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        hidden_dims = [32, 64, 128]
        kernels = 5
        self.pixels = 128
        modules = []
        for i,h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(num_channels, out_channels=h_dim,
                              kernel_size= kernels, stride= 2, padding  = 2),
                    nn.LeakyReLU())
            )
            num_channels = h_dim

        self.conv_layers = nn.Sequential(*modules)

        # Calculate the size after the convolutional layers
        self.conv_output_size = self.pixels * 16 * 16

        self.fc_layers = nn.Linear(self.conv_output_size + y_dim, 2*z_dim)



    def forward(self, x, y=None):
        x = x.view(-1, 3, self.pixels, self.pixels)  # Reshape to [batch_size, 3, 128, 128]
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output before concatenating with y
        x = F.dropout(x, p=0.2, training=self.training)
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.fc_layers(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0, num_channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.y_dim = y_dim
        self.pixels = 128
        hidden_dims = [128, 64, 32]
        kernels = 5
        modules = []
        self.fc_layers = nn.Linear(z_dim + y_dim, self.pixels * 16 *16)


        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=kernels,
                                       stride = 2,
                                       padding=2,
                                       output_padding=1),
                    nn.LeakyReLU())
            )


        self.deconv_layers = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=5,
                                               stride=2,
                                               padding=2,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())
        
        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(hidden_dims[-1],
        #                                        out_channels = 3,
        #                                        kernel_size=kernels,
        #                                        stride=2,
        #                                        padding=3,
        #                                        output_padding=1),
        #                     nn.Sigmoid())

                             

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        x = self.fc_layers(zy)
        x = x.view(x.size(0), self.pixels, 16, 16)  # Reshape to match conv transpose input dimensions
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


class Classifier(nn.Module):
    def __init__(self, y_dim, num_channels=3, hidden_dims=[32, 64, 128], pixels=128):
        super(Classifier, self).__init__()
        self.y_dim = y_dim
        self.num_channels = num_channels
        self.pixels = pixels

        # Convolutional layers
        modules = []
        for i, h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(num_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            num_channels = h_dim

        self.conv_layers = nn.Sequential(*modules)

        # Calculate the size after the convolutional layers
        conv_output_size = pixels // (2 ** len(hidden_dims)) * pixels // (2 ** len(hidden_dims))

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, y_dim)
        )

    def forward(self, x):
        x = x.view(-1, 3, self.pixels, self.pixels)  # Reshape to [batch_size, 3, 128, 128]
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output before feeding into fully connected layers
        y = self.fc_layers(x)
        return y
