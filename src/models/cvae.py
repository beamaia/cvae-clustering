from torch import nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),  # (32)
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),  # (16)
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),  # (8)
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),  # (4)
                nn.Flatten(),
            )
        self.output_dim = 4 * 4 * 256        

    def forward(self, x):
        features = self.encoder(x)
        return features

class Decoder(nn.Module):
    pass


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__(self)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def encode(self, x):
        pass


    def decode(self, z):
        pass


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar