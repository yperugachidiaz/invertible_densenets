import torch
import torch.nn.functional as F

class LipschitzDenseLayer(torch.nn.Module):
    def __init__(self, network, learnable_concat=False, lip_coeff=0.98):
        super(LipschitzDenseLayer, self).__init__()
        self.network = network
        self.lip_coeff = lip_coeff

        if learnable_concat:
            self.K1_unnormalized = torch.nn.Parameter(torch.tensor([1.]))
            self.K2_unnormalized = torch.nn.Parameter(torch.tensor([1.]))
        else:
            self.register_buffer("K1_unnormalized", torch.tensor([1.]))
            self.register_buffer("K2_unnormalized", torch.tensor([1.]))

    def get_eta1_eta2(self, beta=0.1):
        eta1 = F.softplus(self.K1_unnormalized) + beta
        eta2 = F.softplus(self.K2_unnormalized) + beta
        divider = torch.sqrt(eta1 ** 2 + eta2 ** 2)

        eta1_normalized = (eta1/divider) * self.lip_coeff
        eta2_normalized = (eta2/divider) * self.lip_coeff
        return eta1_normalized, eta2_normalized

    def forward(self, x):
        out = self.network(x)
        eta1_normalized, eta2_normalized = self.get_eta1_eta2()
        return torch.cat([x * eta1_normalized, out * eta2_normalized], dim=1)
