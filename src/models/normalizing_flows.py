import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarFlow(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        self.n_features = z_dim
        self.weights = nn.Parameter(torch.randn(1, z_dim).normal_(0, 0.01))
        self.bias = nn.Parameter(torch.zeros(1).normal_(0, 0.01))
        self.u = nn.Parameter(torch.randn(1, z_dim).normal_(0, 0.01))
    
    def forward(self, z):
        # https://arxiv.org/pdf/1505.05770.pdf
        # https://stats.stackexchange.com/questions/465184/planar-flow-in-normalizing-flows
        u_temp = (self.weights @ self.u.t()).squeeze()
        m_u_temp = -1 + F.softplus(u_temp)
        uhat = self.u + (m_u_temp - u_temp) * (self.weights / (self.weights @ self.weights.t()))
        z_temp = z @ self.weights.t() + self.bias
        new_z = z + uhat * torch.tanh(z_temp)

        psi = (1 - torch.tanh(z_temp)**2) @ self.weights
        det_jac = 1 + psi @ uhat.t()
        logdet_jacobian = torch.log(torch.abs(det_jac) + 1e-8).squeeze()

        return new_z, logdet_jacobian


class NormalizingFlows(nn.Module):
    def __init__(self, z_dim, n_flows=4, flow_type=PlanarFlow):
        super().__init__()
        self.z_dim = z_dim
        self.n_flows = n_flows
        self.flow_type = flow_type

        self.flows = nn.ModuleList([
            self.flow_type(self.z_dim) for _ in range(self.n_flows)
        ])
    
    def forward(self, z):
        
        logdet_jacobians = []
        
        for flow in self.flows:
            z, logdet_j = flow(z)
            logdet_jacobians.append(logdet_j)
        
        z_k = z
        logdet_jacobians = torch.stack(logdet_jacobians, dim=1)

        return z_k, torch.sum(logdet_jacobians, dim=1)
