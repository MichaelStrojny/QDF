import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

try:
    import dimod
    from dwave.neal import SimulatedAnnealingSampler
except ImportError:  # Optional dependency
    dimod = None
    SimulatedAnnealingSampler = None

class StraightThroughBinarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return (inputs >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def stochastic_binarize(x: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(x)
    rand = torch.rand_like(prob)
    return (rand < prob).float()

class BinaryAutoencoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(512 * 2 * 2, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h = self.enc_fc(h)
        return h

    def binarize(self, h: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        if stochastic:
            return stochastic_binarize(h)
        return StraightThroughBinarization.apply(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 512, 2, 2)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        h = self.encode(x)
        z = self.binarize(h, stochastic=stochastic)
        x_recon = self.decode(z)
        return x_recon, z


def autoencoder_loss(x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor,
                     prior_prob: float = 0.5, kl_weight: float = 1e-2,
                     ent_weight: float = 1e-3) -> torch.Tensor:
    recon_loss = F.mse_loss(x_recon, x)
    q_prob = torch.sigmoid(z)
    prior = torch.full_like(q_prob, prior_prob)
    kl = F.kl_div(q_prob.log(), prior, reduction="batchmean")
    entropy = -(q_prob * q_prob.log() + (1 - q_prob) * (1 - q_prob).log()).mean()
    return recon_loss + kl_weight * kl + ent_weight * entropy

class RBM(nn.Module):
    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_from_p(self, p: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(p)

    def v_to_h(self, v: torch.Tensor) -> torch.Tensor:
        p_h = torch.sigmoid(F.linear(v, self.W.t(), self.h_bias))
        return p_h

    def h_to_v(self, h: torch.Tensor) -> torch.Tensor:
        p_v = torch.sigmoid(F.linear(h, self.W, self.v_bias))
        return p_v

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = F.linear(v, self.W.t(), self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return (-vbias_term - hidden_term).mean()

    def contrastive_divergence(self, v: torch.Tensor, k: int = 1, lr: float = 1e-3):
        v0 = v
        ph0 = self.v_to_h(v0)
        h0 = self.sample_from_p(ph0)
        v_k = v0
        h_k = h0
        for _ in range(k):
            pv = self.h_to_v(h_k)
            v_k = self.sample_from_p(pv)
            ph = self.v_to_h(v_k)
            h_k = self.sample_from_p(ph)
        self.W.data += lr * (torch.matmul(v0.t(), ph0) - torch.matmul(v_k.t(), ph)) / v0.size(0)
        self.v_bias.data += lr * torch.mean(v0 - v_k, dim=0)
        self.h_bias.data += lr * torch.mean(ph0 - ph, dim=0)
        return self.free_energy(v0) - self.free_energy(v_k)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        ph = self.v_to_h(v)
        h = self.sample_from_p(ph)
        return h

class DBN(nn.Module):
    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        self.rbms = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.rbms.append(RBM(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        h = v
        for rbm in self.rbms:
            h = rbm(h)
        return h

    def pretrain(self, data: torch.Tensor, epochs: int = 10, lr: float = 1e-3, k: int = 1, batch_size: int = 64):
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        v = data
        for idx, rbm in enumerate(self.rbms):
            for epoch in range(epochs):
                for batch in loader:
                    v_batch = batch[0]
                    rbm.contrastive_divergence(v_batch, k=k, lr=lr)
                print(f"RBM {idx} epoch {epoch} done")
            v = rbm(v).detach()

    def estimate_prior(self, data: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            probs = data.float()
            h = probs
            for rbm in self.rbms:
                h = torch.sigmoid(F.linear(h, rbm.W.t(), rbm.h_bias))
            mean = h.mean(dim=0)
            cov = torch.matmul(h.t(), h) / h.size(0) - torch.ger(mean, mean)
            bias = -torch.log(mean / (1 - mean))
            return bias, cov

class BinaryDiffusion:
    def __init__(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.05):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)

    def q_sample(self, z0: torch.Tensor, t: int) -> torch.Tensor:
        beta = self.betas[t]
        flip_mask = torch.rand_like(z0) < beta
        zt = torch.where(flip_mask, 1 - z0, z0)
        return zt

    def q_sample_progressive(self, z0: torch.Tensor) -> List[torch.Tensor]:
        z = z0
        history = []
        for t in range(self.timesteps):
            z = self.q_sample(z, t)
            history.append(z)
        return history

    def q_posterior_bias(self, zt: torch.Tensor, t: int) -> torch.Tensor:
        beta = self.betas[t]
        log_ratio = torch.log((1 - beta) / beta)
        return log_ratio * (2 * zt - 1)

def solve_qubo(Q: np.ndarray, bias: np.ndarray, num_reads: int = 100) -> np.ndarray:
    if dimod is None or SimulatedAnnealingSampler is None:
        raise ImportError('dimod and dwave-neal must be installed')
    bqm = dimod.BinaryQuadraticModel({i: float(bias[i]) for i in range(len(bias))},
                                     {(i, j): float(Q[i, j]) for i in range(len(bias)) for j in range(i + 1, len(bias)) if Q[i, j] != 0},
                                     0.0,
                                     vartype=dimod.BINARY)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    sample = sampleset.first.sample
    result = np.array([sample[i] for i in range(len(bias))], dtype=np.float32)
    return result

def build_joint_qubo(z_t: torch.Tensor, dbn_bias: torch.Tensor, dbn_coupling: torch.Tensor,
                     diffusion: BinaryDiffusion, window: int) -> (np.ndarray, np.ndarray):
    n = z_t.numel()
    w = window
    Q = np.zeros((n * w, n * w), dtype=np.float32)
    bias = np.zeros(n * w, dtype=np.float32)
    for i in range(w):
        step = diffusion.timesteps - i - 1
        data_bias = diffusion.q_posterior_bias(z_t, step)
        bias[i * n:(i + 1) * n] += data_bias.detach().cpu().numpy() + dbn_bias.detach().cpu().numpy()
        Q_block = dbn_coupling.detach().cpu().numpy()
        Q[i * n:(i + 1) * n, i * n:(i + 1) * n] += Q_block
        if i > 0:
            gamma = -np.log(diffusion.betas[step + 1].item() / (1 - diffusion.betas[step + 1].item()))
            for j in range(n):
                idx1 = (i - 1) * n + j
                idx2 = i * n + j
                Q[idx1, idx2] += 2 * gamma
                bias[idx1] += -2 * gamma
                bias[idx2] += -2 * gamma
    return Q, bias

def generate(autoencoder: BinaryAutoencoder, dbn: DBN, diffusion: BinaryDiffusion,
             num_samples: int = 1, window: int = 5, num_reads: int = 100) -> torch.Tensor:
    device = next(autoencoder.parameters()).device
    z_t = torch.bernoulli(torch.full((num_samples, autoencoder.latent_dim), 0.5)).to(device)
    for step in reversed(range(0, diffusion.timesteps, window)):
        with torch.no_grad():
            dbn_bias, dbn_coupling = dbn.estimate_prior(z_t)
        Q, bias = build_joint_qubo(z_t[0], dbn_bias, dbn_coupling, diffusion, min(window, step + 1))
        z_block = solve_qubo(Q, bias, num_reads=num_reads)
        z_block = torch.from_numpy(z_block).view(-1, autoencoder.latent_dim)
        z_t = z_block[0:1]
    with torch.no_grad():
        images = autoencoder.decode(z_t.to(device))
    return images

def train_autoencoder(autoencoder: BinaryAutoencoder, dataloader, epochs: int = 10,
                      lr: float = 1e-3, device: str = 'cuda'):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    autoencoder.to(device)
    for epoch in range(epochs):
        for x, in dataloader:
            x = x.to(device)
            x_recon, z = autoencoder(x)
            loss = autoencoder_loss(x, x_recon, z)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f'Epoch {epoch} AE loss {loss.item():.4f}')


def train_dbn(dbn: DBN, dataloader, epochs: int = 10, lr: float = 1e-3, k: int = 1, device: str = 'cuda'):
    data = []
    for x, in dataloader:
        data.append(x.view(x.size(0), -1))
    data = torch.cat(data, dim=0).to(device)
    dbn.pretrain(data, epochs=epochs, lr=lr, k=k)

if __name__ == '__main__':
    import argparse
    from torchvision import datasets, transforms
    parser = argparse.ArgumentParser(description='QuDiffuse Training and Generation')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        input_channels = 1
    else:
        raise ValueError('Unknown dataset')

    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    autoencoder = BinaryAutoencoder(input_channels, latent_dim=1024)
    if not args.generate:
        train_autoencoder(autoencoder, loader, epochs=args.epochs, device=device)
    dbn = DBN([1024, 2048, 2048])
    if not args.generate:
        train_dbn(dbn, loader, epochs=args.epochs, device=device)

    diffusion = BinaryDiffusion(args.timesteps)

    if args.generate:
        images = generate(autoencoder, dbn, diffusion, num_samples=1, window=args.window)
        from torchvision.utils import save_image
        save_image(images.cpu(), 'generated.png')
        print('Saved generated image to generated.png')
