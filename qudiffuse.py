import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

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


def _norm(ch: int) -> nn.Module:
    """Simple GroupNorm used for residual blocks."""
    return nn.GroupNorm(8, ch)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = _norm(in_channels)
        self.norm2 = _norm(out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        b, c, h_, w_ = q.shape
        q = q.reshape(b, c, h_ * w_).permute(0, 2, 1)
        k = k.reshape(b, c, h_ * w_)
        w = torch.bmm(q, k) * (c ** -0.5)
        w = torch.softmax(w, dim=-1)
        v = v.reshape(b, c, h_ * w_)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w).reshape(b, c, h_, w_)
        h = self.proj(h)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class BinaryAutoencoder(nn.Module):
    """Improved binary autoencoder with residual and attention blocks."""

    def __init__(self, input_channels: int, latent_dim: int, base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        ch = base_channels
        self.final_channels = ch * 8

        enc_layers = [
            nn.Conv2d(input_channels, ch, kernel_size=4, stride=2, padding=1),
            ResBlock(ch),
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1),
            ResBlock(ch * 2),
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1),
            ResBlock(ch * 4),
            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1),
            ResBlock(ch * 8),
            AttnBlock(ch * 8),
        ]
        self.encoder = nn.Sequential(*enc_layers)

        self.enc_fc = nn.Linear(self.final_channels * 2 * 2, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, self.final_channels * 2 * 2)

        dec_layers = [
            ResBlock(ch * 8),
            AttnBlock(ch * 8),
            Upsample(ch * 8),
            ResBlock(ch * 8, ch * 4),
            Upsample(ch * 4),
            ResBlock(ch * 4, ch * 2),
            Upsample(ch * 2),
            ResBlock(ch * 2, ch),
            Upsample(ch),
            nn.Conv2d(ch, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ]
        self.decoder = nn.Sequential(*dec_layers)

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
        h = h.view(h.size(0), self.final_channels, 2, 2)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x: torch.Tensor, stochastic: bool = True):
        h = self.encode(x)
        z = self.binarize(h, stochastic=stochastic)
        x_recon = self.decode(z)
        return x_recon, z, h


class HierarchicalBinaryAutoencoder(nn.Module):
    """Stacked binary autoencoder producing hierarchical latent codes."""

    def __init__(self, input_channels: int, latent_dims: List[int], base_channels: int = 64):
        super().__init__()
        assert len(latent_dims) >= 2, "Need at least two latent levels"
        self.levels = len(latent_dims)
        self.base = BinaryAutoencoder(input_channels, latent_dims[0], base_channels)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(1, self.levels):
            self.encoders.append(nn.Linear(latent_dims[i-1], latent_dims[i]))
        for i in range(self.levels - 1, 0, -1):
            self.decoders.append(nn.Linear(latent_dims[i], latent_dims[i-1]))

    def encode_levels(self, x: torch.Tensor) -> List[torch.Tensor]:
        logits = [self.base.encode(x)]
        h = logits[0]
        for enc in self.encoders:
            h = enc(stochastic_binarize(h))
            logits.append(h)
        return logits

    def decode_levels(self, codes: List[torch.Tensor]) -> torch.Tensor:
        h = codes[-1]
        for dec, skip in zip(self.decoders, reversed(codes[:-1])):
            h = dec(h) + skip
        x = self.base.decode(h)
        return x

    def forward(self, x: torch.Tensor, stochastic: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.encode_levels(x)
        bin_codes = [stochastic_binarize(l) if stochastic else StraightThroughBinarization.apply(l) for l in logits]
        x_recon = self.decode_levels(bin_codes)
        flat_z = torch.cat([b.view(b.size(0), -1) for b in bin_codes], dim=1)
        flat_logits = torch.cat([l.view(l.size(0), -1) for l in logits], dim=1)
        return x_recon, flat_z, flat_logits


def autoencoder_loss(x: torch.Tensor, x_recon: torch.Tensor, logits: torch.Tensor,
                     prior_prob: float = 0.5, kl_weight: float = 1e-2,
                     ent_weight: float = 1e-3) -> torch.Tensor:
    recon_loss = F.mse_loss(x_recon, x)
    q_prob = torch.sigmoid(logits)
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

    def contrastive_divergence(self, v: torch.Tensor, k: int = 1, lr: float = 1e-3, orth_lambda: float = 0.0):
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
        if orth_lambda > 0:
            gram = self.W.t() @ self.W
            grad = (gram - torch.eye(self.n_hidden, device=self.W.device)) @ self.W.t()
            self.W.data -= lr * orth_lambda * grad.t() / v0.size(0)
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

    def pretrain(self, data: torch.Tensor, epochs: int = 10, lr: float = 1e-3, k: int = 1, batch_size: int = 64, orth_lambda: float = 0.0):
        """Layer-wise contrastive divergence pretraining with optional orthogonality regularization."""
        v = data
        for idx, rbm in enumerate(self.rbms):
            dataset = torch.utils.data.TensorDataset(v)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for epoch in range(epochs):
                activations = []
                for batch in loader:
                    v_batch = batch[0]
                    rbm.contrastive_divergence(v_batch, k=k, lr=lr, orth_lambda=orth_lambda)
                    activations.append(rbm.v_to_h(v_batch))
                if isinstance(self, SHDBN) and self.gates is not None:
                    mean_act = torch.stack([a.mean(0) for a in activations]).mean(0)
                    target = mean_act.mean()
                    self.gates.data[idx] -= lr * (mean_act.mean() - target)
                print(f"RBM {idx} epoch {epoch} done")
            with torch.no_grad():
                v = rbm(v)

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

class SHDBN(DBN):
    """Structured Hierarchical DBN with optional skip connections and layer gates."""

    def __init__(self, layer_sizes: List[int], use_skip: bool = True,
                 use_gates: bool = True, orth_reg: float = 0.0):
        super().__init__(layer_sizes)
        self.use_skip = use_skip
        self.orth_reg = orth_reg
        self.gates = nn.Parameter(torch.ones(len(self.rbms))) if use_gates else None

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        h = v
        prev = v
        for idx, rbm in enumerate(self.rbms):
            h = rbm(h)
            if self.gates is not None:
                h = torch.sigmoid(self.gates[idx]) * h
            if self.use_skip and idx > 0:
                h = h + prev
            prev = h
        return h

    def regularization_loss(self) -> torch.Tensor:
        if self.orth_reg == 0.0:
            return torch.tensor(0.0, device=self.rbms[0].W.device)
        loss = 0.0
        for rbm in self.rbms:
            W = rbm.W
            gram = W.t() @ W
            loss = loss + ((gram - torch.eye(gram.size(0), device=W.device)) ** 2).mean()
        return self.orth_reg * loss


class TimeConditionedDBN(DBN):
    """DBN with biases and couplings conditioned on timestep."""

    def __init__(self, layer_sizes: List[int], timesteps: int):
        super().__init__(layer_sizes)
        self.timesteps = timesteps
        self.bias_table = nn.Parameter(torch.zeros(timesteps, layer_sizes[-1]), requires_grad=False)
        self.cov_table = nn.Parameter(torch.zeros(timesteps, layer_sizes[-1], layer_sizes[-1]), requires_grad=False)

    def estimate_time_conditioned(self, base_codes: torch.Tensor, diffusion: 'BinaryDiffusion'):
        for t in range(self.timesteps):
            z_t = diffusion.q_sample(base_codes, t)
            b, c = super().estimate_prior(z_t)
            self.bias_table.data[t] = b
            self.cov_table.data[t] = c

    def get_prior(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = min(max(t, 0), self.timesteps - 1)
        return self.bias_table[t], self.cov_table[t]

class BinaryDiffusion:
    def __init__(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.05):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)

    def q_sample(self, z0: torch.Tensor, t: int) -> torch.Tensor:
        beta = self.betas[t]
        flip_mask = torch.rand_like(z0) < beta
        return torch.where(flip_mask, 1 - z0, z0)

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

def qubo_energy(x: np.ndarray, Q: np.ndarray, bias: np.ndarray) -> float:
    return float(x @ Q @ x + bias @ x)

def tabu_search(initial: np.ndarray, Q: np.ndarray, bias: np.ndarray, iters: int = 100) -> np.ndarray:
    x = initial.copy()
    best = x.copy()
    best_e = qubo_energy(x, Q, bias)
    tabu: set = set()
    n = len(x)
    for _ in range(iters):
        e_candidates = []
        for i in range(n):
            cand = x.copy()
            cand[i] = 1 - cand[i]
            key = tuple(cand.tolist())
            if key in tabu:
                continue
            e = qubo_energy(cand, Q, bias)
            e_candidates.append((e, i, cand))
        if not e_candidates:
            break
        e_candidates.sort(key=lambda t: t[0])
        e, idx, cand = e_candidates[0]
        x = cand
        tabu.add(tuple(x.tolist()))
        if e < best_e:
            best_e = e
            best = x.copy()
        if len(tabu) > 10 * n:
            tabu.clear()
    return best

def solve_qubo(Q: np.ndarray, bias: np.ndarray, num_reads: int = 100, tabu_iters: int = 100,
               sub_size: int = 2048, use_qpu: bool = False, chain_strength: float | None = None) -> np.ndarray:
    if dimod is None or SimulatedAnnealingSampler is None:
        raise ImportError('dimod and dwave-neal must be installed')
    n = len(bias)
    if n > sub_size:
        return solve_qubo_partitioned(Q, bias, num_reads, tabu_iters, sub_size, use_qpu=use_qpu, chain_strength=chain_strength)
    bqm = dimod.BinaryQuadraticModel({i: float(bias[i]) for i in range(n)},
                                     {(i, j): float(Q[i, j]) for i in range(n) for j in range(i + 1, n) if Q[i, j] != 0},
                                     0.0,
                                     vartype=dimod.BINARY)
    if use_qpu:
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
        except Exception as e:
            raise ImportError('D-Wave system libraries required for QPU sampling') from e
        sampler = EmbeddingComposite(DWaveSampler())
        if chain_strength is None:
            chain_strength = 2 * np.max(np.abs(Q))
        sampleset = sampler.sample(bqm, num_reads=num_reads, chain_strength=chain_strength, chain_break_method='majority_vote')
        sample = sampleset.first.sample
    else:
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=num_reads)
        sample = sampleset.first.sample
    result = np.array([sample[i] for i in range(n)], dtype=np.float32)
    result = tabu_search(result, Q, bias, iters=tabu_iters)
    return result

def solve_qubo_partitioned(Q: np.ndarray, bias: np.ndarray, num_reads: int, tabu_iters: int,
                           sub_size: int, use_qpu: bool = False, chain_strength: float | None = None) -> np.ndarray:
    n = len(bias)
    solution = np.random.randint(0, 2, n).astype(np.float32)
    energy = qubo_energy(solution, Q, bias)
    for _ in range(10):
        idx = np.random.choice(n, size=min(sub_size, n), replace=False)
        others = np.setdiff1d(np.arange(n), idx)
        Q_sub = Q[np.ix_(idx, idx)]
        b_sub = bias[idx] + Q[np.ix_(idx, others)] @ solution[others]
        sub_sol = solve_qubo(Q_sub, b_sub, num_reads=num_reads, tabu_iters=tabu_iters,
                             use_qpu=use_qpu, chain_strength=chain_strength)
        candidate = solution.copy()
        candidate[idx] = sub_sol
        e = qubo_energy(candidate, Q, bias)
        if e < energy:
            solution = candidate
            energy = e
    return solution


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

def generate(autoencoder: nn.Module, dbn: DBN, diffusion: BinaryDiffusion,
             num_samples: int = 1, window: int = 5, num_reads: int = 100,
             use_qpu: bool = False) -> torch.Tensor:
    device = next(autoencoder.parameters()).device
    latent_dim = getattr(autoencoder, 'latent_dim', None)
    if latent_dim is None and hasattr(autoencoder, 'latent_dims'):
        latent_dim = sum(autoencoder.latent_dims)
    z_t = torch.bernoulli(torch.full((num_samples, latent_dim), 0.5)).to(device)
    for step in reversed(range(0, diffusion.timesteps, window)):
        with torch.no_grad():
            if isinstance(dbn, TimeConditionedDBN):
                dbn_bias, dbn_coupling = dbn.get_prior(step)
            else:
                dbn_bias, dbn_coupling = dbn.estimate_prior(z_t)
        Q, bias = build_joint_qubo(z_t[0], dbn_bias, dbn_coupling, diffusion, min(window, step + 1))
        z_block = solve_qubo(Q, bias, num_reads=num_reads, use_qpu=use_qpu)
        z_block = torch.from_numpy(z_block).view(-1, latent_dim)
        z_t = z_block[0:1]
    with torch.no_grad():
        if hasattr(autoencoder, 'decode_levels'):
            segments = []
            start = 0
            for ld in autoencoder.latent_dims:
                segments.append(z_t[:, start:start + ld])
                start += ld
            images = autoencoder.decode_levels(segments)
        else:
            images = autoencoder.decode(z_t.to(device))
    return images

def evaluate_generation(autoencoder: nn.Module, dbn: DBN, diffusion: BinaryDiffusion,
                        real_loader, num_samples: int = 1000, window: int = 5,
                        device: str = 'cuda', use_qpu: bool = False):
    fid = FrechetInceptionDistance(feature=64).to(device)
    isc = InceptionScore().to(device)
    count = 0
    for img, in real_loader:
        img = img.to(device)
        fid.update(img * 255.0, real=True)
        count += img.size(0)
        if count >= num_samples:
            break
    with torch.no_grad():
        gen = generate(autoencoder, dbn, diffusion, num_samples=num_samples, window=window, use_qpu=use_qpu).to(device)
    fid.update(gen * 255.0, real=False)
    isc.update(gen * 255.0)
    fid_score = fid.compute().item()
    is_score = isc.compute()[0].item()
    return fid_score, is_score

def train_autoencoder(autoencoder: nn.Module, dataloader, epochs: int = 10,
                      lr: float = 1e-3, device: str = 'cuda'):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    autoencoder.to(device)
    for epoch in range(epochs):
        for x, in dataloader:
            x = x.to(device)
            x_recon, z, logits = autoencoder(x)
            loss = autoencoder_loss(x, x_recon, logits)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f'Epoch {epoch} AE loss {loss.item():.4f}')

def train_hierarchical_autoencoder(autoencoder: HierarchicalBinaryAutoencoder, dataloader,
                                   epochs: int = 10, lr: float = 1e-3, device: str = 'cuda'):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    autoencoder.to(device)
    for epoch in range(epochs):
        for x, in dataloader:
            x = x.to(device)
            x_recon, z, logits = autoencoder(x)
            loss = autoencoder_loss(x, x_recon, logits)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f'Hier AE epoch {epoch} loss {loss.item():.4f}')


def train_dbn(dbn: DBN, autoencoder: nn.Module, dataloader, epochs: int = 10,
              lr: float = 1e-3, k: int = 1, device: str = 'cuda',
              joint_window: int = 0, diffusion: BinaryDiffusion | None = None):
    autoencoder.to(device)
    autoencoder.eval()
    codes = []
    with torch.no_grad():
        for x, in dataloader:
            x = x.to(device)
            h = autoencoder.encode(x)
            if joint_window and diffusion is not None:
                z0 = stochastic_binarize(h)
                seq = diffusion.q_sample_progressive(z0)[:joint_window]
                flat = torch.cat(seq, dim=1)
                codes.append(flat)
            else:
                codes.append(torch.sigmoid(h))
    data = torch.cat(codes, dim=0)
    orth = dbn.orth_reg if isinstance(dbn, SHDBN) else 0.0
    dbn.pretrain(data, epochs=epochs, lr=lr, k=k, orth_lambda=orth)
    if isinstance(dbn, TimeConditionedDBN) and diffusion is not None:
        dbn.estimate_time_conditioned(data, diffusion)

if __name__ == '__main__':
    import argparse
    from torchvision import datasets, transforms
    parser = argparse.ArgumentParser(description='QuDiffuse Training and Generation')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','cifar10','celeba'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--prior', type=str, default='dbn', choices=['dbn','shdbn','tcdbn'])
    parser.add_argument('--hier', action='store_true')
    parser.add_argument('--joint-train', type=int, default=0, help='window size for joint training')
    parser.add_argument('--use-qpu', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        input_channels = 1
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        input_channels = 3
    elif args.dataset == 'celeba':
        transform = transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()])
        train_set = datasets.CelebA('./data', split='train', download=True, transform=transform)
        input_channels = 3
    else:
        raise ValueError('Unknown dataset')

    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    if args.hier:
        autoencoder = HierarchicalBinaryAutoencoder(input_channels, [512, 512])
        if not args.generate:
            train_hierarchical_autoencoder(autoencoder, loader, epochs=args.epochs, device=device)
    else:
        autoencoder = BinaryAutoencoder(input_channels, latent_dim=1024)
        if not args.generate:
            train_autoencoder(autoencoder, loader, epochs=args.epochs, device=device)
    if args.prior == 'dbn':
        dbn = DBN([sum(getattr(autoencoder, 'latent_dims', [autoencoder.latent_dim])), 2048, 2048])
    elif args.prior == 'tcdbn':
        dbn = TimeConditionedDBN([sum(getattr(autoencoder, 'latent_dims', [autoencoder.latent_dim])), 2048], args.timesteps)
    else:
        dbn = SHDBN([sum(getattr(autoencoder, 'latent_dims', [autoencoder.latent_dim])), 2048, 2048, sum(getattr(autoencoder, 'latent_dims', [autoencoder.latent_dim]))], orth_reg=1e-3)
    if not args.generate:
        train_dbn(dbn, autoencoder, loader, epochs=args.epochs, device=device,
                  joint_window=args.joint_train, diffusion=BinaryDiffusion(args.timesteps))

    diffusion = BinaryDiffusion(args.timesteps)

    if args.generate:
        images = generate(autoencoder, dbn, diffusion, num_samples=1, window=args.window, use_qpu=args.use_qpu)
        from torchvision.utils import save_image
        save_image(images.cpu(), 'generated.png')
        print('Saved generated image to generated.png')
        if args.evaluate:
            fid, isc = evaluate_generation(autoencoder, dbn, diffusion, loader, num_samples=1000, window=args.window, device=device, use_qpu=args.use_qpu)
            print(f'FID {fid:.3f} IS {isc:.3f}')
