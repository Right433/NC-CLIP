
import torch
from train_utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import sys, os, math, json
from dataclasses import dataclass, asdict
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(".."); sys.path.append("../.."); sys.path.append("../../..")

from sharegpt4v import share4v_val_dataset, share4v_train_dataset
from model import longclip
from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.amp import GradScaler
from train_utils import LossManager, get_run_id, eval_coco
from gap_tracker import GapTracker
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================================
# 0. CosineTracker
# ============================================================================

COLOR_OURS = '#2166AC'
COLOR_BASE = '#878787'

@dataclass
class _CosRec:
    step: int; epoch: int
    pos_mean: float; pos_std: float
    neg_mean: float; neg_std: float
    gap: float

class CosineTracker:
    def __init__(self, save_dir, log_every=100, model_name="InvCLIP"):
        self.save_dir = save_dir; self.log_every = log_every
        self.model_name = model_name
        os.makedirs(save_dir, exist_ok=True)
        self.records_full: List[_CosRec] = []
        self.records_com:  List[_CosRec] = []
        self.json_path = os.path.join(save_dir, "cosine_tracker.json")

    @staticmethod
    def _stats(img_f, txt_f):
        with torch.no_grad():
            B = img_f.shape[0]
            sim = img_f @ txt_f.T
            pos = sim.diagonal()
            neg = sim[~torch.eye(B, dtype=torch.bool, device=sim.device)]
            return dict(pos_mean=pos.mean().item(), pos_std=pos.std().item(),
                        neg_mean=neg.mean().item(), neg_std=neg.std().item(),
                        gap=(pos.mean()-neg.mean()).item())

    def update(self, image_features, text_features, step, epoch=0,
               z_com_I=None, z_com_T=None):
        img = F.normalize(image_features.detach().float(), dim=-1)
        txt = F.normalize(text_features.detach().float(), dim=-1)
        self.records_full.append(_CosRec(step=step, epoch=epoch, **self._stats(img, txt)))
        if z_com_I is not None and z_com_T is not None:
            zi = F.normalize(z_com_I.detach().float(), dim=-1)
            zt = F.normalize(z_com_T.detach().float(), dim=-1)
            self.records_com.append(_CosRec(step=step, epoch=epoch, **self._stats(zi, zt)))
        self._save()

    def _save(self):
        data = {"model": self.model_name,
                "full": [asdict(r) for r in self.records_full],
                "com":  [asdict(r) for r in self.records_com]}
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)

    def plot(self, output_path=None, title=None):
        if not self.records_full: return
        if output_path is None:
            output_path = os.path.join(self.save_dir, "cosine_similarity.png")
        if title is None:
            title = f"{self.model_name}: Image-Text Cosine Similarity"
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
        steps_f = [r.step for r in self.records_full]
        ax = axes[0]
        if self.records_com:
            steps_c = [r.step for r in self.records_com]
            ax.plot(steps_c, [r.pos_mean for r in self.records_com],
                    color=COLOR_OURS, lw=2.5, label='Pos (InvCLIP)')
            ax.fill_between(steps_c,
                            [r.pos_mean-r.pos_std for r in self.records_com],
                            [r.pos_mean+r.pos_std for r in self.records_com],
                            color=COLOR_OURS, alpha=0.15)
            ax.plot(steps_c, [r.neg_mean for r in self.records_com],
                    color=COLOR_OURS, lw=2.5, ls='--', label='Neg (InvCLIP)')
        ax.plot(steps_f, [r.pos_mean for r in self.records_full],
                color=COLOR_BASE, lw=1.5, alpha=0.8, label='Pos (CLIP baseline)')
        ax.plot(steps_f, [r.neg_mean for r in self.records_full],
                color=COLOR_BASE, lw=1.5, ls='--', alpha=0.8, label='Neg (CLIP baseline)')
        ax.set_ylabel('Cosine Similarity', fontsize=11)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.25, linestyle=':')
        ax.axhline(0, color='#AAAAAA', lw=0.8, ls=':')
        ax2 = axes[1]
        if self.records_com:
            ax2.plot(steps_c, [r.gap for r in self.records_com],
                     color=COLOR_OURS, lw=2.5, label='InvCLIP')
        ax2.plot(steps_f, [r.gap for r in self.records_full],
                 color=COLOR_BASE, lw=1.5, alpha=0.8, label='CLIP baseline')
        ax2.set_xlabel('Training Step', fontsize=11)
        ax2.set_ylabel('Alignment gap  (pos − neg)', fontsize=11)
        ax2.legend(loc='upper left', fontsize=9, framealpha=0.8)
        ax2.grid(True, alpha=0.25, linestyle=':')
        ax2.axhline(0, color='#AAAAAA', lw=0.8, ls=':')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[CosineTracker] Saved → {output_path}")


# ============================================================================
# 1. Simplex ETF
# ============================================================================

def generate_simplex_etf(num_classes, feature_dim):
    K = num_classes
    I_K = torch.eye(K)
    ones = torch.ones(K, 1)
    H = I_K - (1.0 / K) * ones @ ones.T
    eigvals, eigvecs = torch.linalg.eigh(H)
    basis = eigvecs[:, -(K - 1):]
    scale = math.sqrt(K / (K - 1))
    etf_low = scale * basis
    torch.manual_seed(42)
    if feature_dim >= K - 1:
        random_matrix = torch.randn(feature_dim, K - 1)
        Q, _ = torch.linalg.qr(random_matrix)
        projection = Q[:, :K - 1]
        etf = etf_low @ projection.T
    else:
        projection = torch.randn(K - 1, feature_dim)
        projection = F.normalize(projection, p=2, dim=1)
        etf = etf_low @ projection
    etf = F.normalize(etf, p=2, dim=1)
    gram = etf @ etf.T
    off_diag = gram[~torch.eye(K, dtype=bool)].mean().item()
    expected = -1.0 / (K - 1)
    if abs(off_diag - expected) > 0.05:
        print(f"[WARNING] ETF off-diag = {off_diag:.4f}, expected {expected:.4f}")
    else:
        print(f"[ETF] Verified: off-diag = {off_diag:.4f} ~ {expected:.4f}")
    return etf


# ============================================================================
# 2. MaskPredictor — instance-level dynamic mask
# ============================================================================

class MaskPredictor(nn.Module):
    """
    动态 mask: 根据 image+text 联合特征预测每个样本的 mask。
    支持指定 target_k (保留维度数)。
    """
    def __init__(self, feature_dim, target_k=None, seed=None,
                 tau_start=2.0, tau_end=0.2):
        super().__init__()
        if target_k is None:
            target_k = feature_dim // 2
        self.target_k = target_k
        self.feature_dim = feature_dim
        self.target_ratio = target_k / feature_dim
        self.tau = tau_start
        self.tau_start = tau_start
        self.tau_end = tau_end
        hidden = max(feature_dim // 4, 64)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )
        self._last_scores = None

    def set_tau(self, progress):
        self.tau = self.tau_start * (self.tau_end / self.tau_start) ** progress

    def forward(self, image_features, text_features):
        joint = F.normalize(0.5 * (image_features + text_features), dim=-1)
        scores = self.mlp(joint)
        self._last_scores = scores.detach()
        soft_mask = torch.sigmoid(scores / self.tau)
        with torch.no_grad():
            _, topk_idx = torch.topk(scores, self.target_k, dim=-1)
            hard_mask = torch.zeros_like(scores)
            hard_mask.scatter_(-1, topk_idx, 1.0)
        if self.training:
            return {'soft': soft_mask, 'hard': hard_mask}
        else:
            return {'soft': hard_mask, 'hard': hard_mask}

    def get_soft_mask(self):
        if self._last_scores is None:
            return torch.zeros(self.feature_dim)
        return torch.sigmoid(self._last_scores.mean(0) / self.tau)

    def get_binary_mask(self, features=None, threshold=0.5):
        if self._last_scores is None:
            return torch.zeros(self.feature_dim)
        with torch.no_grad():
            scores = self._last_scores.mean(0)
            _, topk_idx = torch.topk(scores, self.target_k)
            mask = torch.zeros_like(scores)
            mask[topk_idx] = 1.0
            return mask

    def get_separation_stats(self):
        with torch.no_grad():
            if self._last_scores is None:
                return {'on_mean': 0, 'off_mean': 0, 'separation': 0,
                        'boundary_gap': 0, 'soft_budget': 0, 'soft_hard_l1': 0}
            scores = self._last_scores.mean(0)
            sorted_scores, _ = torch.sort(scores, descending=True)
            on_scores = sorted_scores[:self.target_k]
            off_scores = sorted_scores[self.target_k:]
            gap = on_scores[-1] - off_scores[0]
            soft = torch.sigmoid(scores / self.tau)
            hard = self.get_binary_mask()
            soft_hard_l1 = (soft - hard).abs().mean().item()
            return {
                'on_mean': on_scores.mean().item(),
                'off_mean': off_scores.mean().item(),
                'separation': on_scores.mean().item() - off_scores.mean().item(),
                'boundary_gap': gap.item(),
                'soft_budget': soft.mean().item(),
                'soft_hard_l1': soft_hard_l1,
            }


# ============================================================================
# 3. MLP Reconstructor
# ============================================================================

class MLPReconstructor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
    def forward(self, z_com):
        return self.net(z_com)


# ============================================================================
# 4. Sinkhorn
# ============================================================================

@torch.no_grad()
def sinkhorn_assign(scores, eps=0.05, num_iters=3):
    Q = torch.exp(scores / eps)
    Q = Q / (Q.sum() + 1e-8)
    B, K = Q.shape
    for _ in range(num_iters):
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
        Q = Q / K
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
    return Q


# ============================================================================
# 5. ★ InvariantMask — two-level mask
# ============================================================================

class InvariantMask(nn.Module):
    """
  
    def __init__(self, feature_dim, num_classes=20,
                 fine_k=None, coarse_k=None,
                 tau_end=0.2, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # fine: keep 30% of dims
        if fine_k is None:
            fine_k = int(feature_dim * 0.3)
        # coarse: keep 50% of dims
        if coarse_k is None:
            coarse_k = int(feature_dim * 0.5)

        self.fine_k   = fine_k
        self.coarse_k = coarse_k

        self.fine_predictor   = MaskPredictor(feature_dim, target_k=fine_k,   seed=42,  tau_end=tau_end)
        self.coarse_predictor = MaskPredictor(feature_dim, target_k=coarse_k, seed=123, tau_end=tau_end)

        class_etf = generate_simplex_etf(num_classes, feature_dim)
        self.register_buffer('class_etf', class_etf)

        # projector operates on fine features (768 dim)
        self.inv_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.recon_i2t = MLPReconstructor(feature_dim)
        self.recon_t2i = MLPReconstructor(feature_dim)

    def forward(self, image_features, text_features):
        # ── Fine mask ──
        fine_masks   = self.fine_predictor(image_features, text_features)
        soft_fine    = fine_masks['soft']   # [B, D]
        hard_fine    = fine_masks['hard']   # [B, D]

        # ── Coarse mask ──
        coarse_masks = self.coarse_predictor(image_features, text_features)
        soft_coarse  = coarse_masks['soft'] # [B, D]
        hard_coarse  = coarse_masks['hard'] # [B, D]

        # ── Soft path: fine features for ETF/recon ──
        z_fine_I = image_features * soft_fine
        z_fine_T = text_features  * soft_fine
        z_fine_I_norm = F.normalize(z_fine_I, dim=-1)
        z_fine_T_norm = F.normalize(z_fine_T, dim=-1)

        # ── Soft path: coarse features ──
        z_coarse_I = image_features * soft_coarse
        z_coarse_T = text_features  * soft_coarse
        z_coarse_I_norm = F.normalize(z_coarse_I, dim=-1)
        z_coarse_T_norm = F.normalize(z_coarse_T, dim=-1)

        # ── Concatenated features for contrastive loss ──
        z_cat_I = torch.cat([z_fine_I_norm, z_coarse_I_norm], dim=-1)  # [B, 2D]
        z_cat_T = torch.cat([z_fine_T_norm, z_coarse_T_norm], dim=-1)  # [B, 2D]
        z_com_I_norm = F.normalize(z_cat_I, dim=-1)
        z_com_T_norm = F.normalize(z_cat_T, dim=-1)

        # ── ETF projector on fine features ──
        z_joint = F.normalize(0.5 * (z_fine_I_norm + z_fine_T_norm), dim=-1)
        z_joint_proj = F.normalize(self.inv_projector(z_joint), dim=-1)
        z_I_proj = F.normalize(self.inv_projector(z_fine_I), dim=-1)
        z_T_proj = F.normalize(self.inv_projector(z_fine_T), dim=-1)

        scores     = z_joint_proj @ self.class_etf.T
        assignment = sinkhorn_assign(scores, eps=0.05, num_iters=3)
        yi         = assignment.argmax(dim=1)

        recon_i2t = self.recon_i2t(z_fine_I)
        recon_t2i = self.recon_t2i(z_fine_T)

        # ── Hard path for consistency loss (fine only) ──
        z_fine_I_hard = image_features * hard_fine
        z_fine_T_hard = text_features  * hard_fine
        z_com_I_hard_norm = F.normalize(z_fine_I_hard, dim=-1)
        z_com_T_hard_norm = F.normalize(z_fine_T_hard, dim=-1)

        return {
            'soft_mask': soft_fine, 'hard_mask': hard_fine,
            'z_com_I': z_fine_I, 'z_com_T': z_fine_T,
            'z_com_I_norm': z_com_I_norm, 'z_com_T_norm': z_com_T_norm,
            'z_fine_I_norm': z_fine_I_norm, 'z_fine_T_norm': z_fine_T_norm,
            'z_I_proj': z_I_proj, 'z_T_proj': z_T_proj,
            'z_joint_proj': z_joint_proj,
            'yi': yi, 'assignment': assignment,
            'recon_i2t': recon_i2t, 'recon_t2i': recon_t2i,
            'z_com_I_hard_norm': z_com_I_hard_norm,
            'z_com_T_hard_norm': z_com_T_hard_norm,
        }

    def get_mask_stats(self):
        with torch.no_grad():
            soft = self.fine_predictor.get_soft_mask()
            hard = self.fine_predictor.get_binary_mask()
            k    = self.fine_predictor.target_k
            D    = self.feature_dim
            sep  = self.fine_predictor.get_separation_stats()
            _ls  = self.fine_predictor._last_scores
            return {
                'sparsity':    (hard < 0.5).float().mean().item(),
                'binary':      ((hard < 0.1) | (hard > 0.9)).float().mean().item(),
                'k_ratio':     k / D,
                'soft_mean':   soft.mean().item(),
                'soft_std':    soft.std().item(),
                'score_mean':  _ls.mean().item() if _ls is not None else 0.0,
                'score_std':   _ls.std().item()  if _ls is not None else 0.0,
                'tau':         self.fine_predictor.tau,
                'separation':  sep['separation'],
                'boundary_gap':sep['boundary_gap'],
                'on_mean':     sep['on_mean'],
                'off_mean':    sep['off_mean'],
                'soft_budget': sep['soft_budget'],
                'soft_hard_l1':sep['soft_hard_l1'],
            }


# ============================================================================
# 6. Loss
# ============================================================================

class InvCLIPLoss(nn.Module):
    def __init__(self, lambda_ctr=2.0, lambda_align=1.0, lambda_div=1.0,
                 lambda_recon=0.5, lambda_sep=1.0, sep_margin=4.0,
                 lambda_budget=2.0, lambda_consist=1.0, lambda_occ=0.5,
                 **kwargs):
        super().__init__()
        self.lambda_ctr     = lambda_ctr
        self.lambda_align   = lambda_align
        self.lambda_div     = lambda_div
        self.lambda_recon   = lambda_recon
        self.lambda_sep     = lambda_sep
        self.sep_margin     = sep_margin
        self.lambda_budget  = lambda_budget
        self.lambda_consist = lambda_consist
        self.lambda_occ     = lambda_occ

    def contrastive_loss(self, z_com_I_norm, z_com_T_norm, logit_scale):
        B = z_com_I_norm.shape[0]
        logits = logit_scale * z_com_I_norm @ z_com_T_norm.T
        labels = torch.arange(B, device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def etf_alignment_loss(self, outputs, etf):
        z_I_proj = outputs['z_I_proj']
        z_T_proj = outputs['z_T_proj']
        yi = outputs['yi']
        v_yi = etf[yi]
        sim_I = (z_I_proj * v_yi).sum(dim=-1)
        sim_T = (z_T_proj * v_yi).sum(dim=-1)
        loss = ((1 - sim_I) + (1 - sim_T)).mean()
        return loss, {'max_sim_I': sim_I.mean().item(), 'max_sim_T': sim_T.mean().item()}

    def etf_diversity_loss(self, outputs, K):
        yi = outputs['yi']
        B  = yi.shape[0]
        counts = torch.zeros(K, device=yi.device)
        counts.scatter_add_(0, yi, torch.ones_like(yi, dtype=torch.float))
        q   = (counts / B).clamp(min=1e-8)
        kl  = (q * (q * K).log()).sum()
        occ = (counts > 0).sum().item()
        return kl, occ, counts

    def occupancy_loss(self, counts, K):
        """★ 强制所有 ETF 顶点被利用"""
        occ = (counts > 0).float().sum()
        return F.relu(K - occ) / K

    def reconstruction_loss(self, outputs, image_features, text_features):
        loss_i2t = ((outputs['recon_i2t'] - text_features.detach()) ** 2).mean()
        loss_t2i = ((outputs['recon_t2i'] - image_features.detach()) ** 2).mean()
        return (loss_i2t + loss_t2i) / 2

    def separation_loss(self, mask_predictor):
        if mask_predictor._last_scores is None:
            return torch.tensor(0.0)
        scores = mask_predictor._last_scores.mean(0)
        k = mask_predictor.target_k
        sorted_scores, _ = torch.sort(scores, descending=True)
        mean_on  = sorted_scores[:k].mean()
        mean_off = sorted_scores[k:].mean()
        return F.relu(self.sep_margin - (mean_on - mean_off))

    def budget_loss(self, mask_predictor):
        if mask_predictor._last_scores is None:
            return torch.tensor(0.0)
        scores = mask_predictor._last_scores
        soft   = torch.sigmoid(scores / mask_predictor.tau)
        target = mask_predictor.target_ratio
        return (soft.mean() - target) ** 2

    def consistency_loss(self, outputs):
        cos_I = (outputs['z_fine_I_norm'] * outputs['z_com_I_hard_norm']).sum(dim=-1)
        cos_T = (outputs['z_fine_T_norm'] * outputs['z_com_T_hard_norm']).sum(dim=-1)
        return ((1 - cos_I) + (1 - cos_T)).mean()

    def forward(self, outputs, mask_module, image_features, text_features, logit_scale, step=0):
        etf = mask_module.class_etf
        K   = etf.shape[0]

        l_ctr   = self.contrastive_loss(outputs['z_com_I_norm'], outputs['z_com_T_norm'], logit_scale)
        l_align, align_stats = self.etf_alignment_loss(outputs, etf)
        l_div, occ, counts   = self.etf_diversity_loss(outputs, K)
        l_recon = self.reconstruction_loss(outputs, image_features, text_features)
        l_sep   = self.separation_loss(mask_module.fine_predictor)
        l_budget= self.budget_loss(mask_module.fine_predictor)
        l_consist = self.consistency_loss(outputs)
        l_occ   = self.occupancy_loss(counts, K)  # ★ new

        total = (self.lambda_ctr    * l_ctr
               + self.lambda_align  * l_align
               + self.lambda_div    * l_div
               + self.lambda_recon  * l_recon
               + self.lambda_sep    * l_sep
               + self.lambda_budget * l_budget
               + self.lambda_consist* l_consist
               + self.lambda_occ    * l_occ)       # ★

        if torch.isnan(total) or torch.isinf(total):
            print("[WARNING] NaN/Inf loss, returning 0")
            total = torch.zeros(1, device=total.device, requires_grad=True)

        return {'total': total, 'ctr': l_ctr, 'align': l_align,
                'div': l_div, 'recon': l_recon, 'sep': l_sep,
                'budget': l_budget, 'consist': l_consist, 'occ': occ,
                'l_occ': l_occ,
                **{f'align_{k}': v for k, v in align_stats.items()}}


# ============================================================================
# 7. Trainer
# ============================================================================

class CLIP_InvCLIP_Train:
    def __init__(self, rank, local_rank, args):
        self.rank = rank
        self.local_rank = local_rank
        self.args = args

        self.model, self.preprocess = longclip.load_from_clip(
            args.base_model, device='cpu', download_root=args.download_root, args=args)
        self.model.train()
        self.model.logit_scale = nn.Parameter(torch.ones([]) * args.log_scale)
        self.model = self.model.cuda()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).cuda()
            self.feature_dim = self.model.encode_image(dummy).shape[-1]
            del dummy; torch.cuda.empty_cache()
        print(f"[Rank {rank}] Feature dim: {self.feature_dim}")

        # ★ 冻结的原始 CLIP，用于 cosine baseline
        import clip as _clip
        self.ref_clip, _ = _clip.load(args.base_model, device='cuda')
        self.ref_clip.eval()
        for p in self.ref_clip.parameters():
            p.requires_grad_(False)
        print(f"[Rank {rank}] Frozen reference CLIP loaded")

        self.mask_module = InvariantMask(
            feature_dim=self.feature_dim,
            num_classes=args.num_classes,
            fine_k=int(self.feature_dim * args.fine_ratio),
            coarse_k=int(self.feature_dim * args.coarse_ratio),
            tau_end=args.tau_end,
        ).cuda()

        self.inv_loss = InvCLIPLoss(
            lambda_ctr=args.lambda_ctr,
            lambda_align=args.lambda_align, lambda_div=args.lambda_div,
            lambda_recon=args.lambda_recon,
            lambda_sep=args.lambda_sep, sep_margin=args.sep_margin,
            lambda_budget=args.lambda_budget,
            lambda_consist=args.lambda_consist,
            lambda_occ=args.lambda_occ)

        self.batch_size = args.batch_size
        self.num_epoch  = args.epochs
        self.lr         = args.lr
        self.mask_lr    = args.mask_lr
        self.phase1_steps      = args.phase1_steps
        self.logit_freeze_steps= args.logit_freeze_steps
        self.lr_decay_epoch    = args.lr_decay_epoch  # ★ early decay

        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[local_rank], find_unused_parameters=True)
        self.mask_module = torch.nn.parallel.DistributedDataParallel(
            self.mask_module, device_ids=[local_rank], find_unused_parameters=True)

        clip_params, logit_params = [], []
        for name, param in self.model.named_parameters():
            if 'logit_scale' in name: logit_params.append(param)
            else: clip_params.append(param)
        self.optimizer = optim.AdamW([
            {'params': clip_params,   'lr': self.lr},
            {'params': logit_params,  'lr': self.lr * 0.1},
        ], weight_decay=args.weight_decay)
        self.mask_optimizer = optim.AdamW(
            list(self.mask_module.parameters()), lr=self.mask_lr, weight_decay=1e-4)
        self.scaler = GradScaler()

        world_size = dist.get_world_size()
        self.accumulation_steps = min(8, max(1, 512 // self.batch_size // world_size))
        print(f'Effective BS: {self.batch_size * self.accumulation_steps * world_size}')

        run_id = get_run_id('runs')
        self.logdir = f'./runs/{run_id:06d}_{args.base_model[-2:]}_InvCLIP_K{args.num_classes}'
        if rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            self.loss_log   = LossManager(file_path=os.path.join(self.logdir, "loss.txt"))
            self.metric_log = LossManager(file_path=os.path.join(self.logdir, "metric.txt"))
            print(f"Log dir: {self.logdir}"); print(args)
        self.writer = SummaryWriter(self.logdir)

        if rank == 0:
            self.gap_tracker    = GapTracker(self.logdir, model_name="InvCLIP")
            self.cosine_tracker = CosineTracker(self.logdir, log_every=100, model_name="InvCLIP")

        # ★ best checkpoint tracking
        self.best_z_com = 0.0

    def train(self, resume=False, warmup_length=200, args=None):
        trainset     = share4v_train_dataset()
        train_sampler= DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=8, pin_memory=True)
        total_steps       = self.num_epoch * len(train_loader)
        self.total_steps  = total_steps
        self.scheduler    = cosine_lr(self.optimizer, base_lr=self.lr,
                                      warmup_length=warmup_length, steps=total_steps)
        self.mask_scheduler = cosine_lr(self.mask_optimizer, base_lr=self.mask_lr,
                                        warmup_length=warmup_length, steps=total_steps)
        for epoch in range(self.num_epoch):
            train_sampler.set_epoch(epoch)

            # ★ lr decay after lr_decay_epoch
            if epoch == self.lr_decay_epoch and self.rank == 0:
                for pg in self.optimizer.param_groups:
                    pg['lr'] *= 0.5
                for pg in self.mask_optimizer.param_groups:
                    pg['lr'] *= 0.5
                print(f"[LR Decay] Halved lr at epoch {epoch}")

            self._train_epoch(train_loader, epoch, args=args)
            if self.rank == 0:
                z_com = self._test(epoch)
                self._save(epoch)
                # ★ save best checkpoint
                if z_com > self.best_z_com:
                    self.best_z_com = z_com
                    self._save_best(epoch)
                result_dict = eval_coco(self.model, self.preprocess)
                print(result_dict)
                self.metric_log.log(f'Epoch {epoch}: COCO = {result_dict}')
                self._log_mask(epoch)

        if self.rank == 0 and hasattr(self, 'gap_tracker') and self.gap_tracker.records:
            try:
                GapTracker.plot(
                    {"InvCLIP": self.gap_tracker.save_path},
                    output_path=os.path.join(self.logdir, "gap_curve_InvCLIP.png"),
                    title="InvCLIP: Train-Test Gap")
            except Exception as e:
                print(f'[GapTracker] Plot error: {e}')

        if self.rank == 0 and hasattr(self, 'cosine_tracker'):
            self.cosine_tracker.plot(
                output_path=os.path.join(self.logdir, "cosine_similarity.png"),
                title="InvCLIP: Image-Text Cosine Similarity")

    def _train_epoch(self, dataloader, epoch, args=None):
        num_batches = len(dataloader)
        self.model.train(); self.mask_module.train()
        pbar = tqdm(total=num_batches, disable=(self.rank != 0))

        for i, (images, texts) in enumerate(dataloader):
            step = num_batches * epoch + i
            raw_texts = texts  # ★ 保留原始字符串供 ref_clip 使用
            with torch.no_grad():
                texts = longclip.tokenize(texts, truncate=True).cuda()
            images = images.cuda()
            self.scheduler(step); self.mask_scheduler(step)

            if self.total_steps > 0:
                progress = min(1.0, step / self.total_steps)
                self.mask_module.module.fine_predictor.set_tau(progress)
                self.mask_module.module.coarse_predictor.set_tau(progress)

            if step < self.logit_freeze_steps:
                for pg in self.optimizer.param_groups:
                    for p in pg['params']:
                        if p.shape == torch.Size([]): p.requires_grad_(False)
            else:
                for pg in self.optimizer.param_groups:
                    for p in pg['params']:
                        if p.shape == torch.Size([]): p.requires_grad_(True)

            in_phase1 = (step < self.phase1_steps)

            if in_phase1:
                image_f  = self.model.module.encode_image(images)
                text_f   = self.model.module.encode_text(texts)
                image_fn = F.normalize(image_f, dim=-1)
                text_fn  = F.normalize(text_f, dim=-1)
                ls       = self.model.module.logit_scale.exp()
                logits   = ls * image_fn @ text_fn.T
                labels   = torch.arange(image_f.shape[0], device=logits.device)
                loss     = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
                loss     = loss / self.accumulation_steps
                loss_dict_p2 = None
            else:
                image_f = F.normalize(self.model.module.encode_image(images), dim=-1)
                text_f  = F.normalize(self.model.module.encode_text(texts),   dim=-1)
                mask_output  = self.mask_module(image_f, text_f)
                ls           = self.model.module.logit_scale.exp()
                loss_dict_p2 = self.inv_loss(
                    mask_output, self.mask_module.module, image_f, text_f, ls,
                    step=step - self.phase1_steps)
                loss = loss_dict_p2['total'] / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                self.scaler.unscale_(self.mask_optimizer)
                has_nan = False
                for param in list(self.model.parameters()) + list(self.mask_module.parameters()):
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan = True; param.grad.zero_()
                if has_nan and self.rank == 0:
                    print(f"[NaN grad] iter {i}")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.mask_module.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                if not in_phase1:
                    self.scaler.step(self.mask_optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(); self.mask_optimizer.zero_grad()
                with torch.no_grad():
                    self.model.module.logit_scale.clamp_(0, 4.6052)

            if self.rank == 0:
                pbar.update(1)
                if in_phase1:
                    pbar.set_description(
                        f'[P1] Ep{epoch:02d} it{i:05d} | ctr:{loss.item()*self.accumulation_steps:.3f}')
                else:
                    d  = loss_dict_p2
                    ms = self.mask_module.module.get_mask_stats()
                    pbar.set_description(
                        f'[P2] Ep{epoch:02d} it{i:05d} | '
                        f'ctr:{d["ctr"].item():.3f} con:{d["consist"].item():.3f} '
                        f'occ:{d["occ"]}/{self.args.num_classes} | '
                        f'τ:{ms["tau"]:.2f} L1:{ms["soft_hard_l1"]:.3f}')

                if i % 100 == 0 and not in_phase1:
                    self._log_detailed(epoch, i, step, loss_dict_p2, mask_output)

                    # ★ CosineTracker: ref_clip vs InvCLIP
                    try:
                        import clip as _clip
                        with torch.no_grad():
                            ref_tokens = _clip.tokenize(raw_texts, truncate=True).cuda()
                            ref_img = F.normalize(self.ref_clip.encode_image(images).float(), dim=-1)
                            ref_txt = F.normalize(self.ref_clip.encode_text(ref_tokens).float(), dim=-1)
                        self.cosine_tracker.update(
                            image_features=ref_img, text_features=ref_txt,
                            step=step, epoch=epoch,
                            z_com_I=mask_output['z_com_I_norm'],
                            z_com_T=mask_output['z_com_T_norm'],
                        )
                    except Exception as e:
                        print(f'[CosineTracker] Error: {e}')

                    # ★ GapTracker
                    try:
                        self.gap_tracker.record_invclip(
                            step=step, mask_module=self.mask_module,
                            image_features=image_f, text_features=text_f,
                            loss_dict=loss_dict_p2, epoch=epoch)
                    except Exception as e:
                        print(f'[GapTracker] Error: {e}')

            if (i + 1) % 2000 == 0 and self.rank == 0:
                self._test(epoch); self._save(epoch)
        if self.rank == 0:
            pbar.close()

    def _log_detailed(self, epoch, i, step, d, mask_output):
        with torch.no_grad():
            etf    = self.mask_module.module.class_etf; K = etf.shape[0]
            yi     = mask_output['yi']; z_I_proj = mask_output['z_I_proj']
            v_yi   = etf[yi]; sim_I = (z_I_proj * v_yi).sum(dim=-1)
            ang_I  = torch.rad2deg(torch.acos(sim_I.clamp(-1, 1)))
            all_sim= z_I_proj @ etf.T
            top2   = torch.topk(all_sim, k=2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).mean().item()
            ms     = self.mask_module.module.get_mask_stats()
        msg = (f'\n[Ep{epoch:02d} it{i:05d}]\n'
               f'  Ctr:{d["ctr"].item():.4f} Consist:{d["consist"].item():.4f}\n'
               f'  Sep:{d["sep"].item():.4f} Budget:{d["budget"].item():.6f}\n'
               f'  ETF: align={d["align"].item():.4f} div={d["div"].item():.4f} '
               f'occ={d["occ"]}/{K} l_occ={d["l_occ"].item():.4f}\n'
               f'  Recon:{d["recon"].item():.4f}\n'
               f'  ETF: ang={ang_I.mean().item():.1f} margin={margin:.4f}\n'
               f'  Mask: sp={ms["sparsity"]:.3f} bi={ms["binary"]:.3f} '
               f'soft={ms["soft_mean"]:.3f}±{ms["soft_std"]:.3f}\n'
               f'  ★ τ={ms["tau"]:.4f} sep={ms["separation"]:.3f} '
               f'gap={ms["boundary_gap"]:.3f} '
               f'on={ms["on_mean"]:.3f} off={ms["off_mean"]:.3f}\n'
               f'  ★ soft_hard_L1={ms["soft_hard_l1"]:.4f} '
               f'budget={ms["soft_budget"]:.3f}\n'
               f'  Total:{d["total"].item():.4f}')
        print(msg); self.loss_log.log(msg)
        self.writer.add_scalar('loss/total',   d['total'].item(),   step)
        self.writer.add_scalar('loss/ctr',     d['ctr'].item(),     step)
        self.writer.add_scalar('loss/align',   d['align'].item(),   step)
        self.writer.add_scalar('loss/div',     d['div'].item(),     step)
        self.writer.add_scalar('loss/recon',   d['recon'].item(),   step)
        self.writer.add_scalar('loss/sep',     d['sep'].item(),     step)
        self.writer.add_scalar('loss/budget',  d['budget'].item(),  step)
        self.writer.add_scalar('loss/consist', d['consist'].item(), step)
        self.writer.add_scalar('loss/l_occ',   d['l_occ'].item(),   step)
        self.writer.add_scalar('etf/occ',      d['occ'],            step)
        self.writer.add_scalar('etf/angle',    ang_I.mean().item(), step)
        self.writer.add_scalar('etf/margin',   margin,              step)
        self.writer.add_scalar('mask/tau',          ms['tau'],          step)
        self.writer.add_scalar('mask/separation',   ms['separation'],   step)
        self.writer.add_scalar('mask/soft_hard_l1', ms['soft_hard_l1'], step)
        self.writer.add_scalar('mask/soft_budget',  ms['soft_budget'],  step)

    @torch.no_grad()
    def _test(self, epoch):
        if self.rank != 0: return 0.0
        self.model.eval(); self.mask_module.eval()
        testset    = share4v_val_dataset()
        testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                                  num_workers=4, pin_memory=True)
        correct_full, correct_com, total = 0, 0, 0

        for images, text in tqdm(testloader, desc='Testing'):
            images = images.cuda()
            image_features = F.normalize(self.model.module.encode_image(images), dim=-1)
            text   = longclip.tokenize(text, truncate=True).cuda()
            text_features  = F.normalize(self.model.module.encode_text(text), dim=-1)
            mask_out = self.mask_module(image_features, text_features)
            z_com_I  = mask_out['z_com_I_norm']
            z_com_T  = mask_out['z_com_T_norm']
            for j in range(text_features.shape[0]):
                if torch.argmax(text_features[j] @ image_features.T) == j:
                    correct_full += 1
                if torch.argmax(z_com_T[j] @ z_com_I.T) == j:
                    correct_com += 1
                total += 1

        acc_full = correct_full / total
        acc_com  = correct_com  / total
        msg = (f"[Test] Share4V retrieval: z_full={acc_full:.4f} z_com={acc_com:.4f} "
               f"(delta={acc_com-acc_full:+.4f}) @ epoch {epoch}")
        print(f"\n{'='*60}\n{msg}\n{'='*60}")

        # random mask baseline
        all_img = torch.cat([F.normalize(self.model.module.encode_image(imgs.cuda()), dim=-1)
                             for imgs, _ in testloader], dim=0)
        all_txt = torch.cat([F.normalize(self.model.module.encode_text(
                             longclip.tokenize(txt, truncate=True).cuda()), dim=-1)
                             for _, txt in testloader], dim=0)
        _ls = self.mask_module.module.fine_predictor._last_scores
        scores = _ls.mean(0).detach() if _ls is not None else torch.zeros(self.feature_dim).cuda()
        k = self.mask_module.module.fine_k
        random_accs = []
        for seed in range(5):
            torch.manual_seed(seed)
            rand_idx  = torch.randperm(scores.shape[0])[:k].cuda()
            rand_mask = torch.zeros_like(scores).cuda()
            rand_mask[rand_idx] = 1.0
            z_I_rand = F.normalize(all_img * rand_mask, dim=-1)
            z_T_rand = F.normalize(all_txt * rand_mask, dim=-1)
            N = z_I_rand.shape[0]
            correct_rand = sum(1 for j in range(N)
                               if torch.argmax(z_T_rand[j] @ z_I_rand.T) == j)
            random_accs.append(correct_rand / N)
        rand_mean = sum(random_accs) / len(random_accs)
        rand_str  = ', '.join(f'{a:.4f}' for a in random_accs)
        print(f"[Random mask x5] accs=[{rand_str}] mean={rand_mean:.4f}")
        print(f"[Learned - Random] {acc_com - rand_mean:+.4f}")
        self.metric_log.log(f"[Random mask] mean={rand_mean:.4f}, learned-random={acc_com-rand_mean:+.4f}")
        self.metric_log.log(msg)
        self.writer.add_scalar('test/acc_full', acc_full, epoch)
        self.writer.add_scalar('test/acc_com',  acc_com,  epoch)
        self.model.train(); self.mask_module.train()
        return acc_com

    def _save(self, epoch):
        if self.rank != 0: return
        torch.save(self.model.module.state_dict(),
                   os.path.join(self.logdir, f"invclip_ep{epoch:02d}.pt"))
        torch.save({'state_dict': self.mask_module.module.state_dict(),
                    'class_etf':  self.mask_module.module.class_etf},
                   os.path.join(self.logdir, f"mask_ep{epoch:02d}.pt"))
        print(f'>>>>> Saved epoch {epoch} <<<<<')

    def _save_best(self, epoch):
        if self.rank != 0: return
        torch.save(self.model.module.state_dict(),
                   os.path.join(self.logdir, "invclip_best.pt"))
        torch.save({'state_dict': self.mask_module.module.state_dict(),
                    'class_etf':  self.mask_module.module.class_etf},
                   os.path.join(self.logdir, "mask_best.pt"))
        print(f'>>>>> Saved BEST checkpoint @ epoch {epoch} (z_com={self.best_z_com:.4f}) <<<<<')

    def _log_mask(self, epoch):
        stats = self.mask_module.module.get_mask_stats()
        _ls   = self.mask_module.module.fine_predictor._last_scores
        scores= _ls.mean(0).cpu().numpy() if _ls is not None else np.zeros(self.feature_dim)
        tau   = stats['tau']
        with np.errstate(over='ignore'):
            soft = 1.0 / (1.0 + np.exp(-np.clip(scores / tau, -88, 88)))
        self.writer.add_histogram('mask/scores', scores, epoch)
        self.writer.add_histogram('mask/soft',   soft,   epoch)
        print(f"\n[Mask @ Epoch {epoch}]")
        print(f"  Sparsity: {stats['sparsity']:.2%}  Binary: {stats['binary']:.2%}")
        print(f"  ★ τ={stats['tau']:.4f} sep={stats['separation']:.3f} gap={stats['boundary_gap']:.3f}")
        print(f"    ON={stats['on_mean']:.4f} OFF={stats['off_mean']:.4f}")
        print(f"    soft_hard_L1={stats['soft_hard_l1']:.4f} budget={stats['soft_budget']:.4f}")
        bins = [-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8]
        hist = np.histogram(scores, bins=bins)[0]
        labels = [f'{bins[j]:.0f}~{bins[j+1]:.0f}' for j in range(len(bins)-1)]
        print(f"  Score hist: {dict(zip(labels, hist))}")


# ============================================================================
# 8. Main
# ============================================================================

def setup_distributed(backend="nccl"):
    num_gpus = torch.cuda.device_count()
    rank      = int(os.environ["RANK"])
    world_size= int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    return rank, rank % num_gpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='InvCLIP')
    parser.add_argument('--lr',           default=1e-6,    type=float)
    parser.add_argument('--mask_lr',      default=5e-3,    type=float)
    parser.add_argument('--weight_decay', default=1e-2,    type=float)
    parser.add_argument('--log_scale',    default=4.6052,  type=float)
    parser.add_argument("--base_model",   default="L14")
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--epochs",       type=int, default=5)
    parser.add_argument("--warmup_length",default=200,     type=int)
    parser.add_argument("--resume",       default=False,   action='store_true')
    parser.add_argument("--download-root",default=None)

    # Loss weights
    parser.add_argument("--lambda_ctr",     default=4.0,  type=float)
    parser.add_argument("--lambda_align",   default=2.0,  type=float)
    parser.add_argument("--lambda_div",     default=3.0,  type=float)
    parser.add_argument("--lambda_recon",   default=0.5,  type=float)
    parser.add_argument("--lambda_sep",     default=1.0,  type=float)
    parser.add_argument("--sep_margin",     default=4.0,  type=float)
    parser.add_argument("--lambda_budget",  default=2.0,  type=float)
    parser.add_argument("--lambda_consist", default=1.0,  type=float)
    parser.add_argument("--lambda_occ",     default=0.5,  type=float,
                        help="★ occupancy loss weight")

    # Structure
    parser.add_argument("--num_classes",          default=20,   type=int)
    parser.add_argument("--fine_ratio",            default=0.3,  type=float,
                        help="fraction of dims kept by fine mask")
    parser.add_argument("--coarse_ratio",          default=0.5,  type=float,
                        help="fraction of dims kept by coarse mask")
    parser.add_argument("--target_sparsity_ratio", default=0.3,  type=float,
                        help="kept for compatibility, not used directly")

    # Stability
    parser.add_argument("--phase1_steps",       default=500,  type=int)
    parser.add_argument("--logit_freeze_steps", default=500,  type=int)
    parser.add_argument("--tau_end",            default=0.1,  type=float)
    parser.add_argument("--lr_decay_epoch",     default=3,    type=int,
                        help="★ halve lr at this epoch")

    args = parser.parse_args()
    if args.base_model == 'L14': args.base_model = 'ViT-L/14'
    elif args.base_model == 'B16': args.base_model = 'ViT-B/16'

    rank, local_rank = setup_distributed()
    print(f"[Rank {rank}] DDP initialized")
    trainer = CLIP_InvCLIP_Train(rank=rank, local_rank=local_rank, args=args)
    trainer.train(resume=args.resume, warmup_length=args.warmup_length, args=args)
