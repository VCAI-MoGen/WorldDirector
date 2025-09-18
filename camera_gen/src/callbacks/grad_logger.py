import os
from lightning.pytorch.callbacks import Callback
import torch


class GradLogger(Callback):
    """
    Logs/prints gradient statistics after backprop and unscaling, right before the
    optimizer step. Useful for quick debugging with small runs (e.g., bs=1, epoch=1).

    Enable by setting env var LOG_GRADS=1 (wired in train.py).
    """

    def __init__(
        self,
        *,
        per_param: bool = True,
        limit_params: int | None = None,
        log_global_norm: bool = True,
    ) -> None:
        super().__init__()
        self.per_param = per_param
        self.limit_params = limit_params
        self.log_global_norm = log_global_norm

    def on_fit_start(self, trainer, pl_module):  # type: ignore[override]
        print("[grad] GradLogger enabled: will print on optimizer step")

    # Match signature used elsewhere in this repo (see FixNANinGrad)
    def on_before_optimizer_step(self, trainer, pl_module, loss) -> None:  # type: ignore[override]
        # Gradients are unscaled at this point in Lightning's precision plugins.
        params = list(pl_module.net.named_parameters()) if hasattr(pl_module, "net") else list(pl_module.named_parameters())

        norms = []
        printed = 0
        for name, p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            # Skip empty grads just in case
            if g.numel() == 0:
                continue

            gn = g.norm(2)
            norms.append(gn)

            if self.per_param:
                if (self.limit_params is None) or (printed < self.limit_params):
                    g_min = torch.min(g).item()
                    g_max = torch.max(g).item()
                    g_mean = torch.mean(g).item()
                    print(
                        f"[grad] {name:50s} shape={tuple(g.shape)} "
                        f"norm={gn.item():.6e} min={g_min:.6e} max={g_max:.6e} mean={g_mean:.6e}"
                    )
                    printed += 1

        if self.log_global_norm and norms:
            # L2 norm across all parameter gradients
            stacked = torch.stack([n for n in norms])
            global_norm = torch.norm(stacked, 2)
            try:
                pl_module.log("train/grad_global_norm", global_norm.item(), prog_bar=False, on_step=True, on_epoch=False, sync_dist=getattr(pl_module, "sync_dist", False))
            except Exception:
                # Fallback to printing if logger is not available
                print(f"[grad] global_norm={global_norm.item():.6e}")
