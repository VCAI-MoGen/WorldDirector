import logging
from pathlib import Path
from lightning.pytorch.callbacks import Callback
import torch

log = logging.getLogger(__name__)


class FixNANinGrad(Callback):
    def __init__(self, monitor, bad_path: str = "assets/bad.txt"):
        super().__init__()
        self.monitor = monitor
        self.continuous_nan_batchs = 0
        self.bad_path = Path(bad_path)
        self.bad_logged_ids: set[str] = set()

    def on_before_backward(self, trainer, pl_module, loss):
        pl_module.do_optimizer_step = True

    def on_before_optimizer_step(self, trainer, pl_module, loss) -> None:
        has_nan = []
        is_inf = []
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan.append(name)
                    pl_module.do_optimizer_step = False
                if torch.isinf(param.grad).any():
                    is_inf.append(name)
                    pl_module.do_optimizer_step = False
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        if len(has_nan) > 0:
            print(f"Found NaN in {has_nan}")
            self._log_bad_samples(pl_module, "grad_nan", has_nan)
        if len(is_inf) > 0:
            print(f"Found Inf in {is_inf}")
            self._log_bad_samples(pl_module, "grad_inf", is_inf)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        logs = trainer.callback_metrics
        i, found_metric = 0, False
        while i < len(self.monitor) and not found_metric:
            if self.monitor[i] in logs.keys():
                current = logs[self.monitor[i]].squeeze()
                found_metric = True
            else:
                i += 1

        if not found_metric:
            raise ValueError("Asked metric not in logs")

        if not torch.isfinite(current):
            self.continuous_nan_batchs += 1
            if self.continuous_nan_batchs >= 5:
                trainer.should_stop = True
                log.info("Training interrupted because of NaN in {self.monitor}")
        else:
            self.continuous_nan_batchs = 0

    # ------------------------------------------------------------------ #

    def _log_bad_samples(self, pl_module, reason: str, params: list[str]):
        filenames = getattr(pl_module, "current_filenames", [])
        if not filenames:
            return
        self.bad_path.parent.mkdir(parents=True, exist_ok=True)
        with self.bad_path.open("a", encoding="utf-8") as f:
            for fn in filenames:
                stem = Path(str(fn)).stem
                if stem in self.bad_logged_ids:
                    continue
                f.write(f"{stem}\t{reason}\t{','.join(params)}\n")
                self.bad_logged_ids.add(stem)
