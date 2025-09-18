from copy import deepcopy
from typing import Optional
import multiprocessing
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets.datamodule import Datamodule
from callbacks.fix_nans import FixNANinGrad
from callbacks.grad_logger import GradLogger
from utils.random_utils import set_random_seed
import torch
from pathlib import Path

torch.set_float32_matmul_precision("medium")


# -------- helpers: read good/bad & filter in-place  -------- #
##### 이 블록은 smplh/*.pkl 일부 샘플 중에서도 smplx parameter의 일부 파일들이 nan/inf 같은 corrupt value가 있는 것이 발견이 되어 
### training sequence에 들어가기 전에 사전에 필터링하기 위한 추가된 함수입니다. 그래서 et-data의 train_test split txt 파일의 목록 중에서 
### bad txt로 분류되는 샘플 목록은 train/test에서 각각 제외가 되도록 했습니다.
def _read_name_set(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    names = set()
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        # bad.txt가 "stem<TAB>reason<TAB>path" 여도 첫 토큰만 사용
        stem = ln.split()[0]
        names.add(stem)
    return names


def _get_names_for_dataset(ds):

    if hasattr(ds, "root_filenames") and ds.root_filenames:
        return list(ds.root_filenames)
    if hasattr(ds, "trajectory_dataset") and hasattr(ds.trajectory_dataset, "filenames"):
        return list(ds.trajectory_dataset.filenames)
    # Fallback: 단일 모달 Dataset
    if hasattr(ds, "filenames"):
        return list(ds.filenames)
    raise AttributeError("Dataset does not expose filenames/root_filenames")


def _set_names_for_dataset(ds, kept):

    if hasattr(ds, "trajectory_dataset") and hasattr(ds.trajectory_dataset, "filenames"):
        ds.trajectory_dataset.filenames = kept
    if hasattr(ds, "root_filenames"):
        ds.root_filenames = kept
    if hasattr(ds, "modality_datasets"):
        for m in ds.modality_datasets.values():
            if hasattr(m, "filenames"):
                m.filenames = kept
    # Fallback
    if hasattr(ds, "filenames"):
        ds.filenames = kept


def _filter_dataset_inplace_multimodal(ds, good: set | None, bad: set | None, tag: str):
    names = _get_names_for_dataset(ds)
    if good is None and bad is None:
        print(f"[split:{tag}] no good/bad lists provided; keep {len(names)}")
        return ds

    kept, dropped = [], []
    seen = set()
    for n in names:
        if not n or n in seen:
            continue
        seen.add(n)
        keep = True
        if good is not None:
            keep = (n in good)
        if bad is not None and (n in bad):
            keep = False
        (kept if keep else dropped).append(n)

    _set_names_for_dataset(ds, kept)
    print(f"[split:{tag}] total={len(names)} kept={len(kept)} dropped={len(dropped)}")
    if dropped[:10]:
        print("  dropped preview:", ", ".join(dropped[:10]))
    return ds
# ---------------------------------------------------------------------------- #


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    OmegaConf.register_new_resolver("eval", eval)
    set_random_seed(config.seed)

    dict_config = OmegaConf.to_container(config, resolve=True)

    if config.log_wandb:
        logger = WandbLogger(
            entity=config.entity,
            project=config.project_name,
            name=config.xp_name,
            save_dir=config.log_dir,
        )
        logger._wandb_init.update({"config": dict_config})
    else:
        logger = None

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=config.save_and_sample_every, save_top_k=-1
    )
    fix_nan = FixNANinGrad(monitor=["train/loss"])
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, fix_nan, lr_monitor]

    # Optional: enable detailed gradient logging via env flag for quick debugging
    # Example: LOG_GRADS=1 python -m src.train num_train_epochs=1 batch_size=1
    import os
    if os.getenv("LOG_GRADS", "0").lower() in {"1", "true", "yes"}:
        callbacks.append(GradLogger(per_param=True, limit_params=None, log_global_norm=True))

    trainer = instantiate(config.trainer)(logger=logger, callbacks=callbacks)

    diffuser = instantiate(config.diffuser)

    # When gradient debugging is enabled, also log gradients to Weights & Biases
    # so you can inspect them in the W&B "Gradients" tab.
    if logger is not None:
        if os.getenv("LOG_GRADS", "0").lower() in {"1", "true", "yes"}:
            try:
                # log='all' tracks both gradients and parameters; adjust log_freq as desired
                logger.watch(diffuser, log="all", log_freq=50, log_graph=False)
            except Exception:
                pass

    # 1) 원본 dataset 한 번 생성
    dataset = instantiate(config.dataset)

    # 2) good/bad 경로는 config.splits.good / config.splits.bad에서 읽음(없으면 None)
    good_path = OmegaConf.select(config, "splits.good")
    bad_path  = OmegaConf.select(config, "splits.bad")
    good_set = _read_name_set(good_path)
    bad_set  = _read_name_set(bad_path)

    print(f"[paths] good={good_path} exists={Path(good_path).exists() if good_path else None} "
          f"bad={bad_path} exists={Path(bad_path).exists() if bad_path else None}")
    print(f"[lists] good_n={len(good_set) if good_set is not None else None} "
          f"bad_n={len(bad_set) if bad_set is not None else None}")

    
    train_ds = deepcopy(dataset).set_split("train")
    _filter_dataset_inplace_multimodal(train_ds, good_set, bad_set, tag="train")

    eval_ds = deepcopy(dataset).set_split("test")
    _filter_dataset_inplace_multimodal(eval_ds, good_set, bad_set, tag="test")

    datamodule = Datamodule(
        train_ds,
        eval_ds,
        config.batch_size,
        config.compnode.num_workers,
    )

    trainer.fit(model=diffuser, datamodule=datamodule, ckpt_path=config.checkpoint_path)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
