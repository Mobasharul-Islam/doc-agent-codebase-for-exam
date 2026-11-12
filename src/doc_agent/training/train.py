"""Training — unified entrypoint"""
from __future__ import annotations

from ..contracts import *  # noqa


def main(component: str, cfg: dict) -> None:
    """Train one component with a seeded Lightning Trainer + W&B (or CSV) logger, save its weights
    under models/ and register them. Usage: python -c "from doc_agent import config;
    from doc_agent.training import train; train.main('enhancer', config.load())"."""
    import lightning as L
    import torch

    from ..ingest.loader import data_paths
    from ..mlops import registry
    from ..settings import settings
    from .datamodule import DocDataModule
    from .lit_modules import LitComponent

    L.seed_everything(cfg.get("seed", 42), workers=True)
    if settings.wandb_api_key:
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(project="doc-agent", config=cfg, tags=[component])
    else:
        from lightning.pytorch.loggers import CSVLogger
        logger = CSVLogger(str(data_paths(cfg)["reports_dir"] / "lightning"), name=component)
    lit = LitComponent(component, cfg)
    trainer = L.Trainer(max_epochs=cfg.get("train", {}).get("epochs", 3), logger=logger,
                        accelerator="auto", devices=1, enable_checkpointing=False,
                        deterministic=False, log_every_n_steps=5)
    trainer.fit(lit, datamodule=DocDataModule(cfg))
    out = data_paths(cfg)["models_dir"] / f"{component}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lit.net.state_dict(), out)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    registry.register(component, str(out), metrics)
