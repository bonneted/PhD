import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from phd.config import load_config
from phd.models.cm import deep_notched


DEFAULT_PROJECT = "deep-notched"
DEFAULT_ENTITY = None


def _set_nested(cfg, key, value):
    OmegaConf.update(cfg, key, value, force_add=True)


def _to_float_scalar(value: Any):
    try:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return float(arr)
    except (TypeError, ValueError):
        return None
    return None


def _material_param_names(cfg):
    law = str(cfg.problem.material.law).lower()
    if law == "isotropic":
        return ["E", "nu"]
    if law == "orthotropic":
        return ["Q11", "Q22", "Q12", "Q66"]
    return []   


def _log_metrics_only(results):
    losshistory = results.get("losshistory")
    runtime = results.get("runtime_metrics", {})
    summary = {
        "elapsed_time_s": float(runtime.get("elapsed_time", np.nan)),
        "iterations_per_sec": float(runtime.get("iterations_per_sec", np.nan)),
    }

    if losshistory is not None:
        if hasattr(losshistory, "metrics_test") and len(losshistory.metrics_test) > 0:
            metric_last = losshistory.metrics_test[-1]
            if hasattr(metric_last, "__iter__"):
                summary["l2_relative_error"] = float(metric_last[0])
            else:
                summary["l2_relative_error"] = float(metric_last)

        if hasattr(losshistory, "loss_train") and len(losshistory.loss_train) > 0:
            summary["final_loss_train"] = float(np.sum(losshistory.loss_train[-1]))

        if hasattr(losshistory, "loss_test") and len(losshistory.loss_test) > 0:
            summary["final_loss_test"] = float(np.sum(losshistory.loss_test[-1]))

    cfg = results.get("config")
    task_type = str(cfg.task.type).lower() if cfg is not None else "forward"
    if task_type == "inverse":
        var_cb = results.get("callbacks", {}).get("variable_value")
        if var_cb is not None and getattr(var_cb, "history", None):
            final_row = var_cb.history[-1]
            names = _material_param_names(cfg)
            values = final_row[1:]
            for name, value in zip(names, values):
                scalar_value = _to_float_scalar(value)
                if scalar_value is not None:
                    summary[f"identified_{name}"] = scalar_value

    wandb.log(summary)


def _build_cfg_from_wandb(base_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    for key, value in wandb.config.items():
        _set_nested(cfg, key, value)
    if not isinstance(cfg, DictConfig):
        raise TypeError("Expected DictConfig after applying W&B overrides.")
    return cfg


def _set_wandb_dirs(study_name):
    project_root = Path(__file__).resolve().parents[3]
    sweep_root = project_root / "results" / "deep_notched" / "sweep" / study_name
    cache_root = sweep_root / "wandb_cache"
    sweep_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(sweep_root)
    os.environ["WANDB_DATA_DIR"] = str(cache_root)
    return sweep_root


def run_one_study(study_name, project=DEFAULT_PROJECT, entity=DEFAULT_ENTITY):
    _set_wandb_dirs(study_name)
    with wandb.init(project=project, entity=entity):
        base_cfg = load_config("deep_notched")
        cfg = _build_cfg_from_wandb(base_cfg)
        results = deep_notched.train(cfg)
        _log_metrics_only(results)


def main():
    parser = argparse.ArgumentParser(description="Run deep-notched W&B sweep agent.")
    parser.add_argument("--study", required=True, help="Study name, e.g. iso_forward")
    parser.add_argument("--sweep-id", required=True, help="W&B sweep ID")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project name")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="W&B entity (optional)")
    parser.add_argument("--count", type=int, default=None, help="Number of runs for this agent")
    args = parser.parse_args()

    _set_wandb_dirs(args.study)
    wandb.agent(
        args.sweep_id,
        function=lambda: run_one_study(args.study, project=args.project, entity=args.entity),
        project=args.project,
        count=args.count,
    )


if __name__ == "__main__":
    main()
