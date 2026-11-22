"""Unified training entry point for HexaMind RUL models (XGBoost, LSTM, GRU, 1D-CNN)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import WINDOW, create_windows  # noqa: E402
from models.rul import DLTrainConfig, SequenceRULTrainer, XGBoostRULModel  # noqa: E402
from validation import get_group_kfold_splits  # noqa: E402

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"


def compute_metrics(y_true: np.ndarray, unc_dict: dict[str, np.ndarray]) -> dict[str, float]:
    y_pred = unc_dict["rul_mean"]
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    coverage = float(
        np.mean((y_true >= unc_dict["rul_lower"]) & (y_true <= unc_dict["rul_upper"]))
    )
    mean_width = float(np.mean(unc_dict["rul_upper"] - unc_dict["rul_lower"]))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "pi90_coverage": round(coverage, 4),
        "pi90_mean_width": round(mean_width, 2),
    }


def evaluate_last_cycle_per_engine(
    meta_df: pd.DataFrame, y_true: np.ndarray, unc_dict: dict[str, np.ndarray]
) -> dict[str, float]:
    """Standard C-MAPSS benchmark: evaluate at the final observed cycle of each test engine."""
    temp = meta_df.copy()
    temp["y_true"] = y_true
    temp["rul_mean"] = unc_dict["rul_mean"]
    temp["rul_lower"] = unc_dict["rul_lower"]
    temp["rul_upper"] = unc_dict["rul_upper"]
    last = temp.sort_values(["unit_id", "end_cycle"]).groupby("unit_id", as_index=False).last()
    return compute_metrics(
        last["y_true"].to_numpy(),
        {
            "rul_mean": last["rul_mean"].to_numpy(),
            "rul_lower": last["rul_lower"].to_numpy(),
            "rul_upper": last["rul_upper"].to_numpy(),
        },
    )


def load_feature_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    order_path = ARTIFACTS_DIR / "feature_order.json"
    if not order_path.is_file():
        raise FileNotFoundError("Missing artifacts/feature_order.json; run preprocessing.py first.")
    feature_cols = json.loads(order_path.read_text(encoding="utf-8"))

    frames = {}
    for split in ("train", "validation", "test"):
        path = FEATURES_DIR / f"{split}_features.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run `python src/features.py` first.")
        frames[split] = pd.read_csv(path)
    return frames["train"], frames["validation"], frames["test"], feature_cols


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 42,
) -> dict:
    print("\n=== Training XGBoost Baseline (with Quantile Uncertainty) ===")
    X_train, y_train = train_df[feature_cols].to_numpy(dtype=np.float32), train_df["RUL"].to_numpy(dtype=np.float32)
    X_val, y_val = val_df[feature_cols].to_numpy(dtype=np.float32), val_df["RUL"].to_numpy(dtype=np.float32)
    X_test, y_test = test_df[feature_cols].to_numpy(dtype=np.float32), test_df["RUL"].to_numpy(dtype=np.float32)

    # 5-Fold GroupKFold CV on training engines
    cv_rmses = []
    for fold, (tr_idx, va_idx) in enumerate(get_group_kfold_splits(train_df, n_splits=5), start=1):
        fold_model = XGBoostRULModel(random_state=seed)
        fold_model.fit(X_train[tr_idx], y_train[tr_idx], X_train[va_idx], y_train[va_idx])
        fold_preds = fold_model.predict(X_train[va_idx])
        cv_rmses.append(float(np.sqrt(np.mean((fold_preds - y_train[va_idx]) ** 2))))

    model = XGBoostRULModel(random_state=seed)
    model.fit(X_train, y_train, X_val, y_val)

    val_metrics = compute_metrics(y_val, model.predict_with_uncertainty(X_val))
    test_all_metrics = compute_metrics(y_test, model.predict_with_uncertainty(X_test))

    test_meta = test_df[["unit_id", "cycle"]].rename(columns={"cycle": "end_cycle"})
    test_last_metrics = evaluate_last_cycle_per_engine(
        test_meta, y_test, model.predict_with_uncertainty(X_test)
    )

    out_path = MODELS_DIR / "xgboost_rul.joblib"
    model.save(str(out_path))
    print(f"5-Fold CV RMSE: {np.mean(cv_rmses):.4f} ± {np.std(cv_rmses):.4f}")
    print(f"Validation: {val_metrics}")
    print(f"Test (Last Cycle): {test_last_metrics}")

    return {
        "model": "xgboost",
        "hyperparameters": model.params,
        "cv_rmse_mean": round(float(np.mean(cv_rmses)), 4),
        "cv_rmse_std": round(float(np.std(cv_rmses)), 4),
        "validation": val_metrics,
        "test_all_cycles": test_all_metrics,
        "test_last_cycle": test_last_metrics,
        "artifact": str(out_path.relative_to(PROJECT_ROOT)),
    }


def train_deep_model(
    arch: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    epochs: int = 35,
    seed: int = 42,
) -> dict:
    print(f"\n=== Training {arch.upper()} Sequence Model (with MC Dropout Uncertainty) ===")
    keep_cols = ["unit_id", "cycle", "RUL"] + feature_cols
    _, X_train, y_train = create_windows(train_df[keep_cols], window=WINDOW)
    _, X_val, y_val = create_windows(val_df[keep_cols], window=WINDOW)
    test_meta, X_test, y_test = create_windows(test_df[keep_cols], window=WINDOW)

    config = DLTrainConfig(
        arch=arch,
        input_dim=len(feature_cols),
        hidden_dim=64,
        num_layers=2,
        dropout=0.25,
        lr=1e-3,
        batch_size=64,
        epochs=epochs,
        patience=6,
        mc_samples=30,
        seed=seed,
    )
    trainer = SequenceRULTrainer(config)
    trainer.fit(X_train, y_train.to_numpy(), X_val, y_val.to_numpy())

    val_metrics = compute_metrics(y_val.to_numpy(), trainer.predict_with_uncertainty(X_val))
    test_unc = trainer.predict_with_uncertainty(X_test)
    test_all_metrics = compute_metrics(y_test.to_numpy(), test_unc)
    test_last_metrics = evaluate_last_cycle_per_engine(test_meta, y_test.to_numpy(), test_unc)

    out_path = MODELS_DIR / f"{arch.lower()}_rul.pt"
    trainer.save(str(out_path))
    print(f"Validation: {val_metrics}")
    print(f"Test (Last Cycle): {test_last_metrics}")

    return {
        "model": arch.lower(),
        "hyperparameters": config.__dict__,
        "validation": val_metrics,
        "test_all_cycles": test_all_metrics,
        "test_last_cycle": test_last_metrics,
        "artifact": str(out_path.relative_to(PROJECT_ROOT)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train HexaMind RUL prediction models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "lstm", "gru", "cnn1d"],
        choices=["xgboost", "lstm", "gru", "cnn1d"],
        help="Models to train (default: all four).",
    )
    parser.add_argument("--epochs", type=int, default=35, help="Max epochs for DL models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df, feature_cols = load_feature_frames()

    results = []
    if "xgboost" in args.models:
        results.append(train_xgboost(train_df, val_df, test_df, feature_cols, seed=args.seed))
    for arch in ("lstm", "gru", "cnn1d"):
        if arch in args.models:
            results.append(
                train_deep_model(arch, train_df, val_df, test_df, feature_cols, epochs=args.epochs, seed=args.seed)
            )

    best = min(results, key=lambda r: r["validation"]["rmse"])
    summary = {
        "best_model_by_val_rmse": best["model"],
        "window_size": WINDOW,
        "num_features": len(feature_cols),
        "rul_cap": 125,
        "runs": results,
    }
    summary_path = MODELS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nTraining complete. Best validation model: {best['model'].upper()} (Val RMSE = {best['validation']['rmse']})")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()