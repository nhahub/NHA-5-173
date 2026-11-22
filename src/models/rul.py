"""RUL prediction models (XGBoost baseline, LSTM, GRU, 1D-CNN) with uncertainty estimation."""
from __future__ import annotations

from dataclasses import dataclass
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# 1. Tabular Baseline + Quantile Uncertainty: XGBoost

class XGBoostRULModel:
    """XGBoost regressor with lower/upper quantile models for RUL uncertainty."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
        )
        self.point_model = xgb.XGBRegressor(objective="reg:squarederror", **self.params)
        self.lower_model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.10, **self.params
        )
        self.upper_model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.90, **self.params
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.point_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.lower_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.upper_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.point_model.predict(X), 0.0, 125.0)

    def predict_with_uncertainty(self, X: np.ndarray) -> dict[str, np.ndarray]:
        mean_pred = self.predict(X)
        low_pred = np.clip(self.lower_model.predict(X), 0.0, 125.0)
        high_pred = np.clip(self.upper_model.predict(X), 0.0, 125.0)
        # Ensure monotonicity: lower <= mean <= upper
        lower = np.minimum(low_pred, mean_pred)
        upper = np.maximum(high_pred, mean_pred)
        # Approximate std from 80% inter-quantile range (z_0.90 ≈ 1.2816)
        std = (upper - lower) / (2.0 * 1.2816)
        return {
            "rul_mean": mean_pred,
            "rul_std": std,
            "rul_lower": lower,
            "rul_upper": upper,
        }

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "XGBoostRULModel":
        return joblib.load(path)

# 2. Sequence Deep Learning Models: LSTM, GRU, 1D-CNN + MC Dropout Uncertainty

class SequenceRULNet(nn.Module):
    """Unified PyTorch architecture supporting 'lstm', 'gru', and 'cnn1d'."""

    def __init__(
        self,
        arch: str,
        input_dim: int = 62,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout

        if self.arch == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            feat_dim = hidden_dim
        elif self.arch == "gru":
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            feat_dim = hidden_dim
        elif self.arch == "cnn1d":
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            feat_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported architecture: {arch!r}. Use 'lstm', 'gru', or 'cnn1d'.")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len=30, features=62)
        if self.arch in ("lstm", "gru"):
            out, _ = self.encoder(x)
            h = out[:, -1, :]  # last time step
        else:  # cnn1d expects (batch, channels, seq_len)
            out = self.encoder(x.transpose(1, 2))
            h = out.squeeze(-1)
        return self.head(h).squeeze(-1)


@dataclass
class DLTrainConfig:
    arch: str = "lstm"
    input_dim: int = 62
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 35
    patience: int = 6
    mc_samples: int = 30
    seed: int = 42


class SequenceRULTrainer:
    """Wrapper for training, saving, and MC-Dropout uncertainty inference."""

    def __init__(self, config: DLTrainConfig):
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SequenceRULNet(
            arch=config.arch,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(self.device)
        self.history: list[dict[str, float]] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        criterion = nn.MSELoss()

        best_val_rmse = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_preds = self.predict(X_val)
            val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
            train_rmse = float(np.sqrt(np.mean(train_losses)))
            self.history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        xb = torch.tensor(X, dtype=torch.float32, device=self.device)
        preds = self.model(xb).cpu().numpy()
        return np.clip(preds, 0.0, 125.0)

    @torch.no_grad()
    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int | None = None) -> dict[str, np.ndarray]:
        """Estimate predictive mean and epistemic uncertainty via Monte Carlo Dropout."""
        samples_count = n_samples or self.config.mc_samples
        self.model.train()  # keep Dropout active at test time
        xb = torch.tensor(X, dtype=torch.float32, device=self.device)

        mc_preds = np.stack(
            [np.clip(self.model(xb).cpu().numpy(), 0.0, 125.0) for _ in range(samples_count)],
            axis=0,
        )
        mean_pred = mc_preds.mean(axis=0)
        std_pred = mc_preds.std(axis=0)
        # 90% confidence interval (z = 1.645)
        lower = np.clip(mean_pred - 1.645 * std_pred, 0.0, 125.0)
        upper = np.clip(mean_pred + 1.645 * std_pred, 0.0, 125.0)
        self.model.eval()
        return {
            "rul_mean": mean_pred,
            "rul_std": std_pred,
            "rul_lower": lower,
            "rul_upper": upper,
        }

    def save(self, path: str):
        torch.save(
            {
                "config": self.config.__dict__,
                "state_dict": self.model.state_dict(),
                "history": self.history,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "SequenceRULTrainer":
        ckpt = torch.load(path, map_location="cpu")
        trainer = SequenceRULTrainer(DLTrainConfig(**ckpt["config"]))
        trainer.model.load_state_dict(ckpt["state_dict"])
        trainer.history = ckpt.get("history", [])
        trainer.model.eval()
        return trainer