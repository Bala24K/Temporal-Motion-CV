"""
A small temporal 1D-CNN classifier (the "1D CNN" rung of V1 §16's own
suggested model ladder: Random Forest -> XGBoost -> MLP -> 1D CNN ->
GRU/LSTM -> Transformer, tried in that order rather than reaching for
the most exotic architecture first).

Operates on the RAW per-frame skeleton sequence
(skeleton_dataset.generate_sequence_dataset) rather than pre-computed
velocity/acceleration — the model learns temporal patterns itself from
the sequence, which is the actual point of trying a temporal
architecture instead of only engineered features.

Deliberately small: this is a methodology comparison on synthetic data,
not a production model. See recognition/model_comparison.py and
STATUS.md for what any resulting number does and doesn't mean.
"""
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class _CNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, n_classes)

    def forward(self, x):  # x: (batch, F, T)
        z = self.net(x).squeeze(-1)
        return self.head(z)


class TemporalCNNClassifier:
    """Minimal sklearn-like wrapper — fit(X, y) / predict(X) /
    predict_proba(X) — where X is (N, T, F): raw per-frame skeleton
    feature sequences. Lets this drop into the same comparison harness
    and evaluation/classification.py metrics as the sklearn/xgboost
    models without special-casing it."""

    def __init__(self, epochs: int = 40, lr: float = 1e-3, seed: int = 0, batch_size: int = 32):
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.batch_size = batch_size
        self.classes_: List[str] = []
        self._model = None
        self._mean = None
        self._std = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemporalCNNClassifier":
        torch.manual_seed(self.seed)
        self.classes_ = sorted(set(y))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[label] for label in y])

        self._mean = X.mean(axis=(0, 1), keepdims=True)
        self._std = X.std(axis=(0, 1), keepdims=True) + 1e-6
        X_norm = (X - self._mean) / self._std

        X_t = torch.tensor(X_norm, dtype=torch.float32).permute(0, 2, 1)  # (N, F, T)
        y_t = torch.tensor(y_idx, dtype=torch.long)

        n_features = X.shape[2]
        self._model = _CNN(n_features, len(self.classes_))
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(self.seed))

        self._model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._model(xb), yb)
                loss.backward()
                optimizer.step()
        self._model.eval()
        return self

    def _proba(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X - self._mean) / self._std
        X_t = torch.tensor(X_norm, dtype=torch.float32).permute(0, 2, 1)
        with torch.no_grad():
            logits = self._model(X_t)
            return torch.softmax(logits, dim=1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self._proba(X), axis=1)
        return np.array([self.classes_[i] for i in idx])
