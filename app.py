from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.app_types import BaseSelfLearnedApp
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class GmmAppConfig(AppConfig):
    n_components: int = 3
    covariance_type: str = "full"  # full|tied|diag|spherical
    tol: float = 0.001
    reg_covar: float = 1e-6
    max_iter: int = 100
    n_init: int = 1
    init_params: str = "kmeans"  # kmeans|random
    random_state: int = 42
    warm_start: bool = 0
    standardize: bool = 1
    output_labels: bool = 1
    output_responsibilities: bool = 1


@dataclass
class GmmAppInputConfig(AppInputConfig):
    features: Any = None


@dataclass
class GmmAppOutputConfig(AppOutputConfig):
    labels: Path | None = None
    responsibilities: Path | None = None
    weights: Path | None = None
    means: Path | None = None
    covariances: Path | None = None
    report: str | None = None


class GmmAPP(BaseSelfLearnedApp[GmmAppConfig, GmmAppInputConfig, GmmAppOutputConfig]):

    def run_algorithm(self, data: GmmAppInputConfig) -> GmmAppOutputConfig:
        df: pd.DataFrame = data.features
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'features' must be a pandas DataFrame.")

        ids, feats = _split_id_and_features(df)
        X = _ensure_numeric_matrix(feats)
        X = _maybe_standardize(X, self.config.standardize)

        model = GaussianMixture(
            n_components=int(self.config.n_components),
            covariance_type=str(self.config.covariance_type),
            tol=float(self.config.tol),
            reg_covar=float(self.config.reg_covar),
            max_iter=int(self.config.max_iter),
            n_init=int(self.config.n_init),
            init_params=str(self.config.init_params),
            random_state=int(self.config.random_state),
            warm_start=bool(self.config.warm_start),
        )

        self.logger.info(
            f"Fitting GaussianMixture on {X.shape[0]} samples with {X.shape[1]} features")
        model.fit(X)
        self.logger.info(
            f"Model fitting complete (converged={bool(model.converged_)}, n_iter={int(model.n_iter_)})")

        # Responsibilities and labels
        resp = model.predict_proba(X)
        labels = resp.argmax(axis=1)

        out_labels_path = Path("labels.csv")
        out_resp_path = Path("responsibilities.csv")
        out_weights_path = Path("weights.csv")
        out_means_path = Path("means.csv")
        out_cov_path = Path("covariances.csv")

        if bool(self.config.output_labels):
            _write_labels_csv(out_labels_path, ids, labels)

        if bool(self.config.output_responsibilities):
            if ids is None:
                resp_df = pd.DataFrame(resp,
                                       columns=[f"p_component_{k}" for k in range(resp.shape[1])])
                resp_df.insert(0, "index", np.arange(resp.shape[0]))
            else:
                resp_df = pd.DataFrame(resp,
                                       columns=[f"p_component_{k}" for k in range(resp.shape[1])])
                resp_df.insert(0, "id", ids.values)
            resp_df.to_csv(out_resp_path, index=False)

        # Parameters
        pd.DataFrame(
            {"component": np.arange(model.weights_.shape[0]), "weight": model.weights_}
        ).to_csv(out_weights_path, index=False)

        pd.DataFrame(model.means_, columns=list(feats.columns)).assign(
            component=np.arange(model.means_.shape[0])
        )[["component"] + list(feats.columns)].to_csv(out_means_path, index=False)

        cov_type = str(self.config.covariance_type)
        cov = model.covariances_

        # Serialize covariances in a stable long format
        rows = []
        if cov_type == "full":
            # cov shape: (K, D, D)
            K, D, _ = cov.shape
            for k in range(K):
                for i in range(D):
                    for j in range(D):
                        rows.append({"component": k, "i": i, "j": j, "value": float(cov[k, i, j])})
        elif cov_type == "tied":
            # cov shape: (D, D)
            D, _ = cov.shape
            for i in range(D):
                for j in range(D):
                    rows.append({"i": i, "j": j, "value": float(cov[i, j])})
        elif cov_type == "diag":
            # cov shape: (K, D)
            K, D = cov.shape
            for k in range(K):
                for i in range(D):
                    rows.append({"component": k, "feature": str(feats.columns[i]),
                                 "value": float(cov[k, i])})
        elif cov_type == "spherical":
            # cov shape: (K,)
            for k in range(cov.shape[0]):
                rows.append({"component": k, "value": float(cov[k])})
        else:
            raise ValueError(f"Unsupported covariance_type: {cov_type}")

        pd.DataFrame(rows).to_csv(out_cov_path, index=False)

        # Metrics
        unique, counts = np.unique(labels, return_counts=True)
        for k, c in zip(unique.tolist(), counts.tolist()):
            self.send_metric("cluster_size", float(c), float(k))
        self.send_metric("aic", float(model.aic(X)), 0.0)
        self.send_metric("bic", float(model.bic(X)), 0.0)
        self.send_metric("lower_bound", float(model.lower_bound_), 0.0)

        report = _html_report(
            "GaussianMixture Report",
            params={
                "n_components": self.config.n_components,
                "covariance_type": cov_type,
                "tol": self.config.tol,
                "reg_covar": self.config.reg_covar,
                "max_iter": self.config.max_iter,
                "n_init": self.config.n_init,
                "init_params": self.config.init_params,
                "random_state": self.config.random_state,
                "warm_start": bool(self.config.warm_start),
                "standardize": bool(self.config.standardize),
                "output_labels": bool(self.config.output_labels),
                "output_responsibilities": bool(self.config.output_responsibilities),
            },
            extra={
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "aic": float(model.aic(X)),
                "bic": float(model.bic(X)),
                "lower_bound": float(model.lower_bound_),
                "n_output_clusters": int(len(np.unique(labels))),
            },
        )

        return GmmAppOutputConfig(
            labels=out_labels_path if bool(self.config.output_labels) else None,
            responsibilities=out_resp_path if bool(self.config.output_responsibilities) else None,
            weights=out_weights_path,
            means=out_means_path,
            covariances=out_cov_path,
            report=report,
        )


def _split_id_and_features(df: pd.DataFrame) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """
    If first column is named 'id' (case-insensitive), treat it as identifier.
    Otherwise return None and full df as features.
    """
    if df.shape[1] == 0:
        raise ValueError("Input dataframe has no columns.")

    first_col = str(df.columns[0]).strip().lower()
    if first_col == "id":
        ids = df.iloc[:, 0]
        feats = df.iloc[:, 1:]
        return ids, feats
    return None, df


def _ensure_numeric_matrix(feats: pd.DataFrame) -> np.ndarray:
    # Will raise if cannot convert
    X = feats.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Feature matrix must be 2D with at least one feature column.")
    return X


def _maybe_standardize(X: np.ndarray, standardize: bool) -> np.ndarray:
    if bool(standardize):
        return StandardScaler().fit_transform(X)
    return X


def _write_labels_csv(path: Path, ids: Optional[pd.Series], labels: np.ndarray) -> None:
    if ids is None:
        out = pd.DataFrame({"index": np.arange(len(labels)), "label": labels})
    else:
        out = pd.DataFrame({"id": ids.values, "label": labels})
    out.to_csv(path, index=False)


def _html_report(title: str, params: dict, extra: dict) -> str:
    def _rows(d: dict) -> str:
        return "\n".join(
            f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"
            for k, v in d.items()
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 980px; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; text-align: left; }}
    code {{ background: #f1f1f1; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h2>{title}</h2>

  <h3>Parameters</h3>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    {_rows(params)}
  </table>

  <h3>Summary</h3>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    {_rows(extra)}
  </table>
</body>
</html>
"""
