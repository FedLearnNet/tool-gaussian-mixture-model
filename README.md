# tool-gaussian-mixture-model

Fits Gaussian mixture models and exports cluster labels, membership probabilities, and model parameters.

## Runtime inputs

Supply input files according to the schemas in `app.yml`.

## Inputs and outputs

### Inputs

| Name | Type | Description |
| --- | --- | --- |
| `features` | CSV | Tabular feature matrix. Rows = samples, columns = numeric features. Optional first column 'id'; otherwise indices are used. |

### Outputs

| Name | Type | Description |
| --- | --- | --- |
| `labels` | CSV | Hard cluster assignments per sample. |
| `responsibilities` | CSV | Soft assignments (responsibilities). Rows=samples, columns=components. (Only if output_responsibilities=true). |
| `weights` | CSV | Mixture weights |
| `means` | CSV | Component means (means_). Rows=components, columns=features. |
| `covariances` | CSV | Covariance parameters (covariances_) serialized to CSV depending on covariance_type. |
| `report` | HTML | HTML report with parameters, convergence, AIC/BIC, log-likelihood and cluster-size stats. |

## Configuration

| Name | Default | Description |
| --- | --- | --- |
| `n_components` | `3` | Number of Gaussian components (clusters). |
| `covariance_type` | `full` | Covariance type for each component. |
| `tol` | `0.001` | Convergence threshold for EM (stop when improvement < tol). |
| `reg_covar` | `0.000001` | Non-negative regularization added to the diagonal of covariance. |
| `max_iter` | `100` | Maximum number of EM iterations per initialization. |
| `n_init` | `1` | Number of initializations to perform (best log-likelihood kept). |
| `init_params` | `kmeans` | Initialization strategy. |
| `random_state` | `42` | Random seed for reproducibility. |
| `warm_start` | `0` | If true, reuse the solution of the previous fit as initialization. |
| `standardize` | `1` | Standardize features to zero mean / unit variance before fitting (recommended). |
| `output_labels` | `1` | Output hard labels (argmax of responsibilities). |
| `output_responsibilities` | `1` | Output responsibilities/probabilities per component. |

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Set APP_ID and platform URLs before connecting.
python main.py
```

The Docker base image provides the FL-Net API. `requirements.txt` lists additional
runtime dependencies; `requirements-dev.txt` includes the API for local development.
Runtime datasets and trained models must be supplied separately; see `data/README.md`.

Build with `docker build -t tool-gaussian-mixture-model .`.
