An end-to-end machine learning project exploring **unsupervised learning** with the Wine Quality dataset. The project includes exploratory data analysis (EDA), an ETL pipeline, and a K-Means clustering workflow using feature scaling and Principal Component Analysis (PCA).

> **Project status:** ETL pipeline and initial clustering model implemented. Automated testing, GitHub Actions CI, and MLflow experiment tracking are planned next.

## Table of Contents

- [Project Overview](#project-overview)
- [Goals](#goals)
- [Dataset](#dataset)
- [Current Workflow](#current-workflow)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Run the ETL Pipeline](#run-the-etl-pipeline)
- [Train the Clustering Model](#train-the-clustering-model)
- [Test the Saved Model](#test-the-saved-model)
- [Current Experiment](#current-experiment)
- [Next Steps](#next-steps)
- [Notes](#notes)

## Project Overview

The goal is to explore whether wines can be grouped by their chemical measurements without using the quality score to train the clustering model.

The project currently focuses on white wine. Red wine data and notebooks are also included for separate exploration.

## Goals

- Explore and understand the Wine Quality dataset through EDA.
- Build a modular Extract, Transform, Load (ETL) pipeline.
- Apply `StandardScaler`, PCA, and K-Means clustering.
- Save and reload the trained scikit-learn pipeline.
- Add automated tests and continuous integration with GitHub Actions.
- Introduce MLflow to track experiment parameters, metrics, and artifacts.

## Dataset

This project uses the UCI Wine Quality dataset, which contains red and white Portuguese *Vinho Verde* wine samples.

The datasets include physicochemical measurements and a quality score.

- `data/winequality-white.csv`
- `data/winequality-red.csv`
- `data/winequality.names` (dataset description, if included)

The white wine dataset contains **4,898 rows and 12 columns**: 11 physicochemical input features and one `quality` column.

**Important:** `quality` is excluded from the clustering input features. It may be used after clustering to help describe or compare the resulting groups; it is not used to train the unsupervised model.

Dataset source: [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

## Current Workflow

### 1. ETL

The ETL code lives in `src/` and is organized into separate responsibilities:

- **Extract (`extract.py`):** Reads a source CSV into a pandas DataFrame.
- **Transform (`transform.py`):** Validates the expected columns and standardizes column names to lowercase snake case.
- **Load (`load.py`):** Writes the transformed DataFrame to a processed CSV file.
- **Orchestration (`run_etl.py` or `main.py`):** Calls the extraction, transformation, and loading steps in sequence.

The current white wine output is saved as:

```text
processed/white_wine_processed.csv
```

### 2. Clustering

The current scikit-learn pipeline performs these steps:

1. `StandardScaler` scales the input features.
2. `PCA(n_components=5)` reduces the features to five principal components.
3. `KMeans(n_clusters=2, random_state=42, n_init=10)` assigns wines to two clusters.

The trained pipeline is saved with Joblib at:

```text
models/wine_clustering.joblib
```

The saved pipeline includes preprocessing and clustering, so new samples can be passed through the same scaling and PCA transformations before cluster prediction.

## Project Structure

The project is organized approximately as follows (actual filenames may vary slightly):

```text
wine_quality_project/
├── data/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── winequality.names
├── models/
│   └── wine_clustering.joblib
├── notebooks/
│   ├── red_wine.ipynb
│   └── white_wine.ipynb
├── processed/
│   └── white_wine_processed.csv
├── src/
│   ├── extract.py
│   ├── transform.py
│   ├── load.py
│   └── run_etl.py
├── tests/                  # Planned: automated tests
├── train.py
├── test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Tech Stack

- **Python** — project language
- **pandas** — data loading and manipulation
- **scikit-learn** — scaling, PCA, and K-Means
- **Joblib** — saving and loading the trained pipeline
- **Jupyter Notebook** — exploratory data analysis
- **Git and GitHub** — version control and project hosting

Planned: **pytest**, **GitHub Actions**, and **MLflow**.

## Setup

### 1. Clone the repository

After publishing the repository, clone it with:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd wine_quality_project
```

Replace the placeholder with your repository URL.

### 2. Create and activate a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the ETL Pipeline

Run commands from the **project root**. Since the ETL modules are inside `src/`, use the command that matches your orchestration filename.

If the file is `src/run_etl.py`:

```bash
python -m src.run_etl
```

If the file is `src/main.py`:

```bash
python -m src.main
```

These module-style commands assume `src` is importable as a Python package (for example, by adding an empty `src/__init__.py`) and that imports are compatible with package execution. If the project currently uses a different import setup, use the command that you've verified works locally and document it here.

The pipeline should read the white wine CSV, validate and transform the data, and save the processed output under `processed/`.

## Train the Clustering Model

From the project root:

```bash
python train.py
```

This script is expected to load the prepared or source data, fit the scikit-learn pipeline, and save the trained model to `models/wine_clustering.joblib`.

Make sure the training script's input path and column names match the output of the ETL pipeline. In particular, the `quality` column must not be included in the clustering features.

## Test the Saved Model

From the project root:

```bash
python test_model.py
```

This script loads the saved pipeline and checks that it can predict cluster assignments for wine samples.

## Current Experiment

The current white wine clustering configuration is:

| Setting | Value |
|---|---:|
| Dataset | White wine |
| Number of input features | 11 |
| PCA components | 5 |
| Number of clusters | 2 |
| K-Means `random_state` | 42 |
| K-Means `n_init` | 10 |
| Number of samples | 4,898 |

An earlier run assigned **1,972 samples to cluster 0** and **2,926 samples to cluster 1**. These counts describe the observed run and should be rechecked if the dataset, preprocessing, or model settings change.

The clusters are **not wine-quality grades**. They represent groups formed from chemical measurements. In earlier analysis, one cluster tended to have higher residual sugar and total sulfur dioxide, higher density, and lower alcohol; the other tended to show the opposite pattern. These are descriptive tendencies, not guarantees for every wine in a cluster.

## Next Steps

- [ ] Finish and verify the ETL modules.
- [ ] Add automated tests with `pytest`.
- [ ] Publish the repository on GitHub.
- [ ] Add a GitHub Actions workflow to run tests automatically.
- [ ] Add MLflow experiment tracking for parameters, metrics, and artifacts.
- [ ] Compare experiments systematically and document model evaluation.
- [ ] Consider serving predictions through an API after the pipeline is tested.

## Notes

- The ETL pipeline standardizes column names, so training and prediction inputs must use consistent feature names.
- Scaling and PCA are part of the saved scikit-learn pipeline. Do not independently fit new scalers or PCA transformations when predicting with the saved model.
- A successful run and balanced-looking cluster sizes do not by themselves prove that a clustering model is useful. Evaluate cluster separation, stability, and interpretability.
- The repository should not contain secrets, virtual environments, or unnecessary generated files. Check the dataset, notebook, and model file sizes before pushing to GitHub.
"""
path = Path("/mnt/data/README.md")
path.write_text(content, encoding="utf-8")
print(f"Created {path}")

