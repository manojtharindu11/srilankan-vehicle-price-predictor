# Sri Lankan Used Car Price Predictor

Predict Sri Lankan used car prices from listing attributes (make, model, year, mileage, transmission, fuel type, engine CC, and simple condition flags extracted from ad text).

## What’s in this repo

- **Data collection (optional)**: scrape listings and save a CSV dataset.
- **Model building**: a notebook that cleans data, engineers features, trains an XGBoost regression model, and exports artifacts.
- **Frontend**: a Streamlit app that loads the exported artifacts and serves predictions + SHAP explanations.

## Repository structure

```
.
├─ collect data/
│  ├─ scrap.py
│  └─ data/
│     └─ sri_lankan_vehicles.csv
├─ data/
│  └─ ml_ready_vehicles.csv
├─ frontend/
│  ├─ app.py
│  └─ requirements.txt
├─ model/
│  ├─ preprocess_ohe.pkl
│  └─ xgb_model.pkl
├─ model building/
│  └─ full_pipeline.ipynb
├─ main.py
├─ requirements.txt
└─ Dockerfile
```

## Quickstart (run the app locally)

### 1) Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Make sure model artifacts exist

The Streamlit app requires these files:

- `model/preprocess_ohe.pkl`
- `model/xgb_model.pkl`

If they don’t exist yet, run the training notebook:

- Open `model building/full_pipeline.ipynb`
- Run all cells

This will generate and save the artifacts into the `model/` folder.

### 3) Start Streamlit

```powershell
streamlit run main.py
```

Then open: http://localhost:8501

## Run with Docker (frontend)

This Docker setup runs the Streamlit frontend via `main.py`.

### Build

```bash
docker build -t sl-vehicle-frontend .
```

### Run

```bash
docker run --rm -p 8501:8501 sl-vehicle-frontend
```

Open: http://localhost:8501

Notes:

- The image expects the model artifacts (`model/preprocess_ohe.pkl`, `model/xgb_model.pkl`) to be present in the repository at build time.
- The Docker image installs `libgomp1` because `xgboost` requires the OpenMP runtime on Debian-based images.

## Training & evaluation

Training is done in `model building/full_pipeline.ipynb`.

The notebook:

- Cleans and normalizes listing fields
- Removes obvious outliers
- One-hot encodes categorical features via a `ColumnTransformer`
- Trains an `XGBRegressor` model
- Includes checks to reduce overfitting (train/validation/test metrics, cross-validation, learning curve)
- Exports the fitted preprocessor and model to `model/`

If you retrain, commit or copy the regenerated artifacts so the frontend uses the latest model.

## Data

The notebook reads the scraped dataset from:

- `collect data/data/sri_lankan_vehicles.csv`

During cleaning it also writes a cleaned dataset to:

- `data/ml_ready_vehicles.csv`

If you don’t want to scrape, you can drop a CSV at the expected path with the required columns.

## Troubleshooting

- **“Missing model artifacts” in the app**: run `model building/full_pipeline.ipynb` to generate `model/preprocess_ohe.pkl` and `model/xgb_model.pkl`.
- **Package conflicts**: use the pinned dependencies in `frontend/requirements.txt` (the root `requirements.txt` includes it).
- **Docker build succeeds but predictions fail**: verify the artifacts were generated with compatible versions (especially `scikit-learn`, `xgboost`, and `shap`).

## Tech stack

- Python 3.11
- Streamlit (UI)
- scikit-learn (preprocessing)
- XGBoost (regression model)
- SHAP (model explanations)
