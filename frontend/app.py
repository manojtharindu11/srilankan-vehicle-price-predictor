from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


@st.cache_resource
def load_artifacts():
    root_dir = Path(__file__).resolve().parents[1]
    model_dir = root_dir / "model"

    preprocess_path = model_dir / "preprocess_ohe.pkl"
    model_path = model_dir / "xgb_model.pkl"

    if not preprocess_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Expected:\n"
            f"- {preprocess_path}\n"
            f"- {model_path}\n\n"
            "Run the training notebook to generate them."
        )

    preprocess = joblib.load(preprocess_path)
    model = joblib.load(model_path)
    return preprocess, model


def get_categories(preprocess) -> dict[str, list[str]]:
    """Extract known categories from the fitted OneHotEncoder."""
    try:
        onehot = preprocess.named_transformers_["cat"].named_steps["onehot"]
        categorical_features = [
            "Make",
            "Model",
            "Gear",
            "FuelType",
        ]
        return {
            feature: [str(x) for x in categories]
            for feature, categories in zip(categorical_features, onehot.categories_)
        }
    except Exception:
        return {}


@st.cache_data
def load_make_model_map() -> dict[str, list[str]]:
    """Build Make -> Models mapping from the cleaned ML dataset.

    This ensures the UI only shows models relevant to a selected make.
    """
    root_dir = Path(__file__).resolve().parents[1]
    dataset_path = root_dir / "data" / "ml_ready_vehicles.csv"
    if not dataset_path.exists():
        return {}

    try:
        df = pd.read_csv(dataset_path, usecols=["Make", "Model"])
    except Exception:
        return {}

    df = df.dropna(subset=["Make", "Model"]).copy()
    df["Make"] = df["Make"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["Model"] = df["Model"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    df = df[(df["Make"] != "") & (df["Model"] != "")]

    # Normalize casing by choosing the most frequent spelling for each value.
    def _mode_string(values: pd.Series) -> str:
        vc = values.astype(str).value_counts()
        return str(vc.index[0]) if len(vc) else ""

    make_key = df["Make"].str.casefold()
    df["Make"] = df.groupby(make_key)["Make"].transform(_mode_string)

    pair_key = df["Make"].str.casefold() + "||" + df["Model"].str.casefold()
    df["Model"] = df.groupby(pair_key)["Model"].transform(_mode_string)

    make_to_models: dict[str, list[str]] = {}
    for make, g in df.groupby("Make", sort=True):
        models = sorted({str(m) for m in g["Model"].dropna().tolist() if str(m).strip()})
        if models:
            make_to_models[str(make)] = models

    return make_to_models


def format_lkr(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"Rs. {value:,.0f}"


def select_or_text(label: str, options: list[str] | None, default: str = "") -> str:
    options = options or []
    if len(options) > 0:
        return st.selectbox(label, options=options)
    return st.text_input(label, value=default)


def select_or_text_keyed(
    label: str,
    options: list[str] | None,
    *,
    key: str,
    default: str = "",
) -> str:
    options = options or []
    if len(options) > 0:
        return st.selectbox(label, options=options, key=key)
    return st.text_input(label, value=default, key=key)


def _clean_feature_name(name: str) -> str:
    # Example: 'num__YOM' -> 'YOM', 'cat__Make_Toyota' -> 'Make=Toyota'
    if "__" in name:
        name = name.split("__", 1)[1]
    if "_" in name and name.count("_") >= 1:
        # For one-hot like Make_Toyota, FuelType_Petrol, Gear_Automatic
        prefix, rest = name.split("_", 1)
        if prefix in {"Make", "Model", "Gear", "FuelType"}:
            return f"{prefix}={rest}"
    return name


@st.cache_resource
def get_shap_explainer(_model):
    # TreeExplainer works well for XGBoost and doesn't require the training data.
    return shap.TreeExplainer(_model)


def explain_prediction(preprocess, model, x_input: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    x_processed = preprocess.transform(x_input)
    explainer = get_shap_explainer(model)

    shap_vals = explainer.shap_values(x_processed)
    base_value = explainer.expected_value

    # Normalize shapes
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).reshape(-1)[0])
    else:
        base_value = float(base_value)

    shap_arr = np.asarray(shap_vals)
    if shap_arr.ndim == 2:
        shap_arr = shap_arr[0]

    try:
        raw_names = preprocess.get_feature_names_out()
        feature_names = [_clean_feature_name(str(n)) for n in raw_names]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(shap_arr.shape[0])]

    contrib = pd.DataFrame({"feature": feature_names, "shap": shap_arr})
    contrib["abs_shap"] = contrib["shap"].abs()
    contrib = contrib.sort_values("abs_shap", ascending=False)
    return base_value, contrib


st.set_page_config(page_title="Sri Lankan Used Car Price Predictor", layout="centered")

st.title("Sri Lankan Used Car Price Predictor")
st.caption("Predict Sri Lankan Used car prices using the trained XGBoost model.")

try:
    preprocess, model = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

categories = get_categories(preprocess)
make_model_map = load_make_model_map()

col1, col2 = st.columns(2)

with col1:
    make_options = sorted(make_model_map.keys()) if make_model_map else (categories.get("Make") or [])
    make = select_or_text_keyed("Make", make_options, key="make")

    model_options = make_model_map.get(make) if make_model_map else None
    if not model_options:
        model_options = categories.get("Model") or []
    model_name = select_or_text_keyed("Model", model_options, key="model")
    yom = st.number_input("Year of Manufacture (YOM)", min_value=1950, max_value=2050, value=2015, step=1)
    engine_cc = st.number_input("Engine CC", min_value=0, max_value=10000, value=1500, step=50)

with col2:
    gear = select_or_text("Gear", categories.get("Gear"))
    fuel_type = select_or_text("Fuel Type", categories.get("FuelType"))
    mileage_km = st.number_input("Mileage (km)", min_value=0.0, max_value=1_000_000.0, value=50_000.0, step=1000.0)

st.subheader("Condition / seller description flags")
flag_col1, flag_col2 = st.columns(2)
with flag_col1:
    is_first_owner = st.checkbox("First owner")
    is_accident_free = st.checkbox("Accident free")
with flag_col2:
    is_mint_condition = st.checkbox("Mint condition")
    has_service_records = st.checkbox("Service records")

if st.button("Predict price"):
    input_row = {
        "Make": make,
        "Model": model_name,
        "YOM": int(yom),
        "MileageKM_Clean": float(mileage_km),
        "Gear": gear,
        "FuelType": fuel_type,
        "EngineCC": float(engine_cc),
        "is_first_owner": int(is_first_owner),
        "is_accident_free": int(is_accident_free),
        "is_mint_condition": int(is_mint_condition),
        "has_service_records": int(has_service_records),
    }

    x_input = pd.DataFrame([input_row])
    x_processed = preprocess.transform(x_input)
    y_pred = float(model.predict(x_processed)[0])

    st.success(f"Predicted price: {format_lkr(y_pred)}")

    with st.expander("Explain this prediction (SHAP)", expanded=True):
        try:
            base_value, contrib = explain_prediction(preprocess, model, x_input)
            st.caption(
                "SHAP shows which inputs increased/decreased the model's output for this one prediction. "
                "Positive SHAP pushes the price up; negative pushes it down."
            )
            st.write(f"Base value (average prediction): {format_lkr(base_value)}")

            top_n = 12
            top = contrib.head(top_n).copy()
            top["direction"] = np.where(top["shap"] >= 0, "increases", "decreases")
            st.dataframe(
                top[["feature", "shap", "direction"]],
                use_container_width=True,
                hide_index=True,
            )

            chart_df = top.set_index("feature")["shap"].sort_values()
            st.bar_chart(chart_df)
        except Exception as exc:
            st.warning(f"Could not compute SHAP explanation: {exc}")
