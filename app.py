# filename: streamlit_bigmart_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import base64
from datetime import datetime
import plotly.express as px

# -------------------------
# Styling (small CSS)
# -------------------------
st.set_page_config(page_title="🛒 BigMart — World Class Sales Predictor", layout="wide", page_icon="🛒")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#ffffff 0%, #f3f6ff 100%); }
    .big-title { font-size:32px; font-weight:700; }
    .small-muted { color: #6c757d; font-size:12px; }
    .card { background: white; padding: 16px; border-radius: 12px; box-shadow: 0 4px 20px rgba(50,50,93,.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helper: cached model loader
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(path="bigmart_best_model.pkl"):
    """
    Expected pickle format:
      - If you saved (model, sklearn_version) -> returns model, sklearn_version, preprocessor=None
      - If you saved dict with {'model': m, 'preprocessor': p, 'sklearn_version': v} -> returns accordingly
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    # support multiple save formats
    if isinstance(payload, tuple) and len(payload) == 2:
        model, sklearn_version = payload
        preprocessor = None
    elif isinstance(payload, dict):
        model = payload.get("model")
        preprocessor = payload.get("preprocessor")
        sklearn_version = payload.get("sklearn_version", "unknown")
    else:
        # try common pattern
        try:
            model, sklearn_version = payload
            preprocessor = None
        except Exception:
            model = payload
            preprocessor = None
            sklearn_version = "unknown"

    return model, preprocessor, sklearn_version

# -------------------------
# Load model once
# -------------------------
try:
    model, preprocessor, sklearn_version = load_model("bigmart_best_model.pkl")
except FileNotFoundError:
    st.error("Model pickle not found. Please upload `bigmart_best_model.pkl` to the app folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------
# Header
# -------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="big-title">🛒 BigMart Sales Forecast — World Class</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>Interactive sales forecasting — powered by scikit-learn v{sklearn_version}</div>", unsafe_allow_html=True)
with col2:
    st.image("https://raw.githubusercontent.com/dataprofessor/data/master/logo.png", width=90)  # optional small logo; replace or remove

st.write("")  # spacing

# -------------------------
# Sidebar: instructions + batch upload
# -------------------------
with st.sidebar:
    st.header("📥 Inputs & Options")
    st.markdown(
        """
        1. Fill inputs manually or upload a CSV for batch predictions.  
        2. Click **Predict** to get a single prediction or **Batch Predict** to process CSV.  
        3. Use the **Feature Importance** panel below to understand drivers.
        """
    )
    upload = st.file_uploader("Upload CSV (batch predict)", type=["csv"])
    st.divider()
    st.markdown("**Display options**")
    show_chart = st.checkbox("Show feature importance chart (if available)", value=True)
    show_history = st.checkbox("Show prediction history in this session", value=True)

# -------------------------
# Main UI: two panels
# -------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("## ✍️ Manual Input")
    # Group inputs in two columns for compact UI
    i1, i2 = st.columns(2)
    with i1:
        Item_Identifier = st.text_input("Item Identifier", "FDA15")
        Item_Weight = st.number_input("Item Weight (kg)", min_value=0.0, value=12.5, format="%.2f")
        Item_Fat_Content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
        Item_Visibility = st.slider("Item Visibility", min_value=0.0, max_value=0.3, step=0.001, value=0.100)
        Item_Type = st.selectbox("Item Type", [
            "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
            "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
            "Health and Hygiene", "Hard Drinks", "Canned", "Breads",
            "Starchy Foods", "Others", "Seafood"
        ])
    with i2:
        Item_MRP = st.number_input("Item MRP (₹)", min_value=0.0, value=150.0, format="%.2f")
        Outlet_Identifier = st.selectbox("Outlet Identifier", [
            "OUT027", "OUT013", "OUT049", "OUT035", "OUT046",
            "OUT017", "OUT045", "OUT018", "OUT019", "OUT010"
        ])
        Outlet_Size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        Outlet_Location_Type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        Outlet_Type = st.selectbox("Outlet Type", [
            "Supermarket Type1", "Supermarket Type2",
            "Supermarket Type3", "Grocery Store"
        ])
    Outlet_Age = st.slider("Outlet Age (Years)", 0, 40, 15)

    st.markdown("---")
    st.markdown("### ▶️ Actions")
    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        if st.button("Predict Sales"):
            # Assemble DataFrame
            input_df = pd.DataFrame([{
                "Item_Identifier": Item_Identifier,
                "Item_Weight": Item_Weight,
                "Item_Fat_Content": Item_Fat_Content,
                "Item_Visibility": Item_Visibility,
                "Item_Type": Item_Type,
                "Item_MRP": Item_MRP,
                "Outlet_Identifier": Outlet_Identifier,
                "Outlet_Size": Outlet_Size,
                "Outlet_Location_Type": Outlet_Location_Type,
                "Outlet_Type": Outlet_Type,
                "Outlet_Age": Outlet_Age
            }])

            # Preprocess if pipeline exists (best-effort)
            try:
                if preprocessor is not None:
                    X = preprocessor.transform(input_df)
                else:
                    X = input_df  # assume model accepts raw columns
                prediction = model.predict(X)[0]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                prediction = None

            if prediction is not None:
                st.balloons()
                st.success(f"📈 Predicted Item Outlet Sales: ₹{prediction:.2f}")
                # store into session state history
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "input": input_df.to_dict(orient="records")[0],
                    "prediction": float(prediction)
                })

    with action_col2:
        if upload is not None:
            st.success("CSV uploaded. Use 'Batch Predict' below to run predictions on the file.")
        if st.button("Batch Predict (CSV)"):
            if upload is None:
                st.warning("Please upload a CSV with the expected columns first.")
            else:
                try:
                    df = pd.read_csv(upload)
                    # attempt preprocessing
                    if preprocessor is not None:
                        X = preprocessor.transform(df)
                    else:
                        X = df
                    preds = model.predict(X)
                    df_out = df.copy()
                    df_out["predicted_sales"] = preds
                    st.session_state.last_batch = df_out
                    st.success(f"Batch predictions done: {len(df_out)} rows.")
                    # download
                    csv = df_out.to_csv(index=False)
                    st.download_button(label="📥 Download predictions CSV", data=csv, file_name="bigmart_batch_predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

    st.markdown("---")
    st.markdown("### 🧾 Sample data preview")
    sample_button = st.button("Show sample rows")
    if sample_button:
        # produce a few synthetic sample rows (best-effort)
        sample = pd.DataFrame([{
            "Item_Identifier": "FDA15", "Item_Weight": 12.5, "Item_Fat_Content": "Low Fat", "Item_Visibility": 0.100,
            "Item_Type": "Dairy", "Item_MRP": 150.0, "Outlet_Identifier": "OUT027", "Outlet_Size": "Medium",
            "Outlet_Location_Type": "Tier 1", "Outlet_Type": "Supermarket Type1", "Outlet_Age": 10
        }])
        st.dataframe(sample)

with right_col:
    st.markdown("## 📊 Insights & Model Info")
    # show model info
    try:
        st.metric("Model backend", f"scikit-learn v{sklearn_version}")
    except Exception:
        st.metric("Model backend", "Unknown")

    # Show feature importances if available
    fi_container = st.empty()
    if show_chart:
        fi_cols = None
        try:
            # If model has feature_importances_ (tree models)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                # if we have preprocessor with get_feature_names_out, try to fetch feature names
                try:
                    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                        feature_names = preprocessor.get_feature_names_out()
                    else:
                        # fallback - try to build from the most common columns used earlier
                        feature_names = np.array([
                            "Item_Weight","Item_Fat_Content","Item_Visibility","Item_Type","Item_MRP",
                            "Outlet_Identifier","Outlet_Size","Outlet_Location_Type","Outlet_Type","Outlet_Age"
                        ])
                except Exception:
                    feature_names = np.array([
                        "Item_Weight","Item_Fat_Content","Item_Visibility","Item_Type","Item_MRP",
                        "Outlet_Identifier","Outlet_Size","Outlet_Location_Type","Outlet_Type","Outlet_Age"
                    ])
                fi_df = pd.DataFrame({
                    "feature": feature_names[:len(importances)],
                    "importance": importances[:len(importances)]
                }).sort_values("importance", ascending=False).head(15)
                fig = px.bar(fi_df, x="importance", y="feature", orientation="h", title="Feature Importance (approx.)", height=400)
                fi_container.plotly_chart(fig, use_container_width=True)
            else:
                fi_container.info("Feature importance not available for this model type.")
        except Exception as e:
            fi_container.error(f"Failed to compute feature importance: {e}")

    st.markdown("---")
    st.markdown("### 🔎 Quick Explainability")
    st.markdown(
        """
        - Feature importance is approximate and available only for tree-based models.  
        - For linear models, coefficients can be shown similarly (not implemented by default).  
        - For deeper explainability, integrate SHAP or LIME and a saved preprocessor.
        """
    )

# -------------------------
# Prediction history
# -------------------------
if show_history:
    st.markdown("## 🕒 Prediction History (this session)")
    hist = st.session_state.get("history", [])
    if len(hist) == 0:
        st.info("No predictions in this session yet. Try the single predict or batch predict.")
    else:
        hist_df = pd.DataFrame([
            {
                "timestamp": h["timestamp"],
                **{f"in_{k}": v for k, v in h["input"].items()},
                "prediction": h["prediction"]
            }
            for h in hist
        ])
        st.dataframe(hist_df)

# -------------------------
# Footer / small utilities
# -------------------------
st.markdown("---")
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.markdown("Built with ❤️ for retail forecasting — drop improvements or share your results on GitHub.")
    if st.button("Celebrate (confetti!)"):
        st.balloons()

# small helper: expose a download of the app code (optional)
st.markdown("#### Quick tips")
st.markdown(
    """
    • Want better predictions? Retrain with more features (dates, promotions, holidays).  
    • Save a preprocessing pipeline (one pickle that contains `preprocessor` + `model`).  
    • Add SHAP for local explanations.  
    """
)

# END
