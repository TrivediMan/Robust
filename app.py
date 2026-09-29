from pathlib import Path
import pickle
import sys

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "Model"
MODEL_PATH = MODEL_DIR / "Best_Model.pkl"


# ============================================================
# MODEL INPUT FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "location_score",
    "property_age",
    "distance_city_km",
    "near_school",
    "near_metro",
    "crime_rate_index"
]


# ============================================================
# APPLICATION CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .price-box {
        padding: 22px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #dddddd;
    }

    .price-label {
        font-size: 16px;
        font-weight: 500;
    }

    .price-value {
        font-size: 34px;
        font-weight: 700;
        margin-top: 5px;
    }

    .info-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK PYTHON / DEPENDENCIES
# ============================================================

def check_dependencies():

    try:
        import sklearn

        return sklearn.__version__

    except ImportError:

        st.error("❌ Scikit-learn is not installed.")

        st.code(
            "python -m pip install scikit-learn"
        )

        st.stop()


SKLEARN_VERSION = check_dependencies()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    # --------------------------------------------------------
    # Check model directory
    # --------------------------------------------------------

    if not MODEL_DIR.exists():

        raise FileNotFoundError(
            f"""
Model directory was not found.

Expected:
{MODEL_DIR}

Create this folder:
Model/

and place Best_Model.pkl inside it.
"""
        )

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"""
Best_Model.pkl was not found.

Expected model location:

{MODEL_PATH}

Project structure should be:

Project/
├── app.py
├── requirements.txt
└── Model/
    └── Best_Model.pkl
"""
        )

    # --------------------------------------------------------
    # Load pickle
    # --------------------------------------------------------

    try:

        with open(MODEL_PATH, "rb") as file:

            model = pickle.load(file)

    except ModuleNotFoundError as e:

        raise RuntimeError(
            f"""
Missing Python dependency while loading Best_Model.pkl.

Missing module:
{e.name}

Current Python:
{sys.version}

Current scikit-learn:
{SKLEARN_VERSION}

Install the required dependencies using:

python -m pip install -r requirements.txt
"""
        ) from e

    except AttributeError as e:

        error_text = str(e)

        if "_RemainderColsList" in error_text:

            raise RuntimeError(
                f"""
SCIKIT-LEARN MODEL COMPATIBILITY ERROR

Best_Model.pkl was created using a different
scikit-learn version than the one currently running.

Current scikit-learn version:
{SKLEARN_VERSION}

Original error:
{error_text}

IMPORTANT:
Do NOT modify app.py to fix this.

You must retrain and save Best_Model.pkl
using the same scikit-learn version specified
in requirements.txt.

Recommended version:

scikit-learn==1.7.2

Then replace:

Model/Best_Model.pkl

with the newly trained model.
"""
            ) from e

        raise RuntimeError(
            f"""
Unable to load Best_Model.pkl.

AttributeError:
{error_text}

This usually means the pickle was created
with an incompatible package version.
"""
        ) from e

    except EOFError as e:

        raise RuntimeError(
            """
Best_Model.pkl appears to be empty or corrupted.

Retrain and save the model again.
"""
        ) from e

    except pickle.UnpicklingError as e:

        raise RuntimeError(
            f"""
Best_Model.pkl could not be unpickled.

The file may be corrupted or may not be
a valid Python pickle.

Error:
{e}
"""
        ) from e

    except Exception as e:

        raise RuntimeError(
            f"""
Unable to load Best_Model.pkl.

Error type:
{type(e).__name__}

Error:
{e}

Check that Best_Model.pkl was created with
a compatible scikit-learn version.
"""
        ) from e

    # --------------------------------------------------------
    # Validate predict()
    # --------------------------------------------------------

    if not hasattr(model, "predict"):

        raise TypeError(
            f"""
Invalid model file.

Loaded object:
{type(model).__name__}

The saved object does not have a predict()
method.

Save the fitted model/pipeline itself:

pickle.dump(best_model, file)

NOT:

pickle.dump("best_model", file)
"""
        )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner("Loading machine learning model..."):

        model = load_model()

except Exception as error:

    st.error("❌ Model Loading Failed")

    st.code(
        str(error),
        language="text"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏠 House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Machine Learning based House Price Prediction
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write(
        "**Model:**",
        type(model).__name__
    )

    st.write(
        "**Scikit-learn:**",
        SKLEARN_VERSION
    )

    st.write(
        "**Model Status:**"
    )

    st.success("Loaded successfully")

    st.divider()

    st.caption(
        "House Price Prediction System"
    )

    st.caption(
        "Built with Python, Pandas, "
        "Scikit-learn and Streamlit"
    )


# ============================================================
# DESCRIPTION
# ============================================================

st.write(
    "Enter the property details below to estimate "
    "the house price."
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("house_price_form"):

    st.subheader("🏡 Property Details")

    col1, col2 = st.columns(2)

    # ========================================================
    # COLUMN 1
    # ========================================================

    with col1:

        area_sqft = st.number_input(
            "Area (sqft)",
            min_value=1.0,
            max_value=100000.0,
            value=1500.0,
            step=50.0
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=20,
            value=3,
            step=1
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0,
            max_value=20,
            value=2,
            step=1
        )

        property_age = st.number_input(
            "Property Age (years)",
            min_value=0,
            max_value=200,
            value=10,
            step=1
        )

        distance_city_km = st.number_input(
            "Distance from City (km)",
            min_value=0.0,
            max_value=500.0,
            value=10.0,
            step=0.5
        )

    # ========================================================
    # COLUMN 2
    # ========================================================

    with col2:

        location_score = st.number_input(
            "Location Score",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

        crime_rate_index = st.number_input(
            "Crime Rate Index",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        )

        near_school = st.selectbox(
            "Near School",
            options=[0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        near_metro = st.selectbox(
            "Near Metro",
            options=[0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔮 Predict House Price",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            area_sqft,
            bedrooms,
            bathrooms,
            location_score,
            property_age,
            distance_city_km,
            near_school,
            near_metro,
            crime_rate_index
        ]],
        columns=FEATURE_COLUMNS
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in input_data.columns
    ]

    if missing_columns:

        st.error(
            f"❌ Missing input columns: {missing_columns}"
        )

        st.stop()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        with st.spinner("Calculating house price..."):

            prediction = model.predict(input_data)

        # ----------------------------------------------------
        # Validate prediction
        # ----------------------------------------------------

        if prediction is None:

            raise ValueError(
                "Model returned None."
            )

        if len(prediction) == 0:

            raise ValueError(
                "Model returned an empty prediction."
            )

        prediction_value = float(prediction[0])

        # ----------------------------------------------------
        # Validate numeric result
        # ----------------------------------------------------

        if pd.isna(prediction_value):

            raise ValueError(
                "Model returned NaN."
            )

        if prediction_value < 0:

            st.warning(
                "⚠️ The model returned a negative price. "
                "Please verify the training data/model."
            )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="price-box">
                <div class="price-label">
                    🏠 Estimated House Price
                </div>
                <div class="price-value">
                    ₹{prediction_value:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "✅ Prediction generated successfully."
        )

        # ----------------------------------------------------
        # Input summary
        # ----------------------------------------------------

        with st.expander("📋 View Input Data"):

            display_data = input_data.copy()

            display_data["near_school"] = (
                display_data["near_school"]
                .map({0: "No", 1: "Yes"})
            )

            display_data["near_metro"] = (
                display_data["near_metro"]
                .map({0: "No", 1: "Yes"})
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # Prediction error
    # --------------------------------------------------------

    except Exception as error:

        st.error(
            "❌ Prediction Failed"
        )

        st.code(
            f"""
Error Type:
{type(error).__name__}

Error:
{error}
""",
            language="text"
        )

        with st.expander("🔧 Debug Information"):

            st.write(
                "**Model Type:**",
                type(model).__name__
            )

            st.write(
                "**Input Columns:**",
                list(input_data.columns)
            )

            st.write(
                "**Expected Features:**",
                FEATURE_COLUMNS
            )

            st.dataframe(
                input_data,
                use_container_width=True
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("🔍 Technical Model Information"):

    st.write(
        "**Model Type:**",
        type(model).__name__
    )

    st.write(
        "**Model File:**",
        str(MODEL_PATH)
    )

    st.write(
        "**Scikit-learn Version:**",
        SKLEARN_VERSION
    )

    st.write(
        "**Predict Method Available:**",
        hasattr(model, "predict")
    )

    # --------------------------------------------------------
    # Show feature information if available
    # --------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        st.write(
            "**Model Features:**",
            list(model.feature_names_in_)
        )

    elif hasattr(model, "named_steps"):

        st.write(
            "**Pipeline Steps:**",
            list(model.named_steps.keys())
        )


# ============================================================
# PROJECT FLOW
# ============================================================

st.divider()

st.subheader("🔄 Project Flow")

st.code(
    """
User Input
     ↓
Pandas DataFrame
     ↓
Saved Scikit-learn Pipeline
     ↓
Preprocessing
     ↓
Regression Model
     ↓
Predicted House Price
"""
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏠 House Price Prediction | "
    "Machine Learning Regression Application"
)

# -----