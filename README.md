# 🏠 Robust — House Price Predictor

A machine learning web app that estimates house prices (in ₹) from property details. A trained scikit-learn regression pipeline is served through a simple [Streamlit](https://streamlit.io/) interface.

🔗 **Live App:** [https://robust-5fd7qcgpjn6hxvnha76eg4.streamlit.app/](https://robust-5fd7qcgpjn6hxvnha76eg4.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://robust-5fd7qcgpjn6hxvnha76eg4.streamlit.app/)

---

---


# 🎥 Project Video


🔗 **[▶️ Watch Project Video](https://drive.google.com/file/d/1hp8G6ceAHJGb_ePIzEmJZYx5Y0InOMYR/view?usp=drive_link)**

---

## ✨ Features

- Interactive form to enter property details and get an instant price estimate
- Pre-trained scikit-learn pipeline loaded from `Model/Best_Model.pkl`
- Input validation and clear error messages (missing model, corrupted pickle, scikit-learn version mismatch)
- Sidebar and expandable panel showing model type, scikit-learn version and pipeline details
- Expandable summary of the inputs used for each prediction

---

## 🧮 Input Features

| Feature | Description | Range |
|---|---|---|
| `area_sqft` | Built-up area in square feet | 1 – 100,000 |
| `bedrooms` | Number of bedrooms | 0 – 20 |
| `bathrooms` | Number of bathrooms | 0 – 20 |
| `location_score` | Desirability score of the location | 0 – 10 |
| `property_age` | Age of the property in years | 0 – 200 |
| `distance_city_km` | Distance from the city center (km) | 0 – 500 |
| `near_school` | School nearby (Yes = 1 / No = 0) | 0 or 1 |
| `near_metro` | Metro station nearby (Yes = 1 / No = 0) | 0 or 1 |
| `crime_rate_index` | Crime rate index of the area | 0 – 100 |

---

## 🔄 How It Works

```
User Input
    ↓
Pandas DataFrame
    ↓
Saved scikit-learn Pipeline (Model/Best_Model.pkl)
    ↓
Preprocessing
    ↓
Regression Model
    ↓
Predicted House Price
```

---

## 📁 Project Structure

```
Robust/
├── .devcontainer/     # Dev container configuration
├── Model/
│   └── Best_Model.pkl # Trained scikit-learn pipeline
├── data/              # Dataset used for training
├── notebook/          # Exploration and model training notebooks
├── app.py             # Streamlit application
├── requirements.txt   # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** & **NumPy** — data handling
- **scikit-learn** — model training and inference
- **SciPy** — scientific computing
- **Streamlit** — web interface

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/TrivediMan/Robust.git
cd Robust
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## ⚠️ Model Compatibility Note

`Best_Model.pkl` is a pickled scikit-learn object, so it must be loaded with the **same scikit-learn version** it was trained with. `requirements.txt` pins `scikit-learn==1.5.1`.

If you see a compatibility error such as `_RemainderColsList`, retrain and re-save the model using the version in `requirements.txt`, then replace `Model/Best_Model.pkl`. Do not edit `app.py` to work around it.

To retrain, use the notebooks in the `notebook/` folder and save the fitted pipeline with:

```python
import pickle

with open("Model/Best_Model.pkl", "wb") as file:
    pickle.dump(best_model, file)
```

---

## 🧪 Usage

1. Launch the app with `streamlit run app.py`.
2. Fill in the property details in the form.
3. Click **🔮 Predict House Price**.
4. View the estimated price and expand **View Input Data** to review your inputs.

---

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request.

---

## 📄 License

No license has been specified yet. Add a `LICENSE` file (e.g., MIT) to define how others may use this project.

---

## 👤 Author

**TrivediMan** — [GitHub](https://github.com/TrivediMan)
