import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Paris Housing Price Predictor", layout="centered")
st.title("🏠 Paris Housing Price Predictor")

# === Upload model and scaler files ===
model_file = st.file_uploader("Upload Trained Model (.pkl)", type=["pkl"], key="model")
scaler_file = st.file_uploader("Upload Feature Scaler (.pkl)", type=["pkl"], key="scaler")

# === Initialize session state for prediction and reset ===
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# === Function to reset session state ===
def reset():
    st.session_state.predicted = False
    st.session_state.prediction = None

# === If model and scaler are loaded ===
if model_file and scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

    st.success("✅ Model and Scaler loaded successfully!")

    # === Input form ===
    with st.form("prediction_form"):
        squareMeters = st.number_input("Area (Square Meters)", min_value=50, max_value=99999, value=50, step=10)
        numberOfRooms = st.slider("Number of Rooms", 1, 100, 3)
        hasYard = st.checkbox("Has Yard")
        hasPool = st.checkbox("Has Pool")
        floors = st.slider("Number of Floors", 1, 100, 1)
        cityCode = st.number_input("City Code", min_value=1000, max_value=99999, value=75000, step=100)
        cityPartRange = st.slider("City Part Range", 1, 10, 5)
        numPrevOwners = st.slider("Previous Owners", 0, 10, 1)
        made = st.number_input("Year Built", min_value=1990, max_value=2026, value=2000)
        isNewBuilt = st.checkbox("Newly Built")
        hasStormProtector = st.checkbox("Has Storm Protector")
        basement = st.slider("Basement Area (sq m)", 0, 10000, 0)
        attic = st.slider("Attic Area (sq m)", 0, 10000, 0)
        garage = st.slider("Garage Area (sq m)", 0, 1000, 0)
        hasStorageRoom = st.checkbox("Has Storage Room")
        hasGuestRoom = st.checkbox("Has Guest Room")

        submitted = st.form_submit_button("💸 Predict Price")

        if submitted:
            input_data = np.array([[
                squareMeters, numberOfRooms, int(hasYard), int(hasPool), floors,
                cityCode, cityPartRange, numPrevOwners, made, int(isNewBuilt),
                int(hasStormProtector), basement, attic, garage,
                int(hasStorageRoom), int(hasGuestRoom)
            ]])

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            st.session_state.predicted = True
            st.session_state.prediction = prediction

    # === Show prediction result ===
    if st.session_state.predicted:
        st.success(f"🏷️ Estimated Price: €{st.session_state.prediction:,.2f}")
        st.button("🔁 New Search", on_click=reset)

else:
    st.warning("👆 Please upload both the trained model and scaler to proceed.")
