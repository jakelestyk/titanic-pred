import streamlit as st
import pickle

# Load the saved model
model_path = "titanic_predictor.sav"

if model_path:
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("Error: Model file not found. Please ensure 'titanic_predictor.sav' is uploaded.")
        st.stop()

# Function to make predictions
def predict_survival(model, features):
    prediction = model.predict([features])
    return "Survived" if prediction[0] == 1 else "Did Not Survive"

# Streamlit app
def main():
    st.title("Titanic Survival Predictor")
    st.write("Enter the details below to predict survival on the Titanic:")

    # Input fields
    gender = st.radio("Gender", ("Male", "Female"))
    age = st.number_input("Age", min_value=0, max_value=100, step=1, value=30)
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, step=1, value=0)
    parch = st.number_input("Number of Parents/Children Aboard", min_value=0, step=1, value=0)
    fare = st.number_input("Fare (in USD)", min_value=0.0, step=0.01, value=50.0)

    # Convert categorical inputs to numerical values
    sex = 1 if gender == "Male" else 0  # Matches the training data conversion

    # Ensure only 6 features are passed (Embarked was dropped in training)
    features = [pclass, sex, age, sibsp, parch, fare]

    # Debugging: Show input feature shape
    st.write(f"Input feature array: {features}")

    # Prediction
    if st.button("Predict"):
        try:
            result = predict_survival(model, features)
            st.write(f"Prediction: {result}")
        except ValueError as e:
            st.error(f"Model input error: {e}")

if __name__ == "__main__":
    main()
