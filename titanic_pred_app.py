import streamlit as st
import pickle

# Load the saved model
with open("titanic_predictor.sav", "rb") as file:
    model = pickle.load(file)

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

    # Add port of embarkation selection
    embarkation = st.selectbox("Port of Embarkation", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])
    
    # Convert categorical inputs to numerical values
    sex = 1 if gender == "Female" else 0
    embarkation_map = {"C (Cherbourg)": 0, "Q (Queenstown)": 1, "S (Southampton)": 2}
    embarked = embarkation_map[embarkation]

    # Prepare features for prediction
    features = [pclass, sex, age, sibsp, parch, fare, embarked]

    # Prediction
    if st.button("Predict"):
        result = predict_survival(model, features)
        st.write(f"Prediction: {result}")

if __name__ == "__main__":
    main()
