import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit webpage title
st.title('Iris Species Prediction')

# Creating input fields for the user
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.5)

# Button for prediction
if st.button('Predict'):
    # Making prediction
    prediction = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
    species = ['Setosa', 'Versicolour', 'Virginica']

    # Displaying the prediction
    st.write(f'The Iris Species is predicted to be: {species[prediction[0]]}')
