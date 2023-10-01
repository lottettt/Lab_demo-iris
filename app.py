import pickle
import warnings
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

pickle_in = open(r"model_iris.pkl", "rb")
classifier = pickle.load(pickle_in)

def predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width):
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return prediction

def input_output():
    st.title("Iris Variety Prediction")
    st.image("https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png", width=600)
    st.markdown("You are using Streamlit...", unsafe_allow_html=True)

    sepal_length = st.text_input("Enter Sepal Length", "")
    sepal_width = st.text_input("Enter Sepal Width", "")
    petal_length = st.text_input("Enter Petal Length", "")
    petal_width = st.text_input("Enter Petal Width", "")

    result = ""
    if st.button("Click here to Predict"):
        try:
            sepal_length = float(sepal_length)
            sepal_width = float(sepal_width)
            petal_length = float(petal_length)
            petal_width = float(petal_width)

            result = predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width)
            st.balloons()
            st.success(f'The output is {result}')
        except ValueError:
            st.error("Please enter valid numerical values for all input fields.")

if __name__ == '__main__':
    input_output()
