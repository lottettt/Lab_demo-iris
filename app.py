import pickle
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

from PIL import Image

pickle_in = open("best_rf_model.pkl", "rb")
classifier = pickle.load(pickle_in)

def predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width):
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction

def input_output():
    st.title("Iris Variety Prediction")
    st.image("https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png", width=600)
    st.markdown("You are using Streamlit...", unsafe_allow_html=True)

    sepal_length = float(st.text_input("Enter Sepal Length", "."))
    sepal_width = float(st.text_input("Enter Sepal Width", "."))
    petal_length = float(st.text_input("Enter Petal Length", "."))
    petal_width = float(st.text_input("Enter Petal Width", "."))


    result = ""
    if st.button("Click here to Predict"):
        result = predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width)
        st.balloons()
        st.success(f'The output is {result}')

if __name__ == '__main__':
    input_output()