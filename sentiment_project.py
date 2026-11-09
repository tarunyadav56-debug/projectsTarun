import streamlit as st
import joblib

model=joblib.load('sentiment_model.pkl')

st.set_page_config(layout='wide')

st.markdown("""
<style>
.banner {
    background: linear-gradient(90deg, #4aaaaa, #000000);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 35px;
    font-weight: bold;
    margin-bottom: 25px;
}
</style>

<div class="banner">
    Sentiment Analysis Project
</div>
""", unsafe_allow_html=True)

st.sidebar.image("flag.jpg")

st.sidebar.title("About us")
st.sidebar.text("We are developing ML projects based on NLP in LN AI Academy")

st.sidebar.title("About Project")
st.sidebar.text("This project predicts sentiment of given text....")

st.sidebar.title("Contact us")
st.sidebar.text("+919958966311")

sample_review=st.selectbox("Sample Reviews",options=['good food','quality was not good','awesome taste'])
if st.button("Predict",key="b1"):
    pred=model.predict([sample_review])
    st.text(pred)
st.text_input("Your Review")
st.button("Predict",key="b2")