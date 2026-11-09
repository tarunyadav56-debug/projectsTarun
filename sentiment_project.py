import streamlit as st
import joblib
import requests
from streamlit_option_menu import option_menu

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="NLP Dashboard",
    layout="wide"
)

# =====================================================
# REMOVE TOP SPACE
# =====================================================
st.markdown("""
<style>
.block-container{
    padding-top:0rem;
}

[data-testid="stHeader"]{
    height:0rem;
}

.banner{
    background: linear-gradient(90deg,#4aaaaa,#000000);
    padding:25px;
    border-radius:12px;
    text-align:center;
    color:white;
    font-size:35px;
    font-weight:bold;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="banner">
NLP & ML Projects Dashboard
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.image("flag.jpg", use_container_width=True)

st.sidebar.markdown("""
<h3 style='text-align:center;color:#4aaaaa;'>
TARUN YADAV
</h3>

<p style='text-align:center;'>
AI & NLP Developer
</p>
""", unsafe_allow_html=True)

st.sidebar.title("About Us")
st.sidebar.write(
    "We are developing ML projects based on NLP in LN AI Academy"
)

st.sidebar.title("Contact Us")
st.sidebar.write("+91 9958966311")

# =====================================================
# HORIZONTAL MENU
# =====================================================
selected = option_menu(
    menu_title=None,
    options=[
        "😊 Sentiment Analysis",
        "📧 Spam Detection",
        "🎬 Movie Recommendation"
    ],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {
            "padding": "0!important"
        },
        "nav-link": {
            "font-size": "18px",
            "text-align": "center"
        },
        "nav-link-selected": {
            "background-color": "#4aaaaa",
            "color": "white"
        }
    }
)

# =====================================================
# SENTIMENT ANALYSIS
# =====================================================
if selected == "😊 Sentiment Analysis":

    model = joblib.load("sentiment_model.pkl")

    st.header("😊 Sentiment Analysis")

    sample_review = st.selectbox(
        "Sample Reviews",
        [
            "good food",
            "quality was not good",
            "awesome taste"
        ]
    )

    if st.button("Predict Sample Review"):

        pred = model.predict([sample_review])
        prob = model.predict_proba([sample_review])

        if pred[0] == 0:
            st.error(
                f"Negative ({prob[0][0]:.2f})"
            )
        else:
            st.success(
                f"Positive ({prob[0][1]:.2f})"
            )
            st.balloons()

    review = st.text_input("Enter Your Review")

    if st.button("Predict Review"):

        pred = model.predict([review])
        prob = model.predict_proba([review])

        if pred[0] == 0:
            st.error(
                f"Negative ({prob[0][0]:.2f})"
            )
        else:
            st.success(
                f"Positive ({prob[0][1]:.2f})"
            )
            st.balloons()

# =====================================================
# SPAM DETECTION
# =====================================================
elif selected == "📧 Spam Detection":

    model = joblib.load("spam_model.pkl")

    st.header("📧 Spam Detection")

    sample_msg = st.selectbox(
        "Sample Messages",
        [
            "Congratulations! You've 
