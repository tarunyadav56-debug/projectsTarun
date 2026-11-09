import streamlit as st
import joblib
import requests
from streamlit_option_menu import option_menu

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="NLP Dashboard",
    layout="wide"
)

# ==================================================
# BANNER
# ==================================================
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
    NLP & ML Projects Dashboard
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.image("flag.jpg", use_container_width=True)

st.sidebar.markdown("""
<h3 style='text-align:center;color:#4aaaaa;'>
TARUN YADAV
</h3>

<p style='text-align:center;'>
AI & NLP Developer
</p>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title="Projects",
    options=[
        "Sentiment Analysis",
        "Spam Detection",
        "Movie Recommendation"
    ],
    icons=[
        "emoji-smile",
        "envelope",
        "film"
    ],
    default_index=0
)

st.sidebar.title("About Us")
st.sidebar.write(
    "We are developing ML projects based on NLP in LN AI Academy"
)

st.sidebar.title("Contact Us")
st.sidebar.write("+91 9958966311")

# ==================================================
# SENTIMENT ANALYSIS
# ==================================================
if selected == "Sentiment Analysis":

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
            st.error(f"Negative ({prob[0]:.2f})")
        else:
            st.success(f"Positive ({prob[0]:.2f})")
            st.balloons()

    user_review = st.text_input("Your Review")

    if st.button("Predict Review"):

        pred = model.predict([user_review])
        prob = model.predict_proba([user_review])

        if pred[0] == 0:
            st.error(f"Negative ({prob[0]:.2f})")
        else:
            st.success(f"Positive ({prob[0]:.2f})")
            st.balloons()

# ==================================================
# SPAM DETECTION
# ==================================================
elif selected == "Spam Detection":

    model = joblib.load("spam_model.pkl")

    st.header("📧 Spam Detection")

    sample_msg = st.selectbox(
        "Sample Messages",
        [
            "Congratulations! You've won a free iPhone.",
            "Your account has been compromised.",
            "Thanks for your help with debugging."
        ]
    )

    if st.button("Predict Sample Message"):

        pred = model.predict([sample_msg])
        prob = model.predict_proba([sample_msg])

        if pred[0] == 1:
            st.error(f"🚨 Spam Detected ({prob[0]:.2f})")
        else:
            st.success(f"✅ No Spam ({prob[0]:.2f})")
            st.snow()

    user_msg = st.text_input("Enter Message")

    if st.button("Predict Message"):

        pred = model.predict([user_msg])
        prob = model.predict_proba([user_msg])

        if pred[0] == 1:
            st.error(f"🚨 Spam Detected ({prob[0]:.2f})")
        else:
            st.success(f"✅ No Spam ({prob[0]:.2f})")
            st.snow()

# ==================================================
# MOVIE RECOMMENDATION
# ==================================================
elif selected == "Movie Recommendation":

    model = joblib.load("model.pkl")
    vectors = joblib.load("vectors.pkl")
    df = joblib.load("dataframe.pkl")

    st.header("🎬 Movie Recommendation System")

    watched_movie = st.selectbox(
        "Select Watched Movie",
        options=df["name"]
    )

    if st.button("Recommend Movies"):

        index = df[df["name"] == watched_movie].index[0]

        distances, indexes = model.kneighbors(
            vectors[index].toarray()
        )

        st.subheader("Recommended Movies")

        cols = st.columns(5)

        for col_no, i in enumerate(indexes[0][1:6]):

            with colsmovie_name = df.loc[i, "name"]
                movie_id = df.loc[i, "movie_id"]

                st.write(movie_name)

                try:
                    response = requests.get(
                        f"http://www.omdbapi.com/?i={movie_id}&apikey=b84e8a7c"
                    )

                    data = response.json()

                    if "Poster" in data and data["Poster"] != "N/A":
                        st.image(
                            data["Poster"],
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Poster Error: {e}")
