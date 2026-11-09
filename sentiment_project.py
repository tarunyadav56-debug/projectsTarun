import streamlit as st
import joblib
import requests
from streamlit_option_menu import option_menu

st.set_page_config(page_title="NLP Dashboard", layout="wide")

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

st.sidebar.image("flag.jpg", use_container_width=True)

selected = option_menu(
    "Projects",
    ["Sentiment Analysis", "Spam Detection", "Movie Recommendation"],
    icons=["emoji-smile", "envelope", "film"],
    default_index=0
)

st.sidebar.title("About Us")
st.sidebar.write("We are developing ML projects based on NLP in LN AI Academy")

st.sidebar.title("Contact")
st.sidebar.write("+919717776422")

# ==========================
# Sentiment Analysis
# ==========================
if selected == "Sentiment Analysis":

    model = joblib.load("sentiment_model.pkl")

    st.header("😊 Sentiment Analysis")

    sample_review = st.selectbox(
        "Sample Reviews",
        ["good food", "quality was not good", "awesome taste"]
    )

    if st.button("Predict Sample Review"):

        pred = model.predict([sample_review])
        prob = model.predict_proba([sample_review])

        if pred[0] == 0:
            st.error(f"Negative ({prob[0][0]:.2f})")
        else:
            st.success(f"Positive ({prob[0][1]:.2f})")
            st.balloons()

    user_review = st.text_input("Your Review")

    if st.button("Predict Review"):

        pred = model.predict([user_review])
        prob = model.predict_proba([user_review])

        if pred[0] == 0:
            st.error(f"Negative ({prob[0][0]:.2f})")
        else:
            st.success(f"Positive ({prob[0][1]:.2f})")
            st.balloons()

# ==========================
# Spam Detection
# ==========================
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
            st.error(f"Spam Detected ({prob[0][1]:.2f})")
        else:
            st.success(f"No Spam ({prob[0][0]:.2f})")
            st.snow()

    user_msg = st.text_input("Enter Message")

    if st.button("Predict Message"):

        pred = model.predict([user_msg])
        prob = model.predict_proba([user_msg])

        if pred[0] == 1:
            st.error(f"Spam Detected ({prob[0][1]:.2f})")
        else:
            st.success(f"No Spam ({prob[0][0]:.2f})")
            st.snow()

# ==========================
# Movie Recommendation
# ==========================
elif selected == "Movie Recommendation":

    model = joblib.load("model.pkl")
    vectors = joblib.load("vectors.pkl")
    df = joblib.load("dataframe.pkl")

    st.header("🎬 Movie Recommendation")

    watched_movie = st.selectbox(
        "Select Movie",
        df["name"]
    )

    if st.button("Recommend Movies"):

        index = df[df["name"] == watched_movie].index[0]

        distances, indexes = model.kneighbors(
            vectors[index].toarray()
        )

        st.subheader("Recommended Movies")

        cols = st.columns(5)

        for col_no, i in enumerate(indexes[0][1:6]):

    with cols[col_no]:

        st.write(df.iloc[i]["name"])

        movie_id = df.iloc[i][0]

                try:
                    response = requests.get(
                        f"http://www.omdbapi.com/?i={movie_id}&apikey=7472611b"
                    )

                    data = response.json()

                    if "Poster" in data:
                        st.image(data["Poster"])
                except:
                    pass
