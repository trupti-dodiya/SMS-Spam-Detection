
import streamlit as st
import joblib

# 1. Configure the application
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# 2. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("spam_model.pkl")

model = load_model()

# 3. Sidebar
st.sidebar.title("About This Project")
st.sidebar.write(
    "This application uses Machine Learning "
    "to classify SMS messages as Spam or Ham."
)

st.sidebar.write("**Algorithm:** Multinomial Naive Bayes")
st.sidebar.write("**Feature Extraction:** TF-IDF")

# 4. Main application
st.title("📩 SMS Spam Detection")
st.caption("A Machine Learning project built using Python and Streamlit.")

st.divider()

st.subheader("Check Your SMS")

message = st.text_area(
    "Enter your message below:",
    placeholder="Example: Hey, are you coming to college tomorrow?",
    height=150
)

# 5. Prediction
if st.button("Check Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        prediction = model.predict([message])[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == "spam":
            st.error("🚨 This message is classified as SPAM.")

        else:
            st.success("✅ This message is classified as HAM (Genuine).")

st.divider()

st.caption(
    "Note: Machine Learning predictions can be incorrect. "
    "Do not rely on this application as the only way to identify suspicious messages."
)
