import streamlit as st
import pickle

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Spam Detector", layout="centered")

st.title("📧 Spam Mail Detector")
st.write("Check whether a message is Spam or Ham")

# =========================
# INPUT
# =========================
message = st.text_area("Enter your message:")

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        transformed = vectorizer.transform([message])
        prediction = model.predict(transformed)[0]

        if prediction == "spam":
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT Spam (HAM)")

# =========================
# FOOTER
# =========================
st.write("---")
st.caption("Built with Machine Learning + Streamlit by Vaisak")