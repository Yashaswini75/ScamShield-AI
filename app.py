from pathlib import Path
from datetime import timezone
import re
import joblib
import streamlit as st
import pandas as pd
from pymongo.errors import PyMongoError
import database

st.set_page_config(page_title="ScamShield | Message Analyzer", page_icon="🛡️", layout="wide")
MODEL_PATH = Path("models/spam_pipeline.joblib")
METRICS_PATH = Path("models/evaluation_metrics.json")

st.title("🛡️ ScamShield")
st.caption("Machine-learning message screening with MongoDB scan history")
st.warning("Prototype notice: this model was trained on SMS spam data. It can miss phishing and may flag legitimate messages. Never use its prediction alone to decide whether a message or link is safe.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python download_dataset.py` and then `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(str(MODEL_PATH))
connected, db_message = database.check_connection()
if connected:
    st.sidebar.success("MongoDB connected")
else:
    st.sidebar.error("MongoDB unavailable")
    st.sidebar.caption("Start the MongoDB service, then refresh this page.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Open", ["Analyze message", "Scan history", "Model evaluation", "About"])
    st.divider()
    st.caption("Privacy: scans are only stored when you select the save option. Avoid pasting passwords, account numbers, or other sensitive data.")

SUSPICIOUS_PATTERNS = [
    (r"\b(urgent|immediately|act now|final warning|expires today)\b", "Urgency or pressure language"),
    (r"\b(verify|suspend(ed)?|blocked|locked|unusual activity)\b", "Account verification or threat language"),
    (r"\b(password|otp|one[- ]time password|pin|cvv|bank details)\b", "Requests or references to sensitive credentials"),
    (r"\b(prize|winner|won|lottery|claim your reward|free gift)\b", "Prize or reward claim language"),
    (r"\b(click here|click the link|login now|confirm your account)\b", "Call to click or sign in"),
]
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
IP_URL_RE = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}", re.IGNORECASE)


def analyze_warning_signs(message):
    signs = []
    for pattern, description in SUSPICIOUS_PATTERNS:
        if re.search(pattern, message, flags=re.IGNORECASE):
            signs.append(description)
    if URL_RE.search(message):
        signs.append("Contains a URL; inspect the domain carefully without opening it")
    if IP_URL_RE.search(message):
        signs.append("URL uses a raw IP address")
    if message.count("!") >= 3 or re.search(r"\b[A-Z]{5,}\b", message):
        signs.append("Unusual punctuation or excessive uppercase text")
    if not signs:
        signs.append("No rule-based warning signs detected; this does not prove the message is safe")
    return signs

if page == "Analyze message":
    st.subheader("Check a suspicious message")
    input_type = st.selectbox("Input type", ["SMS", "Email", "URL / message containing URL"])
    message = st.text_area("Paste the message", height=180, placeholder="Example: Your account is locked. Verify immediately...")
    save_to_db = st.checkbox("Save this scan to MongoDB history", value=False)
    if st.button("Analyze message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Enter a message before analyzing it.")
        elif len(message) > 20000:
            st.warning("Please limit the message to 20,000 characters.")
        else:
            # Pipeline handles text vectorization; preserve original punctuation and numbers.
            prediction_id = int(model.predict([message])[0])
            probabilities = model.predict_proba([message])[0]
            confidence = float(probabilities[prediction_id])
            prediction = "Spam" if prediction_id == 1 else "Legitimate"
            signs = analyze_warning_signs(message)
            left, right = st.columns(2)
            with left:
                if prediction == "Spam":
                    st.error("🚨 Model prediction: Spam")
                else:
                    st.success("Model prediction: Legitimate")
                st.metric("Model confidence (not calibrated risk)", f"{confidence:.1%}")
            with right:
                st.markdown("**Warning signs**")
                for sign in signs:
                    st.write(f"- {sign}")
            st.caption("Warning signs are transparent keyword/URL heuristics, not explanations of the ML model's internal decision.")
            if save_to_db:
                if connected:
                    try:
                        scan_id = database.save_scan(message, input_type, prediction, confidence, signs)
                        st.session_state["last_scan_id"] = scan_id
                        st.success("Scan saved to MongoDB.")
                    except PyMongoError as exc:
                        st.error(f"Could not save scan: {exc}")
                else:
                    st.error("MongoDB is unavailable, so this scan was not saved.")
            if st.session_state.get("last_scan_id"):
                with st.expander("Give feedback on the most recently saved scan"):
                    feedback = st.radio("Was the prediction correct?", ["Correct", "Incorrect"], horizontal=True)
                    comment = st.text_input("Optional note (do not include sensitive information)")
                    if st.button("Submit feedback"):
                        if connected:
                            database.save_feedback(st.session_state["last_scan_id"], feedback, comment)
                            st.success("Feedback recorded.")
                        else:
                            st.error("MongoDB is unavailable.")

elif page == "Scan history":
    st.subheader("MongoDB scan history")
    if not connected:
        st.error(f"Cannot load history: {db_message}")
    else:
        stats = database.get_dashboard_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total saved scans", stats["total"])
        c2.metric("Predicted spam", stats["spam"])
        c3.metric("Predicted legitimate", stats["legitimate"])
        rows = database.get_recent_scans(100)
        if rows:
            for row in rows:
                row["timestamp"] = row["timestamp"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if row.get("timestamp") else ""
                row["message"] = (row.get("message", "")[:180] + "…") if len(row.get("message", "")) > 180 else row.get("message", "")
                row.pop("_id", None)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No scans saved yet. Analyze a message and opt in to saving it.")

elif page == "Model evaluation":
    st.subheader("Held-out test-set evaluation")
    if METRICS_PATH.exists():
        import json
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset rows", f"{metrics['dataset_rows']:,}")
        c2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        c3.metric("Spam recall", f"{metrics['spam_recall']:.1%}")
        c4.metric("Spam F1", f"{metrics['spam_f1']:.1%}")
        st.write("Confusion matrix (rows = true class, columns = predicted class; order: ham, spam)")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["True ham", "True spam"], columns=["Predicted ham", "Predicted spam"]), use_container_width=True)
        st.json(metrics["label_counts"])
        st.caption("Metrics are measured on a held-out split of the SMS dataset, not a guarantee of performance on email or modern phishing messages.")
    else:
        st.info("Evaluation metrics not found. Run `python train_model.py` first.")

else:
    st.subheader("About this project")
    st.markdown("""
    **Machine learning:** TF-IDF text features + Multinomial Naive Bayes.

    **NoSQL:** MongoDB stores opted-in scan history and user feedback as flexible documents. Indexes support recent-history and prediction queries.

    **Limitations:** The UCI dataset contains SMS spam/ham examples, not a comprehensive phishing-email corpus. The URL checks are simple heuristics; the application does not visit URLs or query reputation services.

    **Dataset citation:** Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
    """)
