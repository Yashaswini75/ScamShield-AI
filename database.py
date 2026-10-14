"""MongoDB access layer for scan history and user feedback."""
import os
from datetime import datetime, timezone
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/")
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2500)
db = client["ScamDetectionDB"]
scan_history = db["scan_history"]
user_feedback = db["user_feedback"]
threat_patterns = db["threat_patterns"]

# Indexes help the dashboard query recent scans efficiently.
try:
    scan_history.create_index([("timestamp", DESCENDING)])
    scan_history.create_index([("prediction", 1)])
except PyMongoError:
    pass


def check_connection():
    try:
        client.admin.command("ping")
        return True, "Connected to MongoDB"
    except PyMongoError as exc:
        return False, str(exc)


def save_scan(message, input_type, prediction, confidence, warning_signs):
    doc = {
        "message": message,
        "input_type": input_type,
        "prediction": prediction,
        "model_confidence": float(confidence),
        "warning_signs": list(warning_signs),
        "timestamp": datetime.now(timezone.utc),
    }
    result = scan_history.insert_one(doc)
    return str(result.inserted_id)


def get_recent_scans(limit=20):
    return list(scan_history.find(
        {}, {"message": 1, "input_type": 1, "prediction": 1,
            "model_confidence": 1, "warning_signs": 1, "timestamp": 1}
    ).sort("timestamp", DESCENDING).limit(int(limit)))


def get_dashboard_stats():
    total = scan_history.count_documents({})
    spam = scan_history.count_documents({"prediction": "Spam"})
    ham = scan_history.count_documents({"prediction": "Legitimate"})
    return {"total": total, "spam": spam, "legitimate": ham}


def save_feedback(scan_id, feedback, comment=""):
    return user_feedback.insert_one({
        "scan_id": scan_id,
        "feedback": feedback,
        "comment": comment[:500],
        "timestamp": datetime.now(timezone.utc),
    })
