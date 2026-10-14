# ScamShield — ML Spam Message Detection with MongoDB

A Streamlit prototype using TF-IDF + Multinomial Naive Bayes and MongoDB for opt-in scan history and user feedback.

## Dataset note

The included `data/sample_dataset_20_rows.csv` is only a tiny sample. Download the larger UCI SMS Spam Collection before training. It contains 5,574 labeled SMS messages (ham/spam), and is not a dedicated phishing-email dataset.

Source: Almeida, T. & Hidalgo, J. (2011), UCI Machine Learning Repository, https://doi.org/10.24432/C5CC84, CC BY 4.0.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_dataset.py
python train_model.py
```

## Start MongoDB Community Server

Install MongoDB Community Server if needed, then start the Windows service (service name can differ by installation):

```powershell
Get-Service *Mongo*
Start-Service MongoDB
```

If the service is already running, do not start it again. The app defaults to `mongodb://127.0.0.1:27017/`. You can override it with the `MONGO_URI` environment variable.

Check connection:

```powershell
python -c "from database import check_connection; print(check_connection())"
```

## Run the app

```powershell
streamlit run app.py
```

## MongoDB collections

- `scan_history`: saved scans, predictions, model confidence, warning signs, timestamp.
- `user_feedback`: whether the prediction was correct and optional comment.
- `threat_patterns`: reserved for future curated indicators; not yet used for automatic detection.

## Evaluation

The training script prints accuracy, precision, recall, F1 and confusion matrix, and saves metrics to `models/evaluation_metrics.json`. Prioritize spam recall and inspect false positives/false negatives, not accuracy alone.

## Limitations

- SMS spam classification does not guarantee phishing detection or general email classification.
- Model confidence is not a calibrated probability of real-world danger.
- Keyword/URL warning signs are heuristics, not a model explanation.
- The app never opens pasted URLs. Do not paste passwords, OTPs or sensitive personal information.

## Project structure

```text
ScamDetection/
├── app.py
├── database.py
├── download_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_dataset_20_rows.csv
└── models/
```
