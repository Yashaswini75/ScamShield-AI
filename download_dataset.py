"""Download and normalize the UCI SMS Spam Collection dataset.
Dataset source: UCI Machine Learning Repository, DOI 10.24432/C5CC84.
"""
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT = DATA_DIR / "spam_dataset.csv"
URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
]

last_error = None
for url in URLS:
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            archive_bytes = response.read()
        with ZipFile(BytesIO(archive_bytes)) as archive:
            filename = next(
                name for name in archive.namelist()
                if name.lower().endswith("smsspamcollection")
            )
            raw = archive.read(filename).decode("utf-8", errors="replace")
        rows = []
        for line in raw.splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0].strip().lower() in {"ham", "spam"}:
                rows.append({"label": parts[0].strip().lower(), "message": parts[1].strip()})
        df = pd.DataFrame(rows).drop_duplicates(subset=["label", "message"])
        df = df[df["message"].str.len() > 0]
        if len(df) < 1000:
            raise ValueError(f"Only {len(df)} rows were parsed; refusing to replace dataset.")
        df.to_csv(OUTPUT, index=False, encoding="utf-8")
        print(f"Saved {len(df):,} labeled messages to {OUTPUT}")
        print("Label counts:")
        print(df["label"].value_counts().to_string())
        print("\nDataset source: UCI SMS Spam Collection (CC BY 4.0).")
        print("Citation: Almeida, T. & Hidalgo, J. (2011), DOI: 10.24432/C5CC84")
        break
    except Exception as exc:
        last_error = exc
else:
    raise SystemExit(
        "Dataset download failed. Check your internet connection and try again. "
        f"Last error: {last_error}"
    )
