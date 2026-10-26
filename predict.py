
import ipaddress
import joblib
import pandas as pd
from pathlib import Path
from urllib.parse import urlsplit

FEATURES = [
    "url_length", "domain_length", "is_ip",
    "dot_count", "hyphen_count", "digit_count",
    "letter_count", "special_char_count",
    "at_count", "question_count",
    "equals_count", "uses_https"
]

def extract_features(url):
    url = str(url).strip()
    parsed = urlsplit(url)
    domain = parsed.hostname or ""

    try:
        ipaddress.ip_address(domain)
        is_ip = 1
    except ValueError:
        is_ip = 0

    return {
        "url_length": len(url),
        "domain_length": len(domain),
        "is_ip": is_ip,
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "digit_count": sum(c.isdigit() for c in url),
        "letter_count": sum(c.isalpha() for c in url),
        "special_char_count": sum(not c.isalnum() for c in url),
        "at_count": url.count("@"),
        "question_count": url.count("?"),
        "equals_count": url.count("="),
        "uses_https": int(parsed.scheme.lower() == "https")
    }

def predict_url(url):
    parsed = urlsplit(str(url).strip())
    if (
        parsed.scheme.lower() not in ("http", "https")
        or not parsed.hostname
        or any(c.isspace() for c in url)
    ):
        raise ValueError("Enter a valid HTTP or HTTPS URL.")

    model_path = Path("artifacts/urlsentinel_model.pkl")
    if not model_path.exists():
        model_path = Path("urlsentinel_model.pkl")

    model_data = joblib.load(model_path)

    model = model_data["model"] if isinstance(model_data, dict) else model_data
    feature_names = (
        model_data.get("features", FEATURES)
        if isinstance(model_data, dict)
        else FEATURES
    )

    values = extract_features(url)
    row = pd.DataFrame([values], columns=feature_names)

    prediction = int(model.predict(row)[0])
    classes = list(model.classes_)
    phishing_score = float(model.predict_proba(row)[0][classes.index(0)])

    return {
        "url": url,
        "prediction": "Potential Phishing" if prediction == 0 else "Likely Legitimate",
        "phishing_score": round(phishing_score * 100, 2),
        "features": values
    }

if __name__ == "__main__":
    test_url = "https://www.google.com"
    result = predict_url(test_url)

    print("URL:", result["url"])
    print("Prediction:", result["prediction"])
    print("Experimental phishing score:", result["phishing_score"], "%")
