
import requests
import pandas as pd
from urllib.parse import urlsplit

FEED_URL = (
    "https://raw.githubusercontent.com/"
    "openphish/public_feed/main/feed.txt"
)

def load_phishing_feed():
    response = requests.get(FEED_URL, timeout=30)
    response.raise_for_status()

    urls = [
        line.strip()
        for line in response.text.splitlines()
        if line.strip()
    ]

    df = pd.DataFrame({"url": urls})
    df = df.drop_duplicates(subset="url")

    df["hostname"] = df["url"].apply(
        lambda url: urlsplit(url).hostname
    )

    return df

def find_related_hosts(df):
    groups = (
        df.groupby("hostname")
        .agg(url_count=("url", "count"))
        .reset_index()
    )

    return groups[
        groups["url_count"] > 1
    ].sort_values("url_count", ascending=False)

if __name__ == "__main__":
    df = load_phishing_feed()
    groups = find_related_hosts(df)

    print("Total phishing URLs:", len(df))
    print("Unique hostnames:", df["hostname"].nunique())
    print("Repeated hostname groups:", len(groups))
    print("\nTop hostname groups:")
    print(groups.head(5).to_string(index=False))
