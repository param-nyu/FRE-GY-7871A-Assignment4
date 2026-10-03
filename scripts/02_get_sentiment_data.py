"""Harvest both sentiment channels across the event window."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.nlp_sentiment import fetch_news, fetch_tweets


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    if which in ("both", "news"):
        news = fetch_news()
        print(f"\n[news] rows={len(news)} days={news['date'].nunique()} "
              f"outlets={news['domain'].nunique()}")
        print(news["bucket"].value_counts().to_string())

    if which in ("both", "tweets"):
        tweets = fetch_tweets()
        print(f"\n[tweets] rows={len(tweets)} days={tweets['date'].nunique()}")
        print(tweets.groupby("family").size().to_string())


if __name__ == "__main__":
    main()
