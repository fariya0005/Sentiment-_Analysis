from transformers import pipeline
import pandas as pd


class TwitterSentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize sentiment analyzer with 3-class sentiment model.
        Model outputs: POSITIVE, NEUTRAL, NEGATIVE
        """
        print("Initializing Twitter sentiment analyzer...")
        print("Loading model: cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.analyzer = pipeline("sentiment-analysis", model=model_name)
        print("✅ Sentiment analyzer ready!")

    def analyze_text_batch(self, texts, text_type="text"):
        """
        Analyze a batch of texts and return sentiment results.
        Handles empty texts and errors gracefully.
        """
        results = []
        for i, text in enumerate(texts):
            if pd.isna(text) or str(text).strip() == "" or str(text) == "nan":
                results.append({"label": "NEUTRAL", "score": 0.0})
            else:
                try:
                    # Truncate to 512 tokens for model
                    result = self.analyzer(str(text)[:512], truncation=True)[0]
                    results.append(result)
                except Exception as e:
                    if i < 5:  # Only print first 5 errors
                        print(f"    ⚠️ Error analyzing {text_type} {i}: {e}")
                    results.append({"label": "NEUTRAL", "score": 0.0})
        return results

    def analyze_twitter_data(self, df):
        """
        Analyze sentiment for Twitter data (tweets and replies).
        Returns: DataFrame with sentiment columns added
        """
        if df.empty:
            print("⚠️  Empty dataframe provided")
            return df

        print(f"\n{'='*60}")
        print("🤖 RUNNING TWITTER SENTIMENT ANALYSIS")
        print(f"{'='*60}")
        print(f"Analyzing sentiment for {len(df)} rows...")

        # Analyze main tweet text
        if "tweet_text" in df.columns:
            print(f"\n  → Analyzing main tweet texts...")
            tweet_texts = df["tweet_text"].fillna("").astype(str).tolist()
            tweet_results = self.analyze_text_batch(tweet_texts, "tweet")

            df["tweet_sentiment_label"] = [s["label"] for s in tweet_results]
            df["tweet_sentiment_score"] = [s["score"] for s in tweet_results]
            print(f"  ✅ Analyzed {len(tweet_results)} tweets")
        else:
            print("  ℹ️  No tweet_text column found")
            df["tweet_sentiment_label"] = "NEUTRAL"
            df["tweet_sentiment_score"] = 0.0

        # Analyze reply/interaction text
        if "text" in df.columns:
            print(f"\n  → Analyzing interaction texts (replies)...")

            # Only analyze replies (retweeters have empty text)
            reply_mask = df["interaction_type"] == "reply"
            reply_texts = df.loc[reply_mask, "text"].fillna("").astype(str).tolist()

            if len(reply_texts) > 0:
                reply_results = self.analyze_text_batch(reply_texts, "reply")

                # Initialize columns with NEUTRAL
                df["interaction_sentiment_label"] = "NEUTRAL"
                df["interaction_sentiment_score"] = 0.0

                # Fill in reply sentiments
                df.loc[reply_mask, "interaction_sentiment_label"] = [
                    s["label"] for s in reply_results
                ]
                df.loc[reply_mask, "interaction_sentiment_score"] = [
                    s["score"] for s in reply_results
                ]

                print(f"  ✅ Analyzed {len(reply_results)} replies")
            else:
                df["interaction_sentiment_label"] = "NEUTRAL"
                df["interaction_sentiment_score"] = 0.0
                print("  ℹ️  No replies to analyze")
        else:
            print("  ℹ️  No text column found")
            df["interaction_sentiment_label"] = "NEUTRAL"
            df["interaction_sentiment_score"] = 0.0

        # Print summary statistics
        self._print_sentiment_summary(df)

        print(f"{'='*60}\n")
        return df

    def _print_sentiment_summary(self, df):
        """
        Print sentiment analysis summary.
        """
        print(f"\n📊 Sentiment Analysis Summary:")

        # Tweet sentiments
        if "tweet_sentiment_label" in df.columns:
            # Get unique tweets for analysis
            unique_tweets = (
                df.drop_duplicates(subset=["tweet_id"])
                if "tweet_id" in df.columns
                else df
            )
            tweet_counts = (
                unique_tweets["tweet_sentiment_label"].value_counts().to_dict()
            )
            print(f"  📝 Main Tweet Sentiments ({len(unique_tweets)} unique tweets):")
            for sentiment, count in sorted(tweet_counts.items()):
                percentage = (count / len(unique_tweets)) * 100
                print(f"     • {sentiment}: {count} ({percentage:.1f}%)")

        # Reply sentiments (exclude retweeters)
        if (
            "interaction_sentiment_label" in df.columns
            and "interaction_type" in df.columns
        ):
            replies = df[df["interaction_type"] == "reply"]
            if len(replies) > 0:
                reply_counts = (
                    replies["interaction_sentiment_label"].value_counts().to_dict()
                )
                print(f"\n  💬 Reply Sentiments ({len(replies)} replies):")
                for sentiment, count in sorted(reply_counts.items()):
                    percentage = (count / len(replies)) * 100
                    print(f"     • {sentiment}: {count} ({percentage:.1f}%)")

        # Interaction type breakdown
        if "interaction_type" in df.columns:
            interaction_counts = df["interaction_type"].value_counts().to_dict()
            print(f"\n  🔄 Interaction Breakdown:")
            for itype, count in interaction_counts.items():
                percentage = (count / len(df)) * 100
                print(f"     • {itype}: {count} ({percentage:.1f}%)")


def analyze_twitter_sentiment(csv_file):
    """
    Standalone function to analyze sentiment of existing Twitter data.
    Can be called separately or integrated into scraping pipeline.
    """
    print(f"\n📂 Loading data from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")

        # Show detected columns
        print(f"\n📋 Detected columns:")
        key_columns = [
            col
            for col in df.columns
            if any(
                key in col.lower()
                for key in ["tweet", "text", "sentiment", "interaction", "reply"]
            )
        ]
        for col in key_columns:
            print(f"   • {col}")

        analyzer = TwitterSentimentAnalyzer()
        df = analyzer.analyze_twitter_data(df)

        # Save with sentiment
        output_file = csv_file.replace(".csv", "_with_sentiment.csv")
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n💾 Saved with sentiment to: {output_file}")

        return df
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        analyze_twitter_sentiment(sys.argv[1])
    else:
        print("\n" + "=" * 60)
        print("Twitter Sentiment Analysis")
        print("=" * 60)
        print("\nUsage: python sentiment_twitter.py <csv_file_path>")
        print("\nExample:")
        print(
            "  python sentiment_twitter.py Data/Twitter/final/username_all_tweets_20241209.csv"
        )
        print("\nAnalyzes sentiment for:")
        print("  • Main tweet texts")
        print("  • Reply texts")
        print("\nOutputs 3 sentiment classes:")
        print("  • POSITIVE")
        print("  • NEUTRAL")
        print("  • NEGATIVE")
        print("=" * 60 + "\n")
