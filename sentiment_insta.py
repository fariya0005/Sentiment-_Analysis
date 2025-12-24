from transformers import pipeline
import pandas as pd


class InstagramSentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize sentiment analyzer with 3-class sentiment model.
        Model outputs: POSITIVE, NEUTRAL, NEGATIVE
        """
        print("Initializing Instagram sentiment analyzer...")
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

    def analyze_instagram_data(self, df):
        """
        Analyze sentiment for Instagram posts and comments.
        Handles different scraping modes:
        - Profile scraping: post_caption, comment_text
        - Keyword scraping: post_caption, all_comments_text
        - URL scraping: post_caption, comment_text
        Returns: DataFrame with sentiment columns added
        """
        if df.empty:
            print("⚠️  Empty dataframe provided")
            return df

        print(f"\n{'='*60}")
        print("🤖 RUNNING INSTAGRAM SENTIMENT ANALYSIS")
        print(f"{'='*60}")
        print(f"Analyzing sentiment for {len(df)} rows...")

        # Detect scraping mode based on columns
        scraping_mode = self._detect_scraping_mode(df)
        print(f"  📋 Detected scraping mode: {scraping_mode}")

        # Analyze post captions
        if "post_caption" in df.columns:
            print(f"\n  → Analyzing post captions...")
            caption_texts = df["post_caption"].fillna("").astype(str).tolist()
            caption_results = self.analyze_text_batch(caption_texts, "caption")

            df["caption_sentiment_label"] = [s["label"] for s in caption_results]
            df["caption_sentiment_score"] = [s["score"] for s in caption_results]
            print(f"  ✅ Analyzed {len(caption_results)} captions")
        else:
            print("  ℹ️  No post_caption column found")
            df["caption_sentiment_label"] = "NEUTRAL"
            df["caption_sentiment_score"] = 0.0

        # Analyze comments based on scraping mode
        if scraping_mode == "profile" or scraping_mode == "post_url":
            # Individual comments in comment_text column
            if "comment_text" in df.columns:
                print(f"\n  → Analyzing individual comments...")
                comment_texts = df["comment_text"].fillna("").astype(str).tolist()
                comment_results = self.analyze_text_batch(comment_texts, "comment")

                df["comment_sentiment_label"] = [s["label"] for s in comment_results]
                df["comment_sentiment_score"] = [s["score"] for s in comment_results]
                print(f"  ✅ Analyzed {len(comment_results)} comments")
            else:
                print("  ℹ️  No comment_text column found")
                df["comment_sentiment_label"] = "NEUTRAL"
                df["comment_sentiment_score"] = 0.0

        elif scraping_mode == "keyword":
            # Aggregated comments in all_comments_text column
            if "all_comments_text" in df.columns:
                print(f"\n  → Analyzing aggregated comments...")
                comment_texts = df["all_comments_text"].fillna("").astype(str).tolist()
                comment_results = self.analyze_text_batch(
                    comment_texts, "aggregated comment"
                )

                df["comments_sentiment_label"] = [s["label"] for s in comment_results]
                df["comments_sentiment_score"] = [s["score"] for s in comment_results]
                print(f"  ✅ Analyzed {len(comment_results)} aggregated comment texts")
            else:
                print("  ℹ️  No all_comments_text column found")
                df["comments_sentiment_label"] = "NEUTRAL"
                df["comments_sentiment_score"] = 0.0

        # Print summary statistics
        self._print_sentiment_summary(df, scraping_mode)

        print(f"{'='*60}\n")
        return df

    def _detect_scraping_mode(self, df):
        """
        Detect which scraping mode was used based on column names.
        """
        if "source_type" in df.columns:
            # Use source_type if available
            mode = df["source_type"].iloc[0] if len(df) > 0 else "unknown"
            return mode

        # Fallback: detect by column structure
        if "comment_text" in df.columns and "profile_username" in df.columns:
            return "profile"
        elif (
            "all_comments_text" in df.columns and "comments_scraped_count" in df.columns
        ):
            return "keyword"
        elif "comment_text" in df.columns and "comment_id" in df.columns:
            return "post_url"
        else:
            return "unknown"

    def _print_sentiment_summary(self, df, scraping_mode):
        """
        Print sentiment analysis summary based on scraping mode.
        """
        print(f"\n📊 Sentiment Analysis Summary:")

        # Caption sentiments (all modes)
        if "caption_sentiment_label" in df.columns:
            caption_counts = df["caption_sentiment_label"].value_counts().to_dict()
            print(f"  📝 Post Caption Sentiments:")
            for sentiment, count in sorted(caption_counts.items()):
                percentage = (count / len(df)) * 100
                print(f"     • {sentiment}: {count} ({percentage:.1f}%)")

        # Comment sentiments (varies by mode)
        if scraping_mode in ["profile", "post_url"]:
            if "comment_sentiment_label" in df.columns:
                # Filter out rows with no comments
                with_comments = df[
                    df["comment_text"].notna() & (df["comment_text"] != "")
                ]
                if len(with_comments) > 0:
                    comment_counts = (
                        with_comments["comment_sentiment_label"]
                        .value_counts()
                        .to_dict()
                    )
                    print(
                        f"\n  💬 Individual Comment Sentiments ({len(with_comments)} comments):"
                    )
                    for sentiment, count in sorted(comment_counts.items()):
                        percentage = (count / len(with_comments)) * 100
                        print(f"     • {sentiment}: {count} ({percentage:.1f}%)")
                else:
                    print(f"\n  💬 No comments found to analyze")

        elif scraping_mode == "keyword":
            if "comments_sentiment_label" in df.columns:
                # Filter out rows with no comments
                with_comments = df[
                    df["all_comments_text"].notna() & (df["all_comments_text"] != "")
                ]
                if len(with_comments) > 0:
                    comment_counts = (
                        with_comments["comments_sentiment_label"]
                        .value_counts()
                        .to_dict()
                    )
                    print(
                        f"\n  💬 Aggregated Comments Sentiments ({len(with_comments)} posts with comments):"
                    )
                    for sentiment, count in sorted(comment_counts.items()):
                        percentage = (count / len(with_comments)) * 100
                        print(f"     • {sentiment}: {count} ({percentage:.1f}%)")
                else:
                    print(f"\n  💬 No comments found to analyze")


def analyze_instagram_sentiment(csv_file):
    """
    Standalone function to analyze sentiment of existing Instagram data.
    Can be called separately or integrated into scraping pipeline.
    Supports all scraping modes: profile, keyword, post_url
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
                for key in ["caption", "comment", "sentiment", "post", "source"]
            )
        ]
        for col in key_columns:
            print(f"   • {col}")

        analyzer = InstagramSentimentAnalyzer()
        df = analyzer.analyze_instagram_data(df)

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
        analyze_instagram_sentiment(sys.argv[1])
    else:
        print("\n" + "=" * 60)
        print("Instagram Sentiment Analysis")
        print("=" * 60)
        print("\nUsage: python sentiment_insta.py <csv_file_path>")
        print("\nExample:")
        print(
            "  python sentiment_insta.py Data/Instagram/final/profile_username_20241209.csv"
        )
        print("\nSupports data from all scraping modes:")
        print("  • Profile scraping")
        print("  • Keyword/hashtag scraping")
        print("  • Post URL scraping")
        print("=" * 60 + "\n")
