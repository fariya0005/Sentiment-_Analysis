from transformers import pipeline
import pandas as pd


class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize sentiment analyzer with 3-class sentiment model.
        Model outputs: POSITIVE, NEUTRAL, NEGATIVE
        """
        print("Initializing 3-class sentiment analyzer...")
        print("Loading model: cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.analyzer = pipeline("sentiment-analysis", model=model_name)
        print("✅ Sentiment analyzer ready!")

    def analyze_posts_and_comments(
        self, df, post_col="post_message", comment_col="comment_text"
    ):
        """
        Analyze sentiment for both posts and comments.
        Returns: POSITIVE, NEUTRAL, NEGATIVE with confidence scores
        """
        if df.empty:
            return df

        print(f"Analyzing sentiment for {len(df)} rows...")

        # Post sentiment
        print("  → Analyzing post sentiments...")
        post_texts = df[post_col].fillna("").astype(str).tolist()

        # Filter out empty texts
        post_results = []
        for i, text in enumerate(post_texts):
            if text.strip() == "" or text == "nan":
                post_results.append({"label": "NEUTRAL", "score": 0.0})
            else:
                try:
                    result = self.analyzer(text[:512], truncation=True)[
                        0
                    ]  # Limit text length
                    post_results.append(result)
                except Exception as e:
                    print(f"    ⚠️ Error analyzing post {i}: {e}")
                    post_results.append({"label": "NEUTRAL", "score": 0.0})

        df["post_sentiment_label"] = [s["label"] for s in post_results]
        df["post_sentiment_score"] = [s["score"] for s in post_results]

        # Comment sentiment
        print("  → Analyzing comment sentiments...")
        comment_texts = df[comment_col].fillna("").astype(str).tolist()

        comment_results = []
        for i, text in enumerate(comment_texts):
            if text.strip() == "" or text == "nan":
                comment_results.append({"label": "NEUTRAL", "score": 0.0})
            else:
                try:
                    result = self.analyzer(text[:512], truncation=True)[0]
                    comment_results.append(result)
                except Exception as e:
                    print(f"    ⚠️ Error analyzing comment {i}: {e}")
                    comment_results.append({"label": "NEUTRAL", "score": 0.0})

        df["comment_sentiment_label"] = [s["label"] for s in comment_results]
        df["comment_sentiment_score"] = [s["score"] for s in comment_results]

        # Print summary
        print("\n📊 Sentiment Analysis Summary:")
        print(
            f"  Post Sentiments: {df['post_sentiment_label'].value_counts().to_dict()}"
        )
        print(
            f"  Comment Sentiments: {df['comment_sentiment_label'].value_counts().to_dict()}"
        )

        return df
