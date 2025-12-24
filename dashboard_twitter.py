import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


class TwitterDashboard:
    """Interactive dashboard for Twitter sentiment analysis visualization"""

    def __init__(self, csv_file=None):
        self.df = None
        self.csv_file = csv_file

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (15, 10)

        if csv_file:
            self.load_data(csv_file)

    def load_data(self, csv_file):
        """Load data from CSV file"""
        try:
            self.df = pd.read_csv(csv_file)
            self.csv_file = csv_file

            print(f"✅ Loaded data from: {csv_file}")
            print(f"   Total rows: {len(self.df)}")
            print(f"   Columns: {len(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def find_latest_data(self):
        """Find the latest CSV file in the final directory"""
        final_dir = Path("Data/Twitter/final")
        if not final_dir.exists():
            print("❌ No data directory found!")
            return None

        csv_files = list(final_dir.glob("*.csv"))
        if not csv_files:
            print("❌ No data files found!")
            return None

        latest_file = max(csv_files, key=os.path.getctime)
        print(f"📁 Found latest file: {latest_file.name}")
        return latest_file

    def create_sentiment_distribution(self):
        """Create sentiment distribution charts for tweets and replies"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        colors = {"POSITIVE": "#2ecc71", "NEUTRAL": "#95a5a6", "NEGATIVE": "#e74c3c"}

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Tweet sentiment (unique tweets only)
        if "tweet_sentiment_label" in self.df.columns:
            unique_tweets = (
                self.df.drop_duplicates(subset=["tweet_id"])
                if "tweet_id" in self.df.columns
                else self.df
            )
            tweet_counts = unique_tweets["tweet_sentiment_label"].value_counts()
            tweet_colors = [
                colors.get(label, "#3498db") for label in tweet_counts.index
            ]

            axes[0].pie(
                tweet_counts.values,
                labels=tweet_counts.index,
                autopct="%1.1f%%",
                colors=tweet_colors,
                startangle=90,
            )
            axes[0].set_title(
                f"Main Tweet Sentiment Distribution\n({len(unique_tweets)} tweets)",
                fontsize=14,
                fontweight="bold",
            )
        else:
            axes[0].text(0.5, 0.5, "No tweet sentiment data", ha="center", va="center")
            axes[0].set_title(
                "Tweet Sentiment Distribution", fontsize=14, fontweight="bold"
            )

        # Reply sentiment (replies only)
        if (
            "interaction_sentiment_label" in self.df.columns
            and "interaction_type" in self.df.columns
        ):
            replies = self.df[self.df["interaction_type"] == "reply"]

            if len(replies) > 0:
                reply_counts = replies["interaction_sentiment_label"].value_counts()
                reply_colors = [
                    colors.get(label, "#3498db") for label in reply_counts.index
                ]

                axes[1].pie(
                    reply_counts.values,
                    labels=reply_counts.index,
                    autopct="%1.1f%%",
                    colors=reply_colors,
                    startangle=90,
                )
                axes[1].set_title(
                    f"Reply Sentiment Distribution\n({len(replies)} replies)",
                    fontsize=14,
                    fontweight="bold",
                )
            else:
                axes[1].text(0.5, 0.5, "No replies available", ha="center", va="center")
                axes[1].set_title(
                    "Reply Sentiment Distribution", fontsize=14, fontweight="bold"
                )
        else:
            axes[1].text(0.5, 0.5, "No reply sentiment data", ha="center", va="center")
            axes[1].set_title(
                "Reply Sentiment Distribution", fontsize=14, fontweight="bold"
            )

        plt.tight_layout()
        self._save_plot("sentiment_distribution.png")
        plt.show()

    def create_engagement_analysis(self):
        """Analyze engagement metrics by sentiment"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        # Filter only replies for engagement analysis
        replies = self.df[self.df["interaction_type"] == "reply"].copy()

        if replies.empty or "interaction_sentiment_label" not in replies.columns:
            print("❌ No reply data available for engagement analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        sentiment_order = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

        # Likes by reply sentiment
        if "like_count" in replies.columns:
            sentiment_likes = replies.groupby("interaction_sentiment_label")[
                "like_count"
            ].mean()
            sentiment_likes = sentiment_likes.reindex(
                [s for s in sentiment_order if s in sentiment_likes.index]
            )

            axes[0, 0].bar(
                sentiment_likes.index,
                sentiment_likes.values,
                color=["#e74c3c", "#95a5a6", "#2ecc71"][: len(sentiment_likes)],
            )
            axes[0, 0].set_title("Average Likes by Reply Sentiment", fontweight="bold")
            axes[0, 0].set_ylabel("Average Likes")
            axes[0, 0].tick_params(axis="x", rotation=45)

        # Retweets by reply sentiment
        if "retweet_count" in replies.columns:
            sentiment_retweets = replies.groupby("interaction_sentiment_label")[
                "retweet_count"
            ].mean()
            sentiment_retweets = sentiment_retweets.reindex(
                [s for s in sentiment_order if s in sentiment_retweets.index]
            )

            axes[0, 1].bar(
                sentiment_retweets.index,
                sentiment_retweets.values,
                color=["#e74c3c", "#95a5a6", "#2ecc71"][: len(sentiment_retweets)],
            )
            axes[0, 1].set_title(
                "Average Retweets by Reply Sentiment", fontweight="bold"
            )
            axes[0, 1].set_ylabel("Average Retweets")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # Reply count by sentiment
        if "reply_count" in replies.columns:
            sentiment_replies = replies.groupby("interaction_sentiment_label")[
                "reply_count"
            ].mean()
            sentiment_replies = sentiment_replies.reindex(
                [s for s in sentiment_order if s in sentiment_replies.index]
            )

            axes[1, 0].bar(
                sentiment_replies.index,
                sentiment_replies.values,
                color=["#e74c3c", "#95a5a6", "#2ecc71"][: len(sentiment_replies)],
            )
            axes[1, 0].set_title(
                "Average Replies by Reply Sentiment", fontweight="bold"
            )
            axes[1, 0].set_ylabel("Average Replies")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # Sentiment confidence distribution
        if "interaction_sentiment_score" in replies.columns:
            for sentiment in ["NEGATIVE", "NEUTRAL", "POSITIVE"]:
                if sentiment in replies["interaction_sentiment_label"].unique():
                    subset = replies[
                        replies["interaction_sentiment_label"] == sentiment
                    ]
                    color = {
                        "POSITIVE": "#2ecc71",
                        "NEUTRAL": "#95a5a6",
                        "NEGATIVE": "#e74c3c",
                    }[sentiment]
                    axes[1, 1].hist(
                        subset["interaction_sentiment_score"],
                        alpha=0.6,
                        label=sentiment,
                        bins=15,
                        color=color,
                    )

            axes[1, 1].set_title(
                "Reply Sentiment Confidence Distribution", fontweight="bold"
            )
            axes[1, 1].set_xlabel("Confidence Score")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        self._save_plot("engagement_analysis.png")
        plt.show()

    def create_interaction_analysis(self):
        """Analyze interaction types and their sentiments"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Interaction type distribution
        if "interaction_type" in self.df.columns:
            interaction_counts = self.df["interaction_type"].value_counts()
            colors_map = {"reply": "#3498db", "retweeter": "#9b59b6"}
            bar_colors = [
                colors_map.get(itype, "#95a5a6") for itype in interaction_counts.index
            ]

            axes[0, 0].bar(
                interaction_counts.index, interaction_counts.values, color=bar_colors
            )
            axes[0, 0].set_title("Interaction Type Distribution", fontweight="bold")
            axes[0, 0].set_ylabel("Count")
            axes[0, 0].tick_params(axis="x", rotation=45)

        # Sentiment by interaction type
        if (
            "interaction_type" in self.df.columns
            and "interaction_sentiment_label" in self.df.columns
        ):
            # Filter to only replies (retweeters don't have sentiment)
            replies = self.df[self.df["interaction_type"] == "reply"]

            if len(replies) > 0:
                sentiment_by_type = (
                    replies.groupby(["interaction_type", "interaction_sentiment_label"])
                    .size()
                    .unstack(fill_value=0)
                )

                sentiment_by_type.plot(
                    kind="bar",
                    stacked=True,
                    ax=axes[0, 1],
                    color={
                        "POSITIVE": "#2ecc71",
                        "NEUTRAL": "#95a5a6",
                        "NEGATIVE": "#e74c3c",
                    },
                )
                axes[0, 1].set_title(
                    "Sentiment Distribution by Interaction Type", fontweight="bold"
                )
                axes[0, 1].set_xlabel("Interaction Type")
                axes[0, 1].set_ylabel("Count")
                axes[0, 1].legend(title="Sentiment")
                axes[0, 1].tick_params(axis="x", rotation=45)

        # Top repliers by followers
        if "interaction_type" in self.df.columns and "followers" in self.df.columns:
            replies = self.df[self.df["interaction_type"] == "reply"]

            if len(replies) > 0 and "username" in replies.columns:
                top_repliers = (
                    replies.groupby("username")["followers"].first().nlargest(10)
                )

                axes[1, 0].barh(
                    range(len(top_repliers)), top_repliers.values, color="#e67e22"
                )
                axes[1, 0].set_yticks(range(len(top_repliers)))
                axes[1, 0].set_yticklabels(top_repliers.index)
                axes[1, 0].set_xlabel("Followers")
                axes[1, 0].set_title(
                    "Top 10 Repliers by Follower Count", fontweight="bold"
                )
                axes[1, 0].invert_yaxis()

        # Verified vs non-verified distribution
        if "verified" in self.df.columns and "interaction_type" in self.df.columns:
            verified_counts = (
                self.df.groupby(["interaction_type", "verified"])
                .size()
                .unstack(fill_value=0)
            )

            verified_counts.plot(
                kind="bar",
                ax=axes[1, 1],
                color={True: "#3498db", False: "#95a5a6"},
            )
            axes[1, 1].set_title(
                "Verified vs Non-Verified by Interaction Type", fontweight="bold"
            )
            axes[1, 1].set_xlabel("Interaction Type")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].legend(title="Verified", labels=["Not Verified", "Verified"])
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        self._save_plot("interaction_analysis.png")
        plt.show()

    def create_tweet_comparison(self):
        """Compare multiple tweets if available"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        if "tweet_id" not in self.df.columns:
            print("❌ No tweet_id column found!")
            return

        unique_tweets = self.df["tweet_id"].nunique()

        if unique_tweets <= 1:
            print("ℹ️  Only one tweet available, skipping comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Replies per tweet
        replies = self.df[self.df["interaction_type"] == "reply"]
        replies_per_tweet = replies.groupby("tweet_id").size()

        axes[0, 0].bar(
            range(len(replies_per_tweet)), replies_per_tweet.values, color="#3498db"
        )
        axes[0, 0].set_xticks(range(len(replies_per_tweet)))
        axes[0, 0].set_xticklabels(
            [f"Tweet {i+1}" for i in range(len(replies_per_tweet))], rotation=45
        )
        axes[0, 0].set_ylabel("Number of Replies")
        axes[0, 0].set_title("Replies per Tweet", fontweight="bold")

        # Retweeters per tweet
        retweeters = self.df[self.df["interaction_type"] == "retweeter"]
        retweets_per_tweet = retweeters.groupby("tweet_id").size()

        axes[0, 1].bar(
            range(len(retweets_per_tweet)), retweets_per_tweet.values, color="#9b59b6"
        )
        axes[0, 1].set_xticks(range(len(retweets_per_tweet)))
        axes[0, 1].set_xticklabels(
            [f"Tweet {i+1}" for i in range(len(retweets_per_tweet))], rotation=45
        )
        axes[0, 1].set_ylabel("Number of Retweeters")
        axes[0, 1].set_title("Retweeters per Tweet", fontweight="bold")

        # Reply sentiment by tweet
        if "interaction_sentiment_label" in replies.columns:
            sentiment_by_tweet = (
                replies.groupby(["tweet_id", "interaction_sentiment_label"])
                .size()
                .unstack(fill_value=0)
            )

            sentiment_by_tweet.plot(
                kind="bar",
                stacked=True,
                ax=axes[1, 0],
                color={
                    "POSITIVE": "#2ecc71",
                    "NEUTRAL": "#95a5a6",
                    "NEGATIVE": "#e74c3c",
                },
            )
            axes[1, 0].set_title("Reply Sentiment by Tweet", fontweight="bold")
            axes[1, 0].set_xlabel("Tweet ID")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].legend(title="Sentiment")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # Average engagement by tweet
        if "like_count" in replies.columns:
            avg_engagement = replies.groupby("tweet_id")["like_count"].mean()

            axes[1, 1].bar(
                range(len(avg_engagement)), avg_engagement.values, color="#e67e22"
            )
            axes[1, 1].set_xticks(range(len(avg_engagement)))
            axes[1, 1].set_xticklabels(
                [f"Tweet {i+1}" for i in range(len(avg_engagement))], rotation=45
            )
            axes[1, 1].set_ylabel("Average Likes")
            axes[1, 1].set_title("Average Reply Likes by Tweet", fontweight="bold")

        plt.tight_layout()
        self._save_plot("tweet_comparison.png")
        plt.show()

    def create_comprehensive_report(self):
        """Generate all visualizations in one go"""
        print("\n" + "=" * 70)
        print("📊 GENERATING COMPREHENSIVE TWITTER DASHBOARD")
        print("=" * 70 + "\n")

        if self.df is None:
            print("❌ No data loaded!")
            return

        print("Creating visualizations...")
        print("  1/4 Sentiment Distribution...")
        self.create_sentiment_distribution()

        print("  2/4 Engagement Analysis...")
        self.create_engagement_analysis()

        print("  3/4 Interaction Analysis...")
        self.create_interaction_analysis()

        print("  4/4 Tweet Comparison...")
        self.create_tweet_comparison()

        print("\n✅ All visualizations generated!")
        print(f"📁 Saved to: Data/Twitter/visualizations/")
        print("=" * 70 + "\n")

    def _save_plot(self, filename):
        """Save plot to visualizations directory"""
        viz_dir = Path("Data/Twitter/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)

        filepath = viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {filename}")

    def print_summary_statistics(self):
        """Print detailed summary statistics"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        print("\n" + "=" * 70)
        print("📊 TWITTER SENTIMENT ANALYSIS - SUMMARY STATISTICS")
        print("=" * 70 + "\n")

        print("DATASET OVERVIEW:")
        print(f"  Total Rows: {len(self.df)}")

        if "tweet_id" in self.df.columns:
            unique_tweets = self.df["tweet_id"].nunique()
            print(f"  Unique Tweets: {unique_tweets}")

        if "interaction_type" in self.df.columns:
            interaction_counts = self.df["interaction_type"].value_counts()
            print(f"\n  Interaction Breakdown:")
            for itype, count in interaction_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"    • {itype}: {count} ({percentage:.1f}%)")

        print("\nMAIN TWEET SENTIMENTS:")
        if "tweet_sentiment_label" in self.df.columns:
            unique_tweets = (
                self.df.drop_duplicates(subset=["tweet_id"])
                if "tweet_id" in self.df.columns
                else self.df
            )
            tweet_sentiments = unique_tweets["tweet_sentiment_label"].value_counts()
            for sentiment in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
                if sentiment in tweet_sentiments.index:
                    count = tweet_sentiments[sentiment]
                    percentage = (count / len(unique_tweets)) * 100
                    print(f"  {sentiment}: {count} ({percentage:.1f}%)")
        else:
            print("  No tweet sentiment data available")

        print("\nREPLY SENTIMENTS:")
        if (
            "interaction_sentiment_label" in self.df.columns
            and "interaction_type" in self.df.columns
        ):
            replies = self.df[self.df["interaction_type"] == "reply"]
            if len(replies) > 0:
                reply_sentiments = replies["interaction_sentiment_label"].value_counts()
                for sentiment in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
                    if sentiment in reply_sentiments.index:
                        count = reply_sentiments[sentiment]
                        percentage = (count / len(replies)) * 100
                        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
            else:
                print("  No replies available")
        else:
            print("  No reply sentiment data available")

        print("\nENGAGEMENT METRICS:")
        replies = (
            self.df[self.df["interaction_type"] == "reply"]
            if "interaction_type" in self.df.columns
            else self.df
        )

        if "like_count" in replies.columns and len(replies) > 0:
            avg_likes = replies["like_count"].mean()
            total_likes = replies["like_count"].sum()
            print(f"  Total Reply Likes: {total_likes:,.0f}")
            print(f"  Average Likes per Reply: {avg_likes:.2f}")

        if "retweet_count" in replies.columns and len(replies) > 0:
            avg_retweets = replies["retweet_count"].mean()
            total_retweets = replies["retweet_count"].sum()
            print(f"  Total Reply Retweets: {total_retweets:,.0f}")
            print(f"  Average Retweets per Reply: {avg_retweets:.2f}")

        print("\n" + "=" * 70 + "\n")


def run_twitter_dashboard(csv_file=None):
    """
    Main function to run the Twitter dashboard.
    If no file is provided, it will find the latest data file.
    """
    print("\n" + "=" * 70)
    print("TWITTER SENTIMENT ANALYSIS DASHBOARD".center(70))
    print("=" * 70 + "\n")

    dashboard = TwitterDashboard(csv_file)

    # If no file provided, find the latest one
    if dashboard.df is None:
        print("🔍 Searching for latest data file...")
        latest_file = dashboard.find_latest_data()
        if latest_file:
            dashboard.load_data(latest_file)
        else:
            print("❌ No data available. Please run the scraper first!")
            return

    # Print summary
    dashboard.print_summary_statistics()

    # Ask user what to generate
    print("📊 Visualization Options:")
    print("  1. Sentiment Distribution")
    print("  2. Engagement Analysis")
    print("  3. Interaction Analysis")
    print("  4. Tweet Comparison")
    print("  5. Generate All Reports")
    print("  6. Exit")

    while True:
        choice = input("\nSelect option (1-6): ").strip()

        if choice == "1":
            dashboard.create_sentiment_distribution()
        elif choice == "2":
            dashboard.create_engagement_analysis()
        elif choice == "3":
            dashboard.create_interaction_analysis()
        elif choice == "4":
            dashboard.create_tweet_comparison()
        elif choice == "5":
            dashboard.create_comprehensive_report()
            break
        elif choice == "6":
            print("👋 Exiting dashboard...")
            break
        else:
            print("❌ Invalid option. Please select 1-6.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_twitter_dashboard(sys.argv[1])
    else:
        run_twitter_dashboard()
