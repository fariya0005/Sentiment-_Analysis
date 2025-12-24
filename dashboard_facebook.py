import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime


class FacebookDashboard:
    """Interactive dashboard for Facebook sentiment analysis visualization"""

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
            print(f"   Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def find_latest_data(self):
        """Find the latest CSV file in the final directory"""
        final_dir = Path("Data/Facebook/final")
        if not final_dir.exists():
            print("❌ No data directory found!")
            return None

        csv_files = list(final_dir.glob("facebook_data_*.csv"))
        if not csv_files:
            print("❌ No data files found!")
            return None

        latest_file = max(csv_files, key=os.path.getctime)
        print(f"📁 Found latest file: {latest_file.name}")
        return latest_file

    def create_sentiment_distribution(self):
        """Create sentiment distribution charts"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Post sentiment
        post_counts = self.df["post_sentiment_label"].value_counts()
        colors = {"POSITIVE": "#2ecc71", "NEUTRAL": "#95a5a6", "NEGATIVE": "#e74c3c"}
        post_colors = [colors.get(label, "#3498db") for label in post_counts.index]

        axes[0].pie(
            post_counts.values,
            labels=post_counts.index,
            autopct="%1.1f%%",
            colors=post_colors,
            startangle=90,
        )
        axes[0].set_title("Post Sentiment Distribution", fontsize=14, fontweight="bold")

        # Comment sentiment
        comment_counts = self.df["comment_sentiment_label"].value_counts()
        comment_colors = [
            colors.get(label, "#3498db") for label in comment_counts.index
        ]

        axes[1].pie(
            comment_counts.values,
            labels=comment_counts.index,
            autopct="%1.1f%%",
            colors=comment_colors,
            startangle=90,
        )
        axes[1].set_title(
            "Comment Sentiment Distribution", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        self._save_plot("sentiment_distribution.png")
        plt.show()

    def create_engagement_analysis(self):
        """Analyze engagement metrics by sentiment"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Reactions by sentiment
        sentiment_reactions = self.df.groupby("post_sentiment_label")[
            "post_total_reactions"
        ].mean()
        axes[0, 0].bar(
            sentiment_reactions.index,
            sentiment_reactions.values,
            color=["#e74c3c", "#95a5a6", "#2ecc71"],
        )
        axes[0, 0].set_title("Average Reactions by Post Sentiment", fontweight="bold")
        axes[0, 0].set_ylabel("Average Reactions")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Comments by sentiment
        sentiment_comments = self.df.groupby("post_sentiment_label")[
            "post_total_comments"
        ].mean()
        axes[0, 1].bar(
            sentiment_comments.index,
            sentiment_comments.values,
            color=["#e74c3c", "#95a5a6", "#2ecc71"],
        )
        axes[0, 1].set_title("Average Comments by Post Sentiment", fontweight="bold")
        axes[0, 1].set_ylabel("Average Comments")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Shares by sentiment
        sentiment_shares = self.df.groupby("post_sentiment_label")[
            "post_total_shares"
        ].mean()
        axes[1, 0].bar(
            sentiment_shares.index,
            sentiment_shares.values,
            color=["#e74c3c", "#95a5a6", "#2ecc71"],
        )
        axes[1, 0].set_title("Average Shares by Post Sentiment", fontweight="bold")
        axes[1, 0].set_ylabel("Average Shares")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Total engagement
        engagement_by_sentiment = self.df.groupby("post_sentiment_label")[
            "post_engagement_total"
        ].mean()
        axes[1, 1].bar(
            engagement_by_sentiment.index,
            engagement_by_sentiment.values,
            color=["#e74c3c", "#95a5a6", "#2ecc71"],
        )
        axes[1, 1].set_title("Average Total Engagement by Sentiment", fontweight="bold")
        axes[1, 1].set_ylabel("Average Total Engagement")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        self._save_plot("engagement_analysis.png")
        plt.show()

    def create_emoji_sentiment_heatmap(self):
        """Create heatmap of emoji reactions by sentiment"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        emoji_cols = [
            "emoji_like",
            "emoji_love",
            "emoji_haha",
            "emoji_wow",
            "emoji_sad",
            "emoji_angry",
            "emoji_care",
        ]

        sentiment_emoji = self.df.groupby("post_sentiment_label")[emoji_cols].sum()

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            sentiment_emoji.T,
            annot=True,
            fmt="g",
            cmap="YlOrRd",
            cbar_kws={"label": "Count"},
        )
        plt.title("Emoji Reactions by Post Sentiment", fontsize=14, fontweight="bold")
        plt.xlabel("Sentiment")
        plt.ylabel("Emoji Type")
        plt.tight_layout()
        self._save_plot("emoji_sentiment_heatmap.png")
        plt.show()

    def create_sentiment_confidence_distribution(self):
        """Show distribution of sentiment confidence scores"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Post sentiment confidence
        for sentiment in self.df["post_sentiment_label"].unique():
            subset = self.df[self.df["post_sentiment_label"] == sentiment]
            axes[0].hist(
                subset["post_sentiment_score"], alpha=0.6, label=sentiment, bins=20
            )

        axes[0].set_title("Post Sentiment Confidence Distribution", fontweight="bold")
        axes[0].set_xlabel("Confidence Score")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Comment sentiment confidence
        for sentiment in self.df["comment_sentiment_label"].unique():
            subset = self.df[self.df["comment_sentiment_label"] == sentiment]
            axes[1].hist(
                subset["comment_sentiment_score"], alpha=0.6, label=sentiment, bins=20
            )

        axes[1].set_title(
            "Comment Sentiment Confidence Distribution", fontweight="bold"
        )
        axes[1].set_xlabel("Confidence Score")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        self._save_plot("sentiment_confidence.png")
        plt.show()

    def create_top_posts_analysis(self, top_n=10):
        """Analyze top posts by engagement"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        # Get unique posts with their metrics
        posts = self.df.drop_duplicates(subset=["post_id"]).copy()
        top_posts = posts.nlargest(top_n, "post_engagement_total")

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Top posts by engagement
        axes[0].barh(range(len(top_posts)), top_posts["post_engagement_total"].values)
        axes[0].set_yticks(range(len(top_posts)))
        axes[0].set_yticklabels([f"Post {i+1}" for i in range(len(top_posts))])
        axes[0].set_xlabel("Total Engagement")
        axes[0].set_title(f"Top {top_n} Posts by Engagement", fontweight="bold")
        axes[0].invert_yaxis()

        # Color by sentiment
        sentiment_colors = top_posts["post_sentiment_label"].map(
            {"POSITIVE": "#2ecc71", "NEUTRAL": "#95a5a6", "NEGATIVE": "#e74c3c"}
        )

        axes[1].scatter(
            top_posts["post_total_reactions"],
            top_posts["post_total_comments"],
            s=top_posts["post_total_shares"] * 10,
            c=sentiment_colors,
            alpha=0.6,
        )
        axes[1].set_xlabel("Total Reactions")
        axes[1].set_ylabel("Total Comments")
        axes[1].set_title(
            "Reactions vs Comments (bubble size = shares)", fontweight="bold"
        )
        axes[1].grid(alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#2ecc71", label="Positive"),
            Patch(facecolor="#95a5a6", label="Neutral"),
            Patch(facecolor="#e74c3c", label="Negative"),
        ]
        axes[1].legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        self._save_plot("top_posts_analysis.png")
        plt.show()

    def create_comprehensive_report(self):
        """Generate all visualizations in one go"""
        print("\n" + "=" * 70)
        print("📊 GENERATING COMPREHENSIVE DASHBOARD")
        print("=" * 70 + "\n")

        if self.df is None:
            print("❌ No data loaded!")
            return

        print("Creating visualizations...")
        print("  1/5 Sentiment Distribution...")
        self.create_sentiment_distribution()

        print("  2/5 Engagement Analysis...")
        self.create_engagement_analysis()

        print("  3/5 Emoji-Sentiment Heatmap...")
        self.create_emoji_sentiment_heatmap()

        print("  4/5 Sentiment Confidence...")
        self.create_sentiment_confidence_distribution()

        print("  5/5 Top Posts Analysis...")
        self.create_top_posts_analysis()

        print("\n✅ All visualizations generated!")
        print(f"📁 Saved to: Data/Facebook/visualizations/")
        print("=" * 70 + "\n")

    def _save_plot(self, filename):
        """Save plot to visualizations directory"""
        viz_dir = Path("Data/Facebook/visualizations")
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
        print("📊 FACEBOOK SENTIMENT ANALYSIS - SUMMARY STATISTICS")
        print("=" * 70 + "\n")

        print("DATASET OVERVIEW:")
        print(f"  Total Rows: {len(self.df)}")
        print(f"  Unique Posts: {self.df['post_id'].nunique()}")
        print(f"  Total Comments: {self.df['comment_id'].ne('').sum()}")

        print("\nPOST SENTIMENTS:")
        post_sentiments = self.df["post_sentiment_label"].value_counts()
        for sentiment, count in post_sentiments.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")

        print("\nCOMMENT SENTIMENTS:")
        comment_sentiments = self.df["comment_sentiment_label"].value_counts()
        for sentiment, count in comment_sentiments.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")

        print("\nENGAGEMENT METRICS:")
        print(f"  Total Reactions: {self.df['post_total_reactions'].sum():,}")
        print(f"  Total Comments: {self.df['post_total_comments'].sum():,}")
        print(f"  Total Shares: {self.df['post_total_shares'].sum():,}")
        print(f"  Avg Post Engagement: {self.df['post_engagement_total'].mean():.2f}")

        print("\nTOP PERFORMING SENTIMENT:")
        avg_engagement_by_sentiment = self.df.groupby("post_sentiment_label")[
            "post_engagement_total"
        ].mean()
        top_sentiment = avg_engagement_by_sentiment.idxmax()
        print(
            f"  {top_sentiment} posts have highest avg engagement: {avg_engagement_by_sentiment[top_sentiment]:.2f}"
        )

        print("\n" + "=" * 70 + "\n")


def run_dashboard(csv_file=None):
    """
    Main function to run the dashboard.
    If no file is provided, it will find the latest data file.
    """
    print("\n" + "=" * 70)
    print("FACEBOOK SENTIMENT ANALYSIS DASHBOARD".center(70))
    print("=" * 70 + "\n")

    dashboard = FacebookDashboard(csv_file)

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
    print("  3. Emoji-Sentiment Heatmap")
    print("  4. Sentiment Confidence Distribution")
    print("  5. Top Posts Analysis")
    print("  6. Generate All Reports")
    print("  7. Exit")

    while True:
        choice = input("\nSelect option (1-7): ").strip()

        if choice == "1":
            dashboard.create_sentiment_distribution()
        elif choice == "2":
            dashboard.create_engagement_analysis()
        elif choice == "3":
            dashboard.create_emoji_sentiment_heatmap()
        elif choice == "4":
            dashboard.create_sentiment_confidence_distribution()
        elif choice == "5":
            dashboard.create_top_posts_analysis()
        elif choice == "6":
            dashboard.create_comprehensive_report()
            break
        elif choice == "7":
            print("👋 Exiting dashboard...")
            break
        else:
            print("❌ Invalid option. Please select 1-7.")


if __name__ == "__main__":
    run_dashboard()
