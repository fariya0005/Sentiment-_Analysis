import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime


class InstagramDashboard:
    """Interactive dashboard for Instagram sentiment analysis visualization"""

    def __init__(self, csv_file=None):
        self.df = None
        self.csv_file = csv_file
        self.scraping_mode = None

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (15, 10)

        if csv_file:
            self.load_data(csv_file)

    def load_data(self, csv_file):
        """Load data from CSV file and detect scraping mode"""
        try:
            self.df = pd.read_csv(csv_file)
            self.csv_file = csv_file

            # Detect scraping mode
            self.scraping_mode = self._detect_scraping_mode()

            print(f"✅ Loaded data from: {csv_file}")
            print(f"   Total rows: {len(self.df)}")
            print(f"   Scraping mode: {self.scraping_mode}")
            print(f"   Columns: {len(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def _detect_scraping_mode(self):
        """Detect which scraping mode was used"""
        if self.df is None:
            return "unknown"

        if "source_type" in self.df.columns and len(self.df) > 0:
            return self.df["source_type"].iloc[0]

        # Fallback detection
        if "comment_text" in self.df.columns and "profile_username" in self.df.columns:
            return "profile"
        elif (
            "all_comments_text" in self.df.columns
            and "comments_scraped_count" in self.df.columns
        ):
            return "keyword"
        elif "comment_text" in self.df.columns and "comment_id" in self.df.columns:
            return "post_url"
        else:
            return "unknown"

    def find_latest_data(self):
        """Find the latest CSV file in the final directory"""
        final_dir = Path("Data/Instagram/final")
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
        """Create sentiment distribution charts based on scraping mode"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        colors = {"POSITIVE": "#2ecc71", "NEUTRAL": "#95a5a6", "NEGATIVE": "#e74c3c"}

        # Determine number of subplots based on available data
        has_caption = "caption_sentiment_label" in self.df.columns
        has_comment = "comment_sentiment_label" in self.df.columns
        has_comments = "comments_sentiment_label" in self.df.columns  # Keyword mode

        num_plots = sum([has_caption, has_comment or has_comments])

        if num_plots == 0:
            print("❌ No sentiment data available!")
            return

        fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Caption sentiment (all modes)
        if has_caption:
            caption_counts = self.df["caption_sentiment_label"].value_counts()
            caption_colors = [
                colors.get(label, "#3498db") for label in caption_counts.index
            ]

            axes[plot_idx].pie(
                caption_counts.values,
                labels=caption_counts.index,
                autopct="%1.1f%%",
                colors=caption_colors,
                startangle=90,
            )
            axes[plot_idx].set_title(
                "Post Caption Sentiment Distribution", fontsize=14, fontweight="bold"
            )
            plot_idx += 1

        # Comment sentiment (profile/post_url mode)
        if has_comment and self.scraping_mode in ["profile", "post_url"]:
            # Filter out rows without comments
            with_comments = self.df[
                self.df["comment_text"].notna() & (self.df["comment_text"] != "")
            ]

            if len(with_comments) > 0:
                comment_counts = with_comments["comment_sentiment_label"].value_counts()
                comment_colors = [
                    colors.get(label, "#3498db") for label in comment_counts.index
                ]

                axes[plot_idx].pie(
                    comment_counts.values,
                    labels=comment_counts.index,
                    autopct="%1.1f%%",
                    colors=comment_colors,
                    startangle=90,
                )
                axes[plot_idx].set_title(
                    f"Individual Comments Sentiment\n({len(with_comments)} comments)",
                    fontsize=14,
                    fontweight="bold",
                )
            else:
                axes[plot_idx].text(
                    0.5, 0.5, "No comments available", ha="center", va="center"
                )
                axes[plot_idx].set_title(
                    "Comment Sentiment", fontsize=14, fontweight="bold"
                )

        # Aggregated comments sentiment (keyword mode)
        elif has_comments and self.scraping_mode == "keyword":
            # Filter out rows without comments
            with_comments = self.df[
                self.df["all_comments_text"].notna()
                & (self.df["all_comments_text"] != "")
            ]

            if len(with_comments) > 0:
                comments_counts = with_comments[
                    "comments_sentiment_label"
                ].value_counts()
                comments_colors = [
                    colors.get(label, "#3498db") for label in comments_counts.index
                ]

                axes[plot_idx].pie(
                    comments_counts.values,
                    labels=comments_counts.index,
                    autopct="%1.1f%%",
                    colors=comments_colors,
                    startangle=90,
                )
                axes[plot_idx].set_title(
                    f"Aggregated Comments Sentiment\n({len(with_comments)} posts)",
                    fontsize=14,
                    fontweight="bold",
                )
            else:
                axes[plot_idx].text(
                    0.5, 0.5, "No comments available", ha="center", va="center"
                )
                axes[plot_idx].set_title(
                    "Comments Sentiment", fontsize=14, fontweight="bold"
                )

        plt.tight_layout()
        self._save_plot("sentiment_distribution.png")
        plt.show()

    def create_engagement_analysis(self):
        """Analyze engagement metrics by sentiment"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        if "caption_sentiment_label" not in self.df.columns:
            print("❌ No sentiment data available for engagement analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        sentiment_col = "caption_sentiment_label"

        # Get unique posts for engagement metrics
        if "post_url" in self.df.columns:
            posts_df = self.df.drop_duplicates(subset=["post_url"]).copy()
        else:
            posts_df = self.df.copy()

        # Likes by sentiment
        if "post_likes" in posts_df.columns:
            sentiment_likes = posts_df.groupby(sentiment_col)["post_likes"].mean()
            sentiment_order = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
            sentiment_likes = sentiment_likes.reindex(
                [s for s in sentiment_order if s in sentiment_likes.index]
            )

            axes[0, 0].bar(
                sentiment_likes.index,
                sentiment_likes.values,
                color=["#e74c3c", "#95a5a6", "#2ecc71"][: len(sentiment_likes)],
            )
            axes[0, 0].set_title(
                "Average Likes by Caption Sentiment", fontweight="bold"
            )
            axes[0, 0].set_ylabel("Average Likes")
            axes[0, 0].tick_params(axis="x", rotation=45)

        # Comments by sentiment
        if "post_comments_count" in posts_df.columns:
            sentiment_comments = posts_df.groupby(sentiment_col)[
                "post_comments_count"
            ].mean()
            sentiment_comments = sentiment_comments.reindex(
                [s for s in sentiment_order if s in sentiment_comments.index]
            )

            axes[0, 1].bar(
                sentiment_comments.index,
                sentiment_comments.values,
                color=["#e74c3c", "#95a5a6", "#2ecc71"][: len(sentiment_comments)],
            )
            axes[0, 1].set_title(
                "Average Comments by Caption Sentiment", fontweight="bold"
            )
            axes[0, 1].set_ylabel("Average Comments")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # Total engagement (likes + comments)
        if (
            "post_likes" in posts_df.columns
            and "post_comments_count" in posts_df.columns
        ):
            posts_df["total_engagement"] = posts_df["post_likes"].fillna(0) + posts_df[
                "post_comments_count"
            ].fillna(0)
            engagement_by_sentiment = posts_df.groupby(sentiment_col)[
                "total_engagement"
            ].mean()
            engagement_by_sentiment = engagement_by_sentiment.reindex(
                [s for s in sentiment_order if s in engagement_by_sentiment.index]
            )

            axes[1, 0].bar(
                engagement_by_sentiment.index,
                engagement_by_sentiment.values,
                color=["#e74c3c", "#95a5a6", "#2ecc71"][: len(engagement_by_sentiment)],
            )
            axes[1, 0].set_title(
                "Average Total Engagement by Sentiment", fontweight="bold"
            )
            axes[1, 0].set_ylabel("Average Total Engagement")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # Sentiment confidence distribution
        if "caption_sentiment_score" in self.df.columns:
            for sentiment in ["NEGATIVE", "NEUTRAL", "POSITIVE"]:
                if sentiment in self.df[sentiment_col].unique():
                    subset = self.df[self.df[sentiment_col] == sentiment]
                    color = {
                        "POSITIVE": "#2ecc71",
                        "NEUTRAL": "#95a5a6",
                        "NEGATIVE": "#e74c3c",
                    }[sentiment]
                    axes[1, 1].hist(
                        subset["caption_sentiment_score"],
                        alpha=0.6,
                        label=sentiment,
                        bins=15,
                        color=color,
                    )

            axes[1, 1].set_title("Sentiment Confidence Distribution", fontweight="bold")
            axes[1, 1].set_xlabel("Confidence Score")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        self._save_plot("engagement_analysis.png")
        plt.show()

    def create_top_posts_analysis(self, top_n=10):
        """Analyze top posts by engagement"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        if "post_likes" not in self.df.columns:
            print("❌ No likes data available")
            return

        # Get unique posts
        if "post_url" in self.df.columns:
            posts = self.df.drop_duplicates(subset=["post_url"]).copy()
        else:
            posts = self.df.copy()

        # Calculate engagement
        posts["engagement"] = posts["post_likes"].fillna(0)
        if "post_comments_count" in posts.columns:
            posts["engagement"] += posts["post_comments_count"].fillna(0)

        top_posts = posts.nlargest(min(top_n, len(posts)), "engagement")

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Top posts by engagement
        axes[0].barh(
            range(len(top_posts)), top_posts["engagement"].values, color="#3498db"
        )
        axes[0].set_yticks(range(len(top_posts)))
        axes[0].set_yticklabels([f"Post {i+1}" for i in range(len(top_posts))])
        axes[0].set_xlabel("Total Engagement (Likes + Comments)")
        axes[0].set_title(
            f"Top {len(top_posts)} Posts by Engagement", fontweight="bold"
        )
        axes[0].invert_yaxis()

        # Scatter plot: likes vs comments
        if "post_comments_count" in top_posts.columns:
            if "caption_sentiment_label" in top_posts.columns:
                sentiment_colors = top_posts["caption_sentiment_label"].map(
                    {"POSITIVE": "#2ecc71", "NEUTRAL": "#95a5a6", "NEGATIVE": "#e74c3c"}
                )
                sentiment_colors = sentiment_colors.fillna("#3498db")
            else:
                sentiment_colors = "#3498db"

            axes[1].scatter(
                top_posts["post_likes"],
                top_posts["post_comments_count"],
                s=200,
                c=sentiment_colors,
                alpha=0.6,
            )
            axes[1].set_xlabel("Likes")
            axes[1].set_ylabel("Comments")
            axes[1].set_title("Likes vs Comments for Top Posts", fontweight="bold")
            axes[1].grid(alpha=0.3)

            # Add legend
            if "caption_sentiment_label" in top_posts.columns:
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

    def create_user_engagement_analysis(self):
        """Analyze engagement by username/profile"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        # Find username column
        username_col = None
        for col in ["post_username", "profile_username", "username"]:
            if col in self.df.columns:
                username_col = col
                break

        if not username_col:
            print("❌ No username column found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Get unique posts for metrics
        if "post_url" in self.df.columns:
            posts_df = self.df.drop_duplicates(subset=["post_url"]).copy()
        else:
            posts_df = self.df.copy()

        # Top users by followers (if available)
        if "profile_followers" in self.df.columns:
            top_users = (
                posts_df.groupby(username_col)["profile_followers"].first().nlargest(10)
            )
            axes[0, 0].barh(range(len(top_users)), top_users.values, color="#e67e22")
            axes[0, 0].set_yticks(range(len(top_users)))
            axes[0, 0].set_yticklabels(top_users.index)
            axes[0, 0].set_xlabel("Followers")
            axes[0, 0].set_title("Top 10 Users by Followers", fontweight="bold")
            axes[0, 0].invert_yaxis()
        else:
            axes[0, 0].text(
                0.5, 0.5, "Follower data not available", ha="center", va="center"
            )
            axes[0, 0].set_title("Top Users by Followers", fontweight="bold")

        # Average likes per user
        if "post_likes" in posts_df.columns:
            avg_likes = posts_df.groupby(username_col)["post_likes"].mean().nlargest(10)
            axes[0, 1].barh(range(len(avg_likes)), avg_likes.values, color="#3498db")
            axes[0, 1].set_yticks(range(len(avg_likes)))
            axes[0, 1].set_yticklabels(avg_likes.index)
            axes[0, 1].set_xlabel("Average Likes")
            axes[0, 1].set_title("Top 10 Users by Avg Likes", fontweight="bold")
            axes[0, 1].invert_yaxis()

        # Sentiment distribution by top users
        if "caption_sentiment_label" in posts_df.columns:
            top_5_users = posts_df[username_col].value_counts().head(5).index
            sentiment_by_user = (
                posts_df[posts_df[username_col].isin(top_5_users)]
                .groupby([username_col, "caption_sentiment_label"])
                .size()
                .unstack(fill_value=0)
            )

            sentiment_by_user.plot(
                kind="bar",
                stacked=True,
                ax=axes[1, 0],
                color={
                    "POSITIVE": "#2ecc71",
                    "NEUTRAL": "#95a5a6",
                    "NEGATIVE": "#e74c3c",
                },
            )
            axes[1, 0].set_title(
                "Sentiment Distribution by Top 5 Users", fontweight="bold"
            )
            axes[1, 0].set_xlabel("Username")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].legend(title="Sentiment")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # Posts per user
        posts_per_user = posts_df[username_col].value_counts().head(10)
        axes[1, 1].barh(
            range(len(posts_per_user)), posts_per_user.values, color="#9b59b6"
        )
        axes[1, 1].set_yticks(range(len(posts_per_user)))
        axes[1, 1].set_yticklabels(posts_per_user.index)
        axes[1, 1].set_xlabel("Number of Posts")
        axes[1, 1].set_title("Top 10 Users by Post Count", fontweight="bold")
        axes[1, 1].invert_yaxis()

        plt.tight_layout()
        self._save_plot("user_engagement_analysis.png")
        plt.show()

    def create_comprehensive_report(self):
        """Generate all visualizations in one go"""
        print("\n" + "=" * 70)
        print("📊 GENERATING COMPREHENSIVE INSTAGRAM DASHBOARD")
        print("=" * 70 + "\n")

        if self.df is None:
            print("❌ No data loaded!")
            return

        print("Creating visualizations...")
        print("  1/4 Sentiment Distribution...")
        self.create_sentiment_distribution()

        print("  2/4 Engagement Analysis...")
        self.create_engagement_analysis()

        print("  3/4 Top Posts Analysis...")
        self.create_top_posts_analysis()

        print("  4/4 User Engagement Analysis...")
        self.create_user_engagement_analysis()

        print("\n✅ All visualizations generated!")
        print(f"📁 Saved to: Data/Instagram/visualizations/")
        print("=" * 70 + "\n")

    def _save_plot(self, filename):
        """Save plot to visualizations directory"""
        viz_dir = Path("Data/Instagram/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)

        filepath = viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {filename}")

    def print_summary_statistics(self):
        """Print detailed summary statistics based on scraping mode"""
        if self.df is None:
            print("❌ No data loaded!")
            return

        print("\n" + "=" * 70)
        print("📊 INSTAGRAM SENTIMENT ANALYSIS - SUMMARY STATISTICS")
        print("=" * 70 + "\n")

        print("DATASET OVERVIEW:")
        print(f"  Scraping Mode: {self.scraping_mode}")
        print(f"  Total Rows: {len(self.df)}")

        # Unique posts
        if "post_url" in self.df.columns:
            unique_posts = self.df["post_url"].nunique()
            print(f"  Unique Posts: {unique_posts}")

        # Unique users
        username_col = None
        for col in ["post_username", "profile_username", "username"]:
            if col in self.df.columns:
                username_col = col
                break
        if username_col:
            print(f"  Unique Users: {self.df[username_col].nunique()}")

        # Source information
        if "source_value" in self.df.columns and len(self.df) > 0:
            source = self.df["source_value"].iloc[0]
            print(f"  Source: {source}")

        print("\nPOST CAPTION SENTIMENTS:")
        if "caption_sentiment_label" in self.df.columns:
            # Get unique posts for caption analysis
            if "post_url" in self.df.columns:
                posts_df = self.df.drop_duplicates(subset=["post_url"])
            else:
                posts_df = self.df

            caption_sentiments = posts_df["caption_sentiment_label"].value_counts()
            for sentiment in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
                if sentiment in caption_sentiments.index:
                    count = caption_sentiments[sentiment]
                    percentage = (count / len(posts_df)) * 100
                    print(f"  {sentiment}: {count} ({percentage:.1f}%)")
        else:
            print("  No caption sentiment data available")

        print("\nCOMMENT SENTIMENTS:")
        if (
            self.scraping_mode in ["profile", "post_url"]
            and "comment_sentiment_label" in self.df.columns
        ):
            # Individual comments
            with_comments = self.df[
                self.df["comment_text"].notna() & (self.df["comment_text"] != "")
            ]
            if len(with_comments) > 0:
                comment_sentiments = with_comments[
                    "comment_sentiment_label"
                ].value_counts()
                print(f"  Total Comments Analyzed: {len(with_comments)}")
                for sentiment in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
                    if sentiment in comment_sentiments.index:
                        count = comment_sentiments[sentiment]
                        percentage = (count / len(with_comments)) * 100
                        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
            else:
                print("  No comments available")

        elif (
            self.scraping_mode == "keyword"
            and "comments_sentiment_label" in self.df.columns
        ):
            # Aggregated comments
            with_comments = self.df[
                self.df["all_comments_text"].notna()
                & (self.df["all_comments_text"] != "")
            ]
            if len(with_comments) > 0:
                comments_sentiments = with_comments[
                    "comments_sentiment_label"
                ].value_counts()
                print(f"  Posts with Comments: {len(with_comments)}")
                for sentiment in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
                    if sentiment in comments_sentiments.index:
                        count = comments_sentiments[sentiment]
                        percentage = (count / len(with_comments)) * 100
                        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
            else:
                print("  No comments available")
        else:
            print("  No comment sentiment data available")

        print("\nENGAGEMENT METRICS:")
        # Get unique posts for engagement metrics
        if "post_url" in self.df.columns:
            posts_df = self.df.drop_duplicates(subset=["post_url"])
        else:
            posts_df = self.df

        if "post_likes" in posts_df.columns:
            total_likes = posts_df["post_likes"].sum()
            avg_likes = posts_df["post_likes"].mean()
            print(f"  Total Likes: {total_likes:,.0f}")
            print(f"  Average Likes per Post: {avg_likes:.2f}")

        if "post_comments_count" in posts_df.columns:
            total_comments = posts_df["post_comments_count"].sum()
            avg_comments = posts_df["post_comments_count"].mean()
            print(f"  Total Comments: {total_comments:,.0f}")
            print(f"  Average Comments per Post: {avg_comments:.2f}")

        if (
            "caption_sentiment_label" in posts_df.columns
            and "post_likes" in posts_df.columns
        ):
            print("\nTOP PERFORMING SENTIMENT:")
            avg_engagement_by_sentiment = posts_df.groupby("caption_sentiment_label")[
                "post_likes"
            ].mean()
            top_sentiment = avg_engagement_by_sentiment.idxmax()
            print(
                f"  {top_sentiment} posts have highest avg likes: {avg_engagement_by_sentiment[top_sentiment]:.2f}"
            )

        print("\n" + "=" * 70 + "\n")


def run_instagram_dashboard(csv_file=None):
    """
    Main function to run the Instagram dashboard.
    If no file is provided, it will find the latest data file.
    """
    print("\n" + "=" * 70)
    print("INSTAGRAM SENTIMENT ANALYSIS DASHBOARD".center(70))
    print("=" * 70 + "\n")

    dashboard = InstagramDashboard(csv_file)

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
    print("  3. Top Posts Analysis")
    print("  4. User Engagement Analysis")
    print("  5. Generate All Reports")
    print("  6. Exit")

    while True:
        choice = input("\nSelect option (1-6): ").strip()

        if choice == "1":
            dashboard.create_sentiment_distribution()
        elif choice == "2":
            dashboard.create_engagement_analysis()
        elif choice == "3":
            dashboard.create_top_posts_analysis()
        elif choice == "4":
            dashboard.create_user_engagement_analysis()
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
        run_instagram_dashboard(sys.argv[1])
    else:
        run_instagram_dashboard()
