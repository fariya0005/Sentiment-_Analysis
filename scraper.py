"""
unified_social_media_scraper.py

A unified scraper that allows users to choose between:
1. Facebook Scraper
2. Instagram Scraper
3. Twitter Scraper

Each scraper runs its original logic after selection.
"""

import os
import sys
from pathlib import Path

# Ensure the main imports work
try:
    from apify_client import ApifyClient
    import pandas as pd
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install apify-client pandas python-dotenv")
    sys.exit(1)

# Load environment variables once
load_dotenv()


def print_banner():
    """Display welcome banner"""
    print("\n" + "=" * 70)
    print(" " * 15 + "🌐 UNIFIED SOCIAL MEDIA SCRAPER 🌐")
    print("=" * 70)
    print()


def get_main_choice():
    """Get user's platform choice"""
    print("Select the platform you want to scrape:\n")
    print("  1️⃣  Facebook")
    print("  2️⃣  Instagram")
    print("  3️⃣  Twitter (X)")
    print("  4️⃣  Exit")
    print("\n" + "-" * 70)

    choice = input("\nYour choice (1-4): ").strip()
    return choice


from apify_client import ApifyClient
import json
from datetime import datetime
import os
import pandas as pd
import re
from pathlib import Path
from dotenv import load_dotenv

from sentiment_facebook import SentimentAnalyzer
from dashboard_facebook import run_dashboard

# Load environment variables
load_dotenv()


class FacebookScraperPipeline:
    """Complete Facebook scraping pipeline with multi-source support."""

    def __init__(self, api_token):
        self.client = ApifyClient(api_token)

        # Actor IDs
        self.posts_actor_id = "powerai/facebook-post-search-scraper"
        self.comments_actor_id = "apify/facebook-comments-scraper"
        self.group_actor_id = "apify/facebook-groups-scraper"

        # Directories

        Facebook_output_dir = Path("Data/Facebook")
        self.preprocessing_dir = Facebook_output_dir / "preprocessing"
        self.final_dir = Facebook_output_dir / "final"

        self.preprocessing_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Pricing
        self.cost_per_1000_posts = 9.99
        self.cost_per_1000_comments = 2.50
        self.group_post_cost_per_1000 = 12.50
        self.actor_start_cost = 0.006

        print("Scraper initialized")

    def scrape_from_url(
        self, page_url, max_posts=10, max_comments_per_post=50, mode="page"
    ):
        print(f"\nScraping {page_url} in mode '{mode}'")
        print(f"Posts: {max_posts}, Comments/post: {max_comments_per_post}")

        # Scrape posts
        if mode == "group":
            posts = self._scrape_posts_with_reactions(page_url, max_posts, mode)
        elif mode == "keyword":
            posts = self._scrape_posts_with_reactions(page_url, max_posts, mode)
        else:
            search_term = self._extract_search_term(page_url)
            if not search_term:
                print(
                    "Could not extract a search term from the URL — using raw input as keyword."
                )
                search_term = page_url
            print(f"Extracted keyword: '{search_term}'")
            posts = self._scrape_posts_with_reactions(search_term, max_posts, mode)

        if not posts:
            print("No posts returned. Exiting scrape_from_url with empty result.")
            return {
                "raw_file": None,
                "final_file": None,
                "posts": 0,
                "comments": 0,
                "cost": 0.0,
            }

        # Scrape comments
        complete_data = []
        for post in posts:
            comments = self._scrape_comments(post.get("url", ""), max_comments_per_post)
            post["comments"] = comments
            complete_data.append(post)

        # Pricing estimator
        total_comments = sum(len(p.get("comments", [])) for p in complete_data)
        post_cost = (len(posts) / 1000) * (
            self.group_post_cost_per_1000
            if mode == "group"
            else self.cost_per_1000_posts
        )
        comment_cost = (total_comments / 1000) * self.cost_per_1000_comments
        actor_starts_cost = (len(posts) + 1) * self.actor_start_cost
        estimated_cost = post_cost + comment_cost + actor_starts_cost

        print(
            f"\nScraping complete: {len(complete_data)} posts, {total_comments} comments"
        )
        print(f"Estimated cost: ${estimated_cost:.4f}")

        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = self._save_raw_data(complete_data, timestamp)
        final_file = self._process_and_save_final(complete_data, timestamp)

        return {
            "raw_file": raw_file,
            "final_file": final_file,
            "posts": len(complete_data),
            "comments": total_comments,
            "cost": estimated_cost,
        }

    # ------------------------
    # Helper Methods
    # ------------------------

    def _extract_search_term(self, url):
        if not url:
            return None
        clean_url = (
            url.replace("https://", "").replace("http://", "").replace("www.", "")
        )
        patterns = [
            r"facebook\.com/pages/([^/]+)/(\d+)",
            r"facebook\.com/([^/\?]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, clean_url)
            if match:
                return match.group(1).replace("-", " ").replace("_", " ")
        return None

    def _is_group_url(self, url):
        if not url:
            return False
        clean_url = (
            url.replace("https://", "").replace("http://", "").replace("www.", "")
        )
        return "facebook.com/groups/" in clean_url

    def _normalize_group_url(self, url):
        if not url:
            return None
        if not url.startswith("http"):
            url = "https://" + url
        if "facebook.com/groups/" not in url:
            return None
        return url

    # ------------------------
    # Post & Comment Scraping
    # ------------------------

    def _scrape_posts_with_reactions(self, search_term, max_posts, mode="page"):
        # Ensure minimum posts
        if max_posts is None or max_posts < 3:
            print(f"max_posts ({max_posts}) is too low; setting to 3.")
            max_posts = 3

        if mode == "group":
            actor_id = self.group_actor_id
            group_url = self._normalize_group_url(search_term)
            if not group_url:
                print(f"Invalid group URL: {search_term}")
                return []
            run_input = {
                "startUrls": [{"url": group_url}],
                "resultsLimit": max_posts,
                "proxyConfiguration": {"useApifyProxy": True},
            }
            actor_runs = [run_input]
        else:
            actor_id = self.posts_actor_id
            actor_runs = [
                {"query": search_term, "maxResults": max_posts, "recent_posts": True},
                {"query": search_term, "maxResults": max_posts, "recent_posts": False},
            ]

        all_posts = []
        for run_input in actor_runs:
            print(f"Calling actor: {actor_id}")
            try:
                run = self.client.actor(actor_id).call(run_input=run_input)
            except Exception as e:
                print(f"Error calling posts actor '{actor_id}': {e}")
                import traceback

                traceback.print_exc()
                continue

            if not run or "defaultDatasetId" not in run:
                print("No defaultDatasetId returned from posts actor.")
                continue

            try:
                for item in self.client.dataset(
                    run["defaultDatasetId"]
                ).iterate_items():
                    if mode == "group":
                        all_posts.append(
                            {
                                "post_id": item.get(
                                    "postId",
                                    item.get(
                                        "post_id", item.get("url", "").split("/")[-1]
                                    ),
                                ),
                                "url": item.get("postUrl", item.get("url", "")),
                                "type": "post",
                                "message": item.get(
                                    "postText",
                                    item.get("text", item.get("message", "")),
                                ),
                                "timestamp": item.get(
                                    "postTime",
                                    item.get("time", item.get("timestamp", "")),
                                ),
                                "author_id": item.get("postAuthor", {}).get("id", ""),
                                "author_name": item.get("postAuthor", {}).get(
                                    "name", item.get("authorName", "")
                                ),
                                "author_url": item.get("postAuthor", {}).get(
                                    "url", item.get("authorUrl", "")
                                ),
                                "author_profile_picture": item.get(
                                    "postAuthor", {}
                                ).get("profilePicture", ""),
                                "total_reactions": item.get(
                                    "likes", item.get("reactions", 0)
                                ),
                                "total_comments": item.get(
                                    "comments", item.get("commentsCount", 0)
                                ),
                                "total_shares": item.get(
                                    "shares", item.get("sharesCount", 0)
                                ),
                                "emoji_like": 0,
                                "emoji_love": 0,
                                "emoji_haha": 0,
                                "emoji_wow": 0,
                                "emoji_sad": 0,
                                "emoji_angry": 0,
                                "emoji_care": 0,
                                "image_url": item.get(
                                    "image",
                                    (
                                        item.get("images", [""])[0]
                                        if item.get("images")
                                        else ""
                                    ),
                                ),
                                "video_url": item.get("video", ""),
                                "video_thumbnail": "",
                                "external_url": "",
                                "comments": [],
                                "scraped_at": datetime.now().isoformat(),
                            }
                        )
                    else:
                        reactions = item.get("reactions", {})
                        all_posts.append(
                            {
                                "post_id": item.get("post_id", ""),
                                "url": item.get("url", ""),
                                "type": item.get("type", "post"),
                                "message": item.get("message", ""),
                                "timestamp": item.get("timestamp", ""),
                                "author_id": item.get("author", {}).get("id", ""),
                                "author_name": item.get("author", {}).get("name", ""),
                                "author_url": item.get("author", {}).get("url", ""),
                                "author_profile_picture": item.get("author", {}).get(
                                    "profile_picture_url", ""
                                ),
                                "total_reactions": item.get("reactions_count", 0),
                                "total_comments": item.get("comments_count", 0),
                                "total_shares": item.get("reshare_count", 0),
                                "emoji_like": reactions.get("like", 0),
                                "emoji_love": reactions.get("love", 0),
                                "emoji_haha": reactions.get("haha", 0),
                                "emoji_wow": reactions.get("wow", 0),
                                "emoji_sad": reactions.get("sad", 0),
                                "emoji_angry": reactions.get("angry", 0),
                                "emoji_care": reactions.get("care", 0),
                                "image_url": item.get("image", ""),
                                "video_url": item.get("video", ""),
                                "video_thumbnail": item.get("video_thumbnail", ""),
                                "external_url": item.get("external_url", ""),
                                "comments": [],
                                "scraped_at": datetime.now().isoformat(),
                            }
                        )
            except Exception as e:
                print(f"Error iterating posts dataset: {e}")
                import traceback

                traceback.print_exc()
                continue

        unique_posts = {p["post_id"]: p for p in all_posts}.values()
        print(f"Successfully scraped {len(unique_posts)} posts")
        return list(unique_posts)

    def _scrape_comments(self, post_url, max_comments):
        if not post_url or str(post_url).strip() == "":
            return []

        comments_limit = (
            max_comments if isinstance(max_comments, int) and max_comments > 0 else 100
        )

        run_input = {
            "startUrls": [{"url": post_url}],
            "resultsLimit": comments_limit,
            "includeNestedComments": True,
        }

        try:
            run = self.client.actor(self.comments_actor_id).call(run_input=run_input)
        except Exception as e:
            print(f"Error calling comments actor: {e}")
            return []

        if not run or "defaultDatasetId" not in run:
            print("No defaultDatasetId returned from comments actor.")
            return []

        try:
            items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
        except Exception as e:
            print(f"Error iterating comments dataset: {e}")
            return []

        comments = []
        for item in items:
            comments.append(
                {
                    "comment_id": item.get("id", ""),
                    "post_id": item.get("facebookId", ""),
                    "text": item.get("text", ""),
                    "timestamp": item.get("date", ""),
                    "author_id": item.get("profileUrl", ""),
                    "author_name": item.get("profileName", ""),
                    "author_url": item.get("profileUrl", ""),
                    "author_profile_picture": item.get("profilePicture", ""),
                    "reactions_count": item.get("likesCount", 0),
                    "replies_count": item.get("commentsCount", 0),
                }
            )

        return comments

    # ------------------------
    # Data Saving / Cleaning
    # ------------------------

    def _save_raw_data(self, data, timestamp):
        json_file = self.preprocessing_dir / f"raw_data_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        csv_file = self.preprocessing_dir / f"raw_data_{timestamp}.csv"
        rows = []
        for post in data:
            if post.get("comments"):
                for comment in post["comments"]:
                    rows.append(
                        {**self._flatten_post(post), **self._flatten_comment(comment)}
                    )
            else:
                rows.append({**self._flatten_post(post), **self._empty_comment()})

        if rows:
            pd.DataFrame(rows).to_csv(csv_file, index=False, encoding="utf-8")

        return json_file

    def _process_and_save_final(self, data, timestamp):
        rows = []
        for post in data:
            if post.get("comments"):
                for comment in post["comments"]:
                    row = {**self._flatten_post(post), **self._flatten_comment(comment)}
                    row = self._clean_row(row)
                    rows.append(row)
            else:
                row = {**self._flatten_post(post), **self._empty_comment()}
                row = self._clean_row(row)
                rows.append(row)

        if not rows:
            return None

        try:
            df = pd.DataFrame(rows)
            df = self._clean_dataframe(df)
            df = df.drop_duplicates(subset=["post_id", "comment_id"], keep="first")
            df = self._add_derived_columns(df)

            # ========== SENTIMENT ANALYSIS TRIGGER ==========
            print("\n" + "=" * 60)
            print("🤖 RUNNING SENTIMENT ANALYSIS...")
            print("=" * 60)

            try:
                sentiment_analyzer = SentimentAnalyzer()
                df = sentiment_analyzer.analyze_posts_and_comments(df)
                print("✅ Sentiment analysis completed successfully!")
            except Exception as e:
                print(f"⚠️  Sentiment analysis failed: {e}")
                print("Continuing without sentiment data...")

            print("=" * 60 + "\n")
            # ================================================

            final_csv = self.final_dir / f"facebook_data_{timestamp}.csv"
            df.to_csv(final_csv, index=False, encoding="utf-8")

            self._save_summary_stats(df, timestamp)

            return final_csv
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            debug_file = self.final_dir / f"processing_error_rows_{timestamp}.json"
            try:
                with open(debug_file, "w", encoding="utf-8") as df_debug:
                    json.dump(rows, df_debug, ensure_ascii=False, indent=2)
                print(f"Saved raw rows to: {debug_file}")
            except Exception:
                pass
            return None

    # ------------------------
    # Flattening / Cleaning Helpers
    # ------------------------

    def _flatten_post(self, post):
        return {
            "post_id": post.get("post_id", ""),
            "post_url": post.get("url", ""),
            "post_type": post.get("type", ""),
            "post_message": post.get("message", ""),
            "post_timestamp": post.get("timestamp", ""),
            "post_author_id": post.get("author_id", ""),
            "post_author_name": post.get("author_name", ""),
            "post_author_url": post.get("author_url", ""),
            "emoji_like": post.get("emoji_like", 0),
            "emoji_love": post.get("emoji_love", 0),
            "emoji_haha": post.get("emoji_haha", 0),
            "emoji_wow": post.get("emoji_wow", 0),
            "emoji_sad": post.get("emoji_sad", 0),
            "emoji_angry": post.get("emoji_angry", 0),
            "emoji_care": post.get("emoji_care", 0),
            "post_total_reactions": post.get("total_reactions", 0),
            "post_total_comments": post.get("total_comments", 0),
            "post_total_shares": post.get("total_shares", 0),
            "post_has_image": 1 if post.get("image_url") else 0,
            "post_has_video": 1 if post.get("video_url") else 0,
            "post_has_link": 1 if post.get("external_url") else 0,
        }

    def _flatten_comment(self, comment):
        return {
            "comment_id": comment.get("comment_id", ""),
            "comment_text": comment.get("text", ""),
            "comment_timestamp": comment.get("timestamp", ""),
            "comment_author_id": comment.get("author_id", ""),
            "comment_author_name": comment.get("author_name", ""),
            "comment_author_url": comment.get("author_url", ""),
            "comment_reactions": comment.get("reactions_count", 0),
            "comment_replies": comment.get("replies_count", 0),
        }

    def _empty_comment(self):
        return {
            "comment_id": "",
            "comment_text": "",
            "comment_timestamp": "",
            "comment_author_id": "",
            "comment_author_name": "",
            "comment_author_url": "",
            "comment_reactions": 0,
            "comment_replies": 0,
        }

    def _clean_row(self, row):
        for field in ["post_message", "comment_text"]:
            if field in row and row[field]:
                row[field] = " ".join(str(row[field]).split()).replace("\x00", "")
        return row

    def _clean_dataframe(self, df):
        text_cols = [
            "post_message",
            "comment_text",
            "post_author_name",
            "comment_author_name",
        ]
        for col in text_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.replace("\x00", "", regex=False)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
        return df

    def _add_derived_columns(self, df):
        emoji_cols = [
            "emoji_like",
            "emoji_love",
            "emoji_haha",
            "emoji_wow",
            "emoji_sad",
            "emoji_angry",
            "emoji_care",
        ]
        for c in emoji_cols:
            if c not in df.columns:
                df[c] = 0
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        for col in ["post_total_reactions", "post_total_comments", "post_total_shares"]:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["post_message_length"] = df["post_message"].astype(str).str.len()
        df["comment_text_length"] = df["comment_text"].astype(str).str.len()
        df["post_engagement_total"] = (
            df["post_total_reactions"]
            + df["post_total_comments"]
            + df["post_total_shares"]
        )
        df["dominant_emotion"] = df[emoji_cols].idxmax(axis=1).str.replace("emoji_", "")
        df["positive_reactions"] = (
            df["emoji_like"]
            + df["emoji_love"]
            + df["emoji_haha"]
            + df["emoji_wow"]
            + df["emoji_care"]
        )
        df["negative_reactions"] = df["emoji_sad"] + df["emoji_angry"]
        df["sentiment_ratio"] = df.apply(
            lambda x: (
                x["positive_reactions"] / x["post_total_reactions"]
                if x["post_total_reactions"] > 0
                else 0
            ),
            axis=1,
        )

        return df

    def _save_summary_stats(self, df, timestamp):
        summary_file = self.final_dir / f"summary_stats_{timestamp}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("FACEBOOK SCRAPING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("STATISTICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total rows: {len(df)}\n")
            f.write(f"Unique posts: {df['post_id'].nunique()}\n")
            f.write(f"Total comments: {df['comment_id'].ne('').sum()}\n\n")

            f.write("EMOJI BREAKDOWN:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Like: {df['emoji_like'].sum():,}\n")
            f.write(f"Love: {df['emoji_love'].sum():,}\n")
            f.write(f"Haha: {df['emoji_haha'].sum():,}\n")
            f.write(f"Wow: {df['emoji_wow'].sum():,}\n")
            f.write(f"Sad: {df['emoji_sad'].sum():,}\n")
            f.write(f"Angry: {df['emoji_angry'].sum():,}\n")
            f.write(f"Care: {df['emoji_care'].sum():,}\n")
            f.write(f"Total: {df['post_total_reactions'].sum():,}\n\n")

            f.write("ENGAGEMENT METRICS:\n")
            f.write("-" * 50 + "\n")
            try:
                f.write(
                    f"Avg reactions/post: {df.groupby('post_id')['post_total_reactions'].first().mean():.2f}\n"
                )
                f.write(
                    f"Avg comments/post: {df.groupby('post_id')['comment_id'].count().mean():.2f}\n"
                )
                f.write(
                    f"Avg shares/post: {df.groupby('post_id')['post_total_shares'].first().mean():.2f}\n"
                )
            except Exception as e:
                f.write(f"Could not compute some engagement metrics: {e}\n")

            # Add sentiment analysis summary if available
            if "post_sentiment_label" in df.columns:
                f.write("\nSENTIMENT ANALYSIS:\n")
                f.write("-" * 50 + "\n")
                f.write("Post Sentiments:\n")
                post_sentiment_counts = df["post_sentiment_label"].value_counts()
                for label, count in post_sentiment_counts.items():
                    f.write(f"  {label}: {count} ({count/len(df)*100:.1f}%)\n")

                if "comment_sentiment_label" in df.columns:
                    f.write("\nComment Sentiments:\n")
                    comment_sentiment_counts = df[
                        "comment_sentiment_label"
                    ].value_counts()
                    for label, count in comment_sentiment_counts.items():
                        f.write(f"  {label}: {count} ({count/len(df)*100:.1f}%)\n")


# ------------------------
# RENAMED MAIN TO run_facebook_scraper
# ------------------------
def run_facebook_scraper():
    """Run Facebook scraper - renamed from main() for unified integration"""
    print("\n" + "=" * 70)
    print("FACEBOOK SCRAPER".center(70))
    print("=" * 70 + "\n")

    API_TOKEN = os.getenv("APIFY_API_TOKEN")
    if not API_TOKEN:
        API_TOKEN = input("Enter Apify API token: ").strip()

    if not API_TOKEN:
        print("❌ No token provided. Returning to main menu.")
        return

    scraper = FacebookScraperPipeline(API_TOKEN)

    print("\n📋 Scraping Mode Options:")
    print("  1. Page (search by page URL or name)")
    print("  2. Group (Facebook group URL)")
    print("  3. Keyword (search by keyword/hashtag)")

    mode_input = input("\nChoose mode (1/2/3 or page/group/keyword): ").strip().lower()
    mode_map = {"1": "page", "2": "group", "3": "keyword"}
    mode = mode_map.get(mode_input, mode_input)

    if mode not in ["page", "group", "keyword"]:
        print(f"⚠️  Invalid mode '{mode_input}', defaulting to 'page'")
        mode = "page"

    print(f"✓ Selected mode: {mode}")

    target = input(f"\nEnter {mode} URL or keyword: ").strip()

    try:
        max_posts_input = input("Number of posts to scrape (default 10): ").strip()
        max_posts = int(max_posts_input) if max_posts_input else 10
    except ValueError:
        max_posts = 10

    if max_posts < 3:
        max_posts = 3

    try:
        max_comments_input = input("Comments per post (default 50): ").strip()
        max_comments = int(max_comments_input) if max_comments_input else 50
    except ValueError:
        max_comments = 50

    if max_comments < 0:
        max_comments = 0

    print("\n🚀 Starting scrape...\n")

    try:
        result = scraper.scrape_from_url(
            page_url=target,
            max_posts=max_posts,
            max_comments_per_post=max_comments,
            mode=mode,
        )

        print("\n" + "=" * 70)
        print("✅ SCRAPING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(
            f"\n📁 Raw data saved: {result['raw_file'].name if result['raw_file'] else 'None'}"
        )
        print(
            f"📊 Final data saved: {result['final_file'].name if result['final_file'] else 'None'}"
        )
        print(f"📝 Posts scraped: {result['posts']}")
        print(f"💬 Comments scraped: {result['comments']}")
        print(f"💰 Estimated cost: ${result['cost']:.4f}")
        print("=" * 70 + "\n")

        # ========== DASHBOARD TRIGGER ==========
        if result["final_file"]:
            print("\n" + "=" * 70)
            show_dashboard = (
                input("📊 Would you like to view the dashboard? (y/n): ")
                .strip()
                .lower()
            )
            if show_dashboard in ["y", "yes"]:
                print("\n🚀 Launching dashboard...\n")
                run_dashboard(result["final_file"])
            else:
                print("💡 You can run the dashboard anytime using: python dashboard.py")
            print("=" * 70 + "\n")
        # ========================================

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR OCCURRED")
        print("=" * 70)
        print(f"\n{str(e)}\n")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 70 + "\n")


# ============================================================================
# INSTAGRAM SCRAPER (from instagram_unified_scraper.py)
# ============================================================================

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from apify_client import ApifyClient

from sentiment_insta import InstagramSentimentAnalyzer
from dashboard_insta import run_instagram_dashboard


class InstagramScraperPipeline:
    """Complete Instagram scraping pipeline with multi-source support."""

    def __init__(self, api_token):
        self.client = ApifyClient(api_token)

        # Actor IDs
        self.profile_actor_id = "apify/instagram-profile-scraper"
        self.post_actor_id = "apify/instagram-post-scraper"
        self.hashtag_actor_id = "apify/instagram-hashtag-scraper"
        self.scraper_actor_id = "apify/instagram-scraper"
        self.comments_actor_id = "louisdeconinck/instagram-comments-scraper"

        # Directory structure
        Instagram_output_dir = Path("Data/Instagram")
        self.preprocessing_dir = Instagram_output_dir / "preprocessing"
        self.final_dir = Instagram_output_dir / "final"

        self.preprocessing_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for file naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(
                    "instagram_scraper.log", mode="a", encoding="utf-8"
                ),
                logging.StreamHandler(),
            ],
        )

        print("Instagram scraper initialized")

    # -------------------------
    # Helper Methods
    # -------------------------

    def _run_apify_actor(self, actor_id: str, run_input: dict):
        """Run an Apify actor and return dataset items."""
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                logging.info(f"Running actor '{actor_id}', attempt {attempt}")
                run = self.client.actor(actor_id).call(run_input=run_input)
                dataset_id = run.get("defaultDatasetId")
                if not dataset_id:
                    logging.warning(f"No data returned from actor '{actor_id}'")
                    return []
                return list(self.client.dataset(dataset_id).iterate_items())
            except Exception as e:
                logging.error(f"Error on attempt {attempt} for actor '{actor_id}': {e}")
                if attempt == attempts:
                    return []
        return []

    def _save_preprocessed_data(self, items: list, name_prefix: str):
        """Save raw JSON of scraped data in preprocessing folder."""
        if not items:
            return None
        json_path = self.preprocessing_dir / f"{name_prefix}_{self.timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        logging.info(f"Raw JSON saved to: {json_path}")
        return json_path

    def _parse_cookies_file(self, cookies_path: str):
        """Parse Netscape cookies.txt format."""
        cookies_list = []
        try:
            with open(cookies_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 7:
                        cookies_list.append(
                            {
                                "name": parts[5],
                                "value": parts[6],
                                "domain": (
                                    parts[0]
                                    if parts[0].startswith(".")
                                    else f".{parts[0]}"
                                ),
                            }
                        )
        except Exception as e:
            logging.error(f"Error parsing cookies file: {e}")
        return cookies_list

    def _apply_sentiment_analysis(self, df):
        """Apply sentiment analysis to dataframe"""
        if df.empty:
            return df

        try:
            print("\n" + "=" * 70)
            print("🤖 APPLYING SENTIMENT ANALYSIS...")
            print("=" * 70)

            analyzer = InstagramSentimentAnalyzer()
            df = analyzer.analyze_instagram_data(df)

            print("✅ Sentiment analysis completed!")
            print("=" * 70 + "\n")
            return df
        except Exception as e:
            print(f"⚠️  Sentiment analysis failed: {e}")
            print("Continuing without sentiment data...")
            return df

    # -------------------------
    # Main Scraping Methods
    # -------------------------

    def scrape_profile(
        self,
        username: str,
        max_posts=12,
        include_comments=True,
        max_comments=50,
        cookies_path="cookies.txt",
    ):
        """Scrape profile with all data integrated into unified records."""
        print(f"\n{'='*70}")
        print(f"🔍 COMPREHENSIVE PROFILE SCRAPE: @{username}")
        print(f"{'='*70}")

        # Step 1: Get profile info
        print(f"\n📊 Step 1/3: Fetching profile information...")
        profile_run_input = {"usernames": [username], "resultsLimit": 1}
        profile_items = self._run_apify_actor(self.profile_actor_id, profile_run_input)
        self._save_preprocessed_data(profile_items, f"profile_{username}")

        profile_data = {}
        if profile_items:
            for item in profile_items:
                if (
                    item.get("followersCount") is not None
                    or item.get("followers") is not None
                ):
                    profile_data = {
                        "profile_username": item.get("username"),
                        "profile_full_name": item.get("fullName")
                        or item.get("full_name"),
                        "profile_bio": item.get("biography") or item.get("bio"),
                        "profile_followers": item.get("followersCount")
                        or item.get("followers"),
                        "profile_following": item.get("followsCount")
                        or item.get("following"),
                        "profile_total_posts": item.get("postsCount")
                        or item.get("posts"),
                    }
                    print(
                        f"✅ Profile: {profile_data['profile_followers']} followers, {profile_data['profile_total_posts']} posts"
                    )
                    break

        if not profile_data:
            print("❌ Could not retrieve profile information")
            return pd.DataFrame()

        # Step 2: Get posts
        print(f"\n📸 Step 2/3: Fetching {max_posts} recent posts...")
        posts_run_input = {"username": [username], "resultsLimit": max_posts}
        post_items = self._run_apify_actor(self.post_actor_id, posts_run_input)
        self._save_preprocessed_data(post_items, f"posts_{username}")

        if not post_items:
            print("⚠️  No posts found")
            return pd.DataFrame()

        # Build post URLs for comment scraping
        post_urls = []
        posts_dict = {}
        post_media_ids = {}

        for item in post_items:
            post_url = item.get("url") or item.get("postUrl")
            shortcode = item.get("shortCode") or item.get("id")
            media_id = item.get("pk") or item.get("media_id") or item.get("id")

            if not post_url and shortcode:
                post_url = f"https://www.instagram.com/p/{shortcode}/"
            elif post_url and "instagram.com" not in post_url:
                post_url = f"https://www.instagram.com/p/{post_url}/"

            if post_url:
                post_urls.append(post_url)
                posts_dict[post_url] = {
                    "post_url": post_url,
                    "post_caption": item.get("caption") or item.get("text"),
                    "post_likes": item.get("likesCount")
                    or item.get("like_count")
                    or item.get("likes"),
                    "post_comments_count": item.get("commentsCount")
                    or item.get("comment_count")
                    or item.get("comments"),
                    "post_date": item.get("timestamp")
                    or item.get("post_date")
                    or item.get("created_at"),
                    "post_type": item.get("type") or "post",
                }
                if media_id:
                    post_media_ids[str(media_id)] = post_url

        print(f"✅ Found {len(post_urls)} posts")

        # Step 3: Get comments
        comments_dict = {}
        if include_comments and post_urls:
            print(f"\n💬 Step 3/3: Fetching up to {max_comments} comments per post...")

            if not Path(cookies_path).exists():
                print(
                    f"⚠️  Cookies file not found at {cookies_path}. Skipping comments."
                )
            else:
                cookies_list = self._parse_cookies_file(cookies_path)
                if not cookies_list:
                    print("⚠️  No valid cookies found. Skipping comments.")
                else:
                    cookies_json_string = json.dumps(cookies_list)
                    comment_run_input = {
                        "urls": post_urls,
                        "maxComments": max_comments,
                        "cookies": cookies_json_string,
                    }

                    comment_items = self._run_apify_actor(
                        self.comments_actor_id, comment_run_input
                    )
                    self._save_preprocessed_data(comment_items, f"comments_{username}")

                    if comment_items:
                        for comment in comment_items:
                            post_url = comment.get("postUrl")
                            media_id = str(comment.get("media_id", ""))

                            if post_url:
                                if "instagram.com/p/" in post_url:
                                    parts = post_url.split("/p/")
                                    if len(parts) > 1:
                                        shortcode = (
                                            parts[1]
                                            .strip("/")
                                            .split("/")[0]
                                            .split("?")[0]
                                        )
                                        post_url = (
                                            f"https://www.instagram.com/p/{shortcode}/"
                                        )

                            if not post_url and media_id and media_id in post_media_ids:
                                post_url = post_media_ids[media_id]

                            if not post_url:
                                continue

                            if post_url not in comments_dict:
                                comments_dict[post_url] = []

                            user = comment.get("user", {})
                            comments_dict[post_url].append(
                                {
                                    "comment_text": comment.get("text"),
                                    "comment_username": user.get("username"),
                                    "comment_full_name": user.get("full_name"),
                                    "comment_likes": comment.get("comment_like_count")
                                    or comment.get("comment_like_count"),
                                    "comment_date": comment.get("created_at")
                                    or comment.get("created_at_utc"),
                                }
                            )

                        print(f"✅ Scraped {len(comment_items)} total comments")
                        print(f"   Comments mapped to {len(comments_dict)} posts")

        # Combine all data into unified records
        print(f"\n🔄 Combining all data...")
        unified_records = []

        for post_url, post_data in posts_dict.items():
            base_record = {
                **profile_data,
                **post_data,
                "source_type": "profile",
                "source_value": username,
            }

            post_comments = comments_dict.get(post_url, [])

            if not post_comments and "instagram.com/p/" in post_url:
                shortcode = (
                    post_url.split("/p/")[1].strip("/").split("/")[0].split("?")[0]
                )
                for comment_url in comments_dict.keys():
                    if shortcode in comment_url:
                        post_comments = comments_dict[comment_url]
                        break

            if post_comments:
                for comment in post_comments:
                    record = {**base_record, **comment}
                    unified_records.append(record)
            else:
                record = {
                    **base_record,
                    "comment_text": None,
                    "comment_username": None,
                    "comment_full_name": None,
                    "comment_likes": None,
                    "comment_date": None,
                }
                unified_records.append(record)

        df = pd.DataFrame(unified_records)

        print(f"\n{'='*70}")
        print(f"✅ SCRAPING COMPLETE!")
        print(f"   • Profile: {profile_data['profile_username']}")
        print(f"   • Posts: {len(posts_dict)}")
        print(f"   • Comments: {sum(len(c) for c in comments_dict.values())}")
        print(f"   • Total rows: {len(unified_records)}")
        print(f"{'='*70}")

        return df

    def scrape_keyword(
        self,
        keyword: str,
        max_posts=50,
        include_comments=True,
        max_comments=50,
        cookies_path="cookies.txt",
    ):
        """Scrape keyword/hashtag with all data integrated."""
        print(f"\n{'='*70}")
        print(f"🔍 COMPREHENSIVE KEYWORD SCRAPE: #{keyword}")
        print(f"{'='*70}")

        keyword = keyword.lstrip("#").strip()

        print(f"\n📸 Step 1/2: Fetching {max_posts} posts for #{keyword}...")
        run_input = {
            "hashtags": [keyword],
            "resultsLimit": max_posts,
            "addParentData": False,
        }
        hashtag_items = self._run_apify_actor(self.hashtag_actor_id, run_input)
        self._save_preprocessed_data(hashtag_items, f"keyword_{keyword}")

        if not hashtag_items:
            print("❌ No posts found for keyword")
            return pd.DataFrame()

        post_urls = []
        posts_dict = {}

        for item in hashtag_items:
            post_url = item.get("url") or item.get("postUrl")
            shortcode = item.get("shortCode") or item.get("id")

            if not post_url and shortcode:
                post_url = f"https://www.instagram.com/p/{shortcode}/"
            elif post_url and "instagram.com" not in post_url:
                post_url = f"https://www.instagram.com/p/{post_url}/"

            if post_url:
                post_urls.append(post_url)
                posts_dict[post_url] = {
                    "post_url": post_url,
                    "post_username": item.get("ownerUsername") or item.get("username"),
                    "post_caption": item.get("caption") or item.get("text"),
                    "post_likes": item.get("likesCount")
                    or item.get("like_count")
                    or item.get("likes"),
                    "post_comments_count": item.get("commentsCount")
                    or item.get("comment_count")
                    or item.get("comments"),
                    "post_date": item.get("timestamp")
                    or item.get("post_date")
                    or item.get("created_at"),
                }

        print(f"✅ Found {len(post_urls)} posts")

        comments_dict = {}
        if include_comments and post_urls:
            print(f"\n💬 Step 2/2: Fetching up to {max_comments} comments per post...")

            if not Path(cookies_path).exists():
                print(f"⚠️  Cookies file not found. Skipping comments.")
            else:
                cookies_list = self._parse_cookies_file(cookies_path)
                if cookies_list:
                    cookies_json_string = json.dumps(cookies_list)
                    comment_run_input = {
                        "urls": post_urls,
                        "maxComments": max_comments,
                        "cookies": cookies_json_string,
                    }

                    comment_items = self._run_apify_actor(
                        self.comments_actor_id, comment_run_input
                    )
                    self._save_preprocessed_data(
                        comment_items, f"comments_keyword_{keyword}"
                    )

                    if comment_items:
                        for comment in comment_items:
                            post_url = comment.get("postUrl")
                            if post_url not in comments_dict:
                                comments_dict[post_url] = []

                            user = comment.get("user", {})
                            comments_dict[post_url].append(
                                {
                                    "comment_text": comment.get("text"),
                                    "comment_username": user.get("username"),
                                    "comment_full_name": user.get("full_name"),
                                    "comment_likes": comment.get("comment_like_count"),
                                    "comment_date": comment.get("created_at"),
                                }
                            )
                        print(f"✅ Scraped {len(comment_items)} total comments")

        print(f"\n🔄 Combining all data...")
        unified_records = []

        for post_url, post_data in posts_dict.items():
            base_record = {
                **post_data,
                "source_type": "keyword",
                "source_value": keyword,
            }

            post_comments = comments_dict.get(post_url, [])
            comments_scraped_count = len(post_comments)

            if comments_scraped_count > 0:
                all_comments_text = " || ".join(
                    [
                        f"{c.get('comment_username')}: {c.get('comment_text')}"
                        for c in post_comments
                        if c.get("comment_text")
                    ]
                )
            else:
                all_comments_text = None

            record = {
                **base_record,
                "comments_scraped_count": comments_scraped_count,
                "all_comments_text": all_comments_text,
            }

            unified_records.append(record)

        df = pd.DataFrame(unified_records)

        print(f"\n{'='*70}")
        print(f"✅ SCRAPING COMPLETE!")
        print(f"   • Keyword: #{keyword}")
        print(f"   • Posts: {len(posts_dict)}")
        print(f"   • Comments scraped: {sum(len(c) for c in comments_dict.values())}")
        print(f"   • Total rows: {len(unified_records)}")
        print(f"{'='*70}")

        return df

    def scrape_post_urls(
        self,
        post_urls: list,
        include_comments=True,
        max_comments=100,
        cookies_path="cookies.txt",
    ):
        """Scrape specific URLs with all data integrated."""
        print(f"\n{'='*70}")
        print(f"🔍 COMPREHENSIVE POST URL SCRAPE: {len(post_urls)} URL(s)")
        print(f"{'='*70}")

        print(f"\n📸 Step 1/2: Fetching post details...")
        run_input = {
            "directUrls": post_urls,
            "resultsType": "posts",
            "resultsLimit": len(post_urls),
            "searchType": "hashtag",
            "searchLimit": 1,
        }

        post_items = self._run_apify_actor(self.scraper_actor_id, run_input)
        self._save_preprocessed_data(post_items, "post_urls")

        if not post_items:
            print("❌ No post data retrieved")
            return pd.DataFrame()

        print(f"✅ Retrieved {len(post_items)} post(s)")
        print(f"\n🔄 Processing posts and comments...")
        unified_records = []

        for item in post_items:
            post_url = item.get("url") or item.get("inputUrl")
            shortcode = item.get("shortCode") or item.get("id")

            if not post_url and shortcode:
                post_url = f"https://www.instagram.com/p/{shortcode}/"
            elif post_url and "instagram.com" not in post_url:
                post_url = f"https://www.instagram.com/p/{post_url}/"

            base_record = {
                "source_type": "post_url",
                "source_value": post_url,
                "post_url": post_url,
                "post_username": item.get("ownerUsername"),
                "post_full_name": item.get("ownerFullName"),
                "post_caption": item.get("caption"),
                "post_likes": item.get("likesCount"),
                "post_comments_count": item.get("commentsCount"),
                "post_date": item.get("timestamp"),
                "post_type": item.get("type"),
                "post_video_views": item.get("videoViewCount"),
                "post_video_plays": item.get("videoPlayCount"),
            }

            comments = []

            if include_comments and "latestComments" in item:
                latest_comments = item.get("latestComments", [])
                comments.extend(latest_comments[:max_comments])

            if comments:
                for comment in comments:
                    owner = comment.get("owner", {})
                    user = comment.get("user", {})

                    comment_record = {
                        **base_record,
                        "comment_id": comment.get("id") or comment.get("pk"),
                        "comment_text": comment.get("text"),
                        "comment_username": owner.get("username")
                        or user.get("username")
                        or comment.get("ownerUsername"),
                        "comment_full_name": owner.get("full_name")
                        or user.get("full_name"),
                        "comment_likes": comment.get("likesCount")
                        or comment.get("comment_like_count"),
                        "comment_date": comment.get("timestamp")
                        or comment.get("created_at"),
                        "comment_replies_count": comment.get("repliesCount")
                        or comment.get("child_comment_count"),
                    }
                    unified_records.append(comment_record)
            else:
                unified_records.append(
                    {
                        **base_record,
                        "comment_id": None,
                        "comment_text": None,
                        "comment_username": None,
                        "comment_full_name": None,
                        "comment_likes": None,
                        "comment_date": None,
                        "comment_replies_count": None,
                    }
                )

        df = pd.DataFrame(unified_records)

        if "comment_id" in df.columns:
            df = df.drop_duplicates(subset=["post_url", "comment_id"], keep="first")

        unique_posts = df["post_url"].nunique()
        total_comments = df[df["comment_id"].notna()].shape[0]

        print(f"\n{'='*70}")
        print(f"✅ SCRAPING COMPLETE!")
        print(f"   • Posts: {unique_posts}")
        print(f"   • Comments: {total_comments}")
        print(f"   • Total rows: {len(df)}")
        print(f"{'='*70}")

        return df

    def save_final_data(self, df, name_prefix):
        """Save final processed data with sentiment analysis"""
        if df.empty:
            return None

        # Apply sentiment analysis
        df = self._apply_sentiment_analysis(df)

        # Save to CSV
        output_file = self.final_dir / f"{name_prefix}_{self.timestamp}.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n💾 Data saved to: {output_file}")

        return output_file


# -------------------------
# Main Function (Renamed)
# -------------------------
def run_instagram_scraper():
    """Run Instagram scraper - refactored to match Facebook scraper structure"""
    load_dotenv()

    logging.info("=== Comprehensive Instagram Scraper Started ===")

    print("\n" + "=" * 70)
    print("INSTAGRAM SCRAPER".center(70))
    print("=" * 70 + "\n")

    API_TOKEN = os.getenv("APIFY_API_TOKEN")
    if not API_TOKEN:
        API_TOKEN = input("Enter Apify API token: ").strip()
        if not API_TOKEN:
            logging.error("No API token provided. Exiting.")
            return

    scraper = InstagramScraperPipeline(API_TOKEN)

    while True:
        print("\n" + "=" * 70)
        print("           COMPREHENSIVE INSTAGRAM SCRAPER")
        print("=" * 70)
        print("Choose scraping mode:")
        print()
        print("1️⃣  Scrape by PROFILE")
        print("2️⃣  Scrape by KEYWORD/HASHTAG ")
        print("3️⃣  Scrape by POST URL(s)")
        print("4️⃣  Exit")
        print("=" * 70)

        choice = input("Your choice (1-4): ").strip()

        if choice == "1":
            print("\n" + "-" * 70)
            print("SCRAPE BY PROFILE")
            print("-" * 70)
            username = input("Enter Instagram username (without @):\n> ").strip()
            if not username:
                print("❌ No username provided. Skipping.")
                continue

            max_posts_input = input(
                "How many recent posts to scrape? (default: 12):\n> "
            ).strip()
            max_posts = int(max_posts_input) if max_posts_input else 12

            include_comments_input = (
                input("Include comments? (y/n, default: y):\n> ").strip().lower()
            )
            include_comments = include_comments_input != "n"

            max_comments = 50
            if include_comments:
                max_comments_input = input(
                    "Max comments per post? (default: 50):\n> "
                ).strip()
                max_comments = int(max_comments_input) if max_comments_input else 50

            df_result = scraper.scrape_profile(
                username, max_posts, include_comments, max_comments
            )

            if not df_result.empty:
                output_file = scraper.save_final_data(df_result, f"profile_{username}")

                # Ask to view dashboard
                if output_file:
                    show_dashboard = (
                        input("\n📊 View dashboard? (y/n): ").strip().lower()
                    )
                    if show_dashboard in ["y", "yes"]:
                        run_instagram_dashboard(output_file)

        elif choice == "2":
            print("\n" + "-" * 70)
            print("SCRAPE BY KEYWORD/HASHTAG")
            print("-" * 70)
            keyword = input("Enter keyword or hashtag (# optional):\n> ").strip()
            if not keyword:
                print("❌ No keyword provided. Skipping.")
                continue

            max_posts_input = input(
                "How many posts to scrape? (default: 50):\n> "
            ).strip()
            max_posts = int(max_posts_input) if max_posts_input else 50

            include_comments_input = (
                input("Include comments? (y/n, default: y):\n> ").strip().lower()
            )
            include_comments = include_comments_input != "n"

            max_comments = 50
            if include_comments:
                max_comments_input = input(
                    "Max comments per post? (default: 50):\n> "
                ).strip()
                max_comments = int(max_comments_input) if max_comments_input else 50

            df_result = scraper.scrape_keyword(
                keyword, max_posts, include_comments, max_comments
            )

            if not df_result.empty:
                output_file = scraper.save_final_data(df_result, f"keyword_{keyword}")

                # Ask to view dashboard
                if output_file:
                    show_dashboard = (
                        input("\n📊 View dashboard? (y/n): ").strip().lower()
                    )
                    if show_dashboard in ["y", "yes"]:
                        run_instagram_dashboard(output_file)

        elif choice == "3":
            print("\n" + "-" * 70)
            print("SCRAPE BY POST URL(s)")
            print("-" * 70)
            post_urls_input = input(
                "Enter Instagram post URLs (comma-separated):\n> "
            ).strip()
            if not post_urls_input:
                print("❌ No URLs provided. Skipping.")
                continue

            post_urls = [u.strip() for u in post_urls_input.split(",") if u.strip()]

            include_comments_input = (
                input("Include comments? (y/n, default: y):\n> ").strip().lower()
            )
            include_comments = include_comments_input != "n"

            max_comments = 100
            if include_comments:
                max_comments_input = input(
                    "Max comments per post? (default: 100):\n> "
                ).strip()
                max_comments = int(max_comments_input) if max_comments_input else 100

            df_result = scraper.scrape_post_urls(
                post_urls, include_comments, max_comments
            )

            if not df_result.empty:
                output_file = scraper.save_final_data(df_result, "post_urls")

                # Ask to view dashboard
                if output_file:
                    show_dashboard = (
                        input("\n📊 View dashboard? (y/n): ").strip().lower()
                    )
                    if show_dashboard in ["y", "yes"]:
                        run_instagram_dashboard(output_file)

        elif choice == "4":
            print("\n" + "=" * 70)
            print("👋 Exiting Instagram scraper...")
            logging.info("=== Exiting Instagram Scraper ===")
            print("=" * 70)
            break

        else:
            print("❌ Invalid choice. Please select a number from 1 to 4.")

    print("\n=== Instagram Scraper Session Ended ===")


# ============================================================================
# TWITTER SCRAPER (from twitter_scraper_combined.py)
# ============================================================================

from apify_client import ApifyClient
import json
from datetime import datetime
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from sentiment_twitter import TwitterSentimentAnalyzer
from dashboard_twitter import run_twitter_dashboard

# Load environment variables
load_dotenv()


class TwitterScraperPipeline:
    """Complete Twitter scraping pipeline with profile, replies, and retweets support."""

    def __init__(self, api_token):
        self.client = ApifyClient(api_token)

        # Actor IDs
        self.profile_actor_id = "web.harvester/twitter-scraper"
        self.replies_actor_id = "kaitoeasyapi/twitter-reply"
        self.retweeters_actor_id = "kaitoeasyapi/tweet-reweet-userlist"

        # Directories
        Twitter_output_dir = Path("Data/Twitter")
        self.preprocessing_dir = Twitter_output_dir / "preprocessing"
        self.final_dir = Twitter_output_dir / "final"

        self.preprocessing_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        print("Twitter Scraper initialized")

    def scrape_from_user(
        self, user_input, tweet_ids, max_replies=200, max_retweets=200
    ):
        """
        Main scraping method for Twitter data.

        Args:
            user_input: Twitter username or profile URL
            tweet_ids: List of tweet IDs to scrape
            max_replies: Maximum replies per tweet
            max_retweets: Maximum retweeters per tweet
        """
        print(f"\nScraping Twitter data for: {user_input}")
        print(f"Tweet IDs: {tweet_ids}")
        print(f"Max replies/tweet: {max_replies}, Max retweets/tweet: {max_retweets}")

        # Normalize username and profile URL
        if not user_input.startswith("http"):
            username = user_input.replace("@", "").strip()
            profile_url = f"https://x.com/{username}"
        else:
            profile_url = user_input.replace("twitter.com", "x.com")
            username = profile_url.rstrip("/").split("/")[-1]

        # Validate tweet IDs
        if not tweet_ids:
            print("No tweet IDs provided. Exiting scrape_from_user with empty result.")
            return {
                "raw_file": None,
                "final_file": None,
                "tweets": 0,
                "replies": 0,
                "retweeters": 0,
            }

        # Combined data storage
        all_data = []

        # Process each tweet ID
        for tid in tweet_ids:
            print(f"\n{'='*70}")
            print(f"🔍 Processing Tweet ID: {tid} (username: @{username})")
            print(f"{'='*70}")

            # Step 1: Fetch profile info + main tweet
            print("📊 Step 1/3: Fetching profile info...")
            profile_info = self._fetch_profile_info(profile_url, username, tid)

            # Step 2: Scrape replies
            print(f"\n💬 Step 2/3: Scraping up to {max_replies} replies...")
            replies_list = self._scrape_replies(tid, max_replies)

            # Step 3: Scrape retweeters
            print(f"\n🔄 Step 3/3: Scraping up to {max_retweets} retweeters...")
            retweets_list = self._scrape_retweeters(tid, max_retweets)

            # Combine for this tweet
            all_data.append(
                {
                    "tweet_id": tid,
                    "profile_info": profile_info,
                    "replies": replies_list,
                    "retweeters": retweets_list,
                }
            )

            print(f"\n✅ Completed processing Tweet ID: {tid}")

        # Calculate totals
        total_replies = sum(len(t.get("replies", [])) for t in all_data)
        total_retweeters = sum(len(t.get("retweeters", [])) for t in all_data)

        print(
            f"\nScraping complete: {len(all_data)} tweets, {total_replies} replies, {total_retweeters} retweeters"
        )

        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = self._save_raw_data(all_data, username, timestamp)
        final_file = self._process_and_save_final(all_data, username, timestamp)

        return {
            "raw_file": raw_file,
            "final_file": final_file,
            "tweets": len(all_data),
            "replies": total_replies,
            "retweeters": total_retweeters,
        }

    # ------------------------
    # Helper Methods
    # ------------------------

    def _run_actor_and_get_items(self, actor_id, run_input):
        """Run an actor and return dataset items list."""
        try:
            run = self.client.actor(actor_id).call(run_input=run_input)
        except Exception as e:
            print(f"Error running actor {actor_id}: {e}")
            return []

        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            return []

        try:
            items = list(self.client.dataset(dataset_id).iterate_items())
        except Exception as e:
            print(
                f"Error fetching dataset {dataset_id} items for actor {actor_id}: {e}"
            )
            return []

        return items

    def _fetch_profile_info(self, profile_url, username, tweet_id):
        """Fetch profile info and main tweet data."""
        profile_items = self._run_actor_and_get_items(
            self.profile_actor_id,
            {
                "startUrls": [{"url": profile_url}],
                "tweetsDesired": 1,
                "includeUserInfo": True,
            },
        )

        if not profile_items:
            print(f"⚠️  Warning: no profile info returned for @{username}.")
            return {}

        item = profile_items[0]
        pdata = item.get("user", {}) or {}
        profile_info = {
            "id": item.get("id"),
            "tweet_url": f"https://x.com/{username}/status/{tweet_id}",
            "verified": pdata.get("verified"),
            "username": f"@{username}",
            "user": {
                "avatar": pdata.get("profileImageUrl") or pdata.get("avatar"),
                "username": f"@{username}",
                "userFullName": pdata.get("name"),
                "url": profile_url,
                "totalFollowers": pdata.get("followers") or pdata.get("totalFollowers"),
            },
            "tweet_text": item.get("text") or item.get("tweetText") or "",
        }
        print(
            f"✅ Profile fetched: {profile_info['user'].get('totalFollowers')} followers"
        )
        return profile_info

    def _scrape_replies(self, tweet_id, max_replies):
        """Scrape replies for a tweet."""
        replies = self._run_actor_and_get_items(
            self.replies_actor_id,
            {"conversation_ids": [tweet_id], "max_items_per_conversation": max_replies},
        )

        replies_list = []
        for r in replies:
            author = r.get("author", {}) or {}
            replies_list.append(
                {
                    "tweet_id": r.get("id"),
                    "text": r.get("text"),
                    "created_at": r.get("createdAt"),
                    "author_username": author.get("userName"),
                    "author_name": author.get("name"),
                    "author_verified": (
                        author.get("isVerified")
                        if author.get("isVerified") is not None
                        else author.get("verified")
                    ),
                    "author_followers": author.get("followers"),
                    "author_following": author.get("following"),
                    "retweet_count": r.get("retweetCount") or r.get("retweets"),
                    "reply_count": r.get("replyCount") or r.get("replies"),
                    "like_count": r.get("likeCount") or r.get("likes"),
                    "quote_count": r.get("quoteCount") or r.get("quotes"),
                    "is_reply": r.get("isReply"),
                    "in_reply_to_id": r.get("inReplyToId")
                    or r.get("inReplyToStatusId"),
                    "tweet_url": (
                        f"https://x.com/{author.get('userName')}/status/{r.get('id')}"
                        if author.get("userName") and r.get("id")
                        else None
                    ),
                }
            )

        print(f"✅ Scraped {len(replies_list)} replies")
        return replies_list

    def _scrape_retweeters(self, tweet_id, max_retweets):
        """Scrape retweeters for a tweet."""
        retweeters = self._run_actor_and_get_items(
            self.retweeters_actor_id,
            {"tweet_ids": [tweet_id], "max_items_per_tweet": max_retweets},
        )

        retweets_list = []
        for r in retweeters:
            retweets_list.append(
                {
                    "userName": r.get("userName"),
                    "name": r.get("name"),
                    "isVerified": r.get("isVerified"),
                    "followers": r.get("followers"),
                    "following": r.get("following"),
                    "profilePicture": r.get("profilePicture"),
                    "description": r.get("description"),
                    "url": r.get("url"),
                }
            )

        print(f"✅ Scraped {len(retweets_list)} retweeters")
        return retweets_list

    # ------------------------
    # Data Saving / Processing
    # ------------------------

    def _save_raw_data(self, data, username, timestamp):
        """Save raw/preprocessed data."""
        json_file = self.preprocessing_dir / f"{username}_raw_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Preprocessing JSON saved: {json_file}")
        return json_file

    def _process_and_save_final(self, data, username, timestamp):
        """Process and save final CSV/JSON with sentiment analysis."""
        rows = []

        for tweet_data in data:
            tid = tweet_data["tweet_id"]
            profile = tweet_data["profile_info"].get("user", {})
            tweet_link = tweet_data["profile_info"].get("tweet_url", "")
            profile_link = profile.get("url", "")

            replies_list = tweet_data["replies"]
            retweeters_list = tweet_data["retweeters"]

            # Get main tweet text
            if replies_list:
                main_tweet_text = replies_list[0].get("text", "")
                replies_list = replies_list[1:]
            else:
                main_tweet_text = tweet_data["profile_info"].get("tweet_text", "")

            # Add replies
            for r in replies_list:
                rows.append(
                    {
                        "tweet_id": tid,
                        "interaction_type": "reply",
                        "username": r.get("author_username"),
                        "name": r.get("author_name"),
                        "verified": r.get("author_verified"),
                        "followers": r.get("author_followers"),
                        "text": r.get("text"),
                        "reply_url": r.get("tweet_url"),
                        "tweet_text": main_tweet_text,
                        "profile_username": profile.get("username"),
                        "profile_fullname": profile.get("userFullName"),
                        "profile_followers": profile.get("totalFollowers"),
                        "profile_url": profile_link,
                        "tweet_url": tweet_link,
                        "retweet_count": r.get("retweet_count"),
                        "reply_count": r.get("reply_count"),
                        "like_count": r.get("like_count"),
                        "quote_count": r.get("quote_count"),
                        "created_at": r.get("created_at"),
                    }
                )

            # Add retweeters
            for r in retweeters_list:
                rows.append(
                    {
                        "tweet_id": tid,
                        "interaction_type": "retweeter",
                        "username": r.get("userName"),
                        "name": r.get("name"),
                        "verified": r.get("isVerified"),
                        "followers": r.get("followers"),
                        "text": "",
                        "reply_url": "",
                        "tweet_text": main_tweet_text,
                        "profile_username": profile.get("username"),
                        "profile_fullname": profile.get("userFullName"),
                        "profile_followers": profile.get("totalFollowers"),
                        "profile_url": profile_link,
                        "tweet_url": tweet_link,
                        "retweet_count": None,
                        "reply_count": None,
                        "like_count": None,
                        "quote_count": None,
                        "created_at": None,
                    }
                )

        if not rows:
            return None

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Apply sentiment analysis
        if not df.empty:
            df = self._apply_sentiment_analysis(df)

        # Save final files
        json_file = self.final_dir / f"{username}_all_tweets_{timestamp}.json"
        csv_file = self.final_dir / f"{username}_all_tweets_{timestamp}.csv"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        df.to_csv(csv_file, index=False, encoding="utf-8-sig")

        print(f"\n{'='*70}")
        print(f"✅ SCRAPING COMPLETE!")
        print(f"{'='*70}")
        print(f"💾 Final Combined JSON saved: {json_file}")
        print(f"💾 Final Combined CSV  saved: {csv_file}")
        print(f"   • Total rows: {len(df)}")
        print(f"   • Replies: {len(df[df['interaction_type'] == 'reply'])}")
        print(f"   • Retweeters: {len(df[df['interaction_type'] == 'retweeter'])}")
        print(f"{'='*70}\n")

        return csv_file

    def _apply_sentiment_analysis(self, df):
        """Apply sentiment analysis to dataframe."""
        if df.empty:
            return df

        try:
            print("\n" + "=" * 70)
            print("🤖 APPLYING SENTIMENT ANALYSIS...")
            print("=" * 70)

            analyzer = TwitterSentimentAnalyzer()
            df = analyzer.analyze_twitter_data(df)

            print("✅ Sentiment analysis completed!")
            print("=" * 70 + "\n")
            return df
        except Exception as e:
            print(f"⚠️  Sentiment analysis failed: {e}")
            print("Continuing without sentiment data...")
            return df


# ------------------------
# MAIN FUNCTION
# ------------------------
def run_twitter_scraper():
    """Run Twitter scraper - main entry point for unified integration"""
    print("\n" + "=" * 70)
    print("TWITTER SCRAPER".center(70))
    print("=" * 70 + "\n")

    API_TOKEN = os.getenv("APIFY_API_TOKEN")
    if not API_TOKEN:
        API_TOKEN = input("Enter your Apify API token: ").strip()

    if not API_TOKEN:
        print("❌ No API token provided. Returning to main menu.")
        return

    scraper = TwitterScraperPipeline(API_TOKEN)

    # Get user input
    user_input = input(
        "Enter Twitter username or profile URL (example: @elonmusk or https://x.com/elonmusk): "
    ).strip()

    if not user_input:
        print("❌ No username provided. Returning to main menu.")
        return

    # Get tweet IDs
    tweet_ids_input = input("Enter Tweet ID(s) separated by commas: ").strip()
    if not tweet_ids_input:
        print("❌ No Tweet IDs provided. Returning to main menu.")
        return

    tweet_ids = [t.strip() for t in tweet_ids_input.split(",") if t.strip()]

    # Get limits
    try:
        max_replies = int(
            input("Enter maximum replies to scrape per tweet (default 200): ").strip()
            or 200
        )
        if max_replies < 20:
            print("Minimum allowed is 20 — setting to 20.")
            max_replies = 20
    except ValueError:
        max_replies = 200

    try:
        max_retweets = int(
            input(
                "Enter maximum retweeters to scrape per tweet (default 200): "
            ).strip()
            or 200
        )
    except ValueError:
        max_retweets = 200

    print("\n🚀 Starting scrape...\n")

    try:
        result = scraper.scrape_from_user(
            user_input=user_input,
            tweet_ids=tweet_ids,
            max_replies=max_replies,
            max_retweets=max_retweets,
        )

        print("\n" + "=" * 70)
        print("✅ SCRAPING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(
            f"\n📁 Raw data saved: {result['raw_file'].name if result['raw_file'] else 'None'}"
        )
        print(
            f"📊 Final data saved: {result['final_file'].name if result['final_file'] else 'None'}"
        )
        print(f"🐦 Tweets scraped: {result['tweets']}")
        print(f"💬 Replies scraped: {result['replies']}")
        print(f"🔄 Retweeters scraped: {result['retweeters']}")
        print("=" * 70 + "\n")

        # Dashboard trigger
        if result["final_file"]:
            print("\n" + "=" * 70)
            show_dashboard = (
                input("📊 Would you like to view the dashboard? (y/n): ")
                .strip()
                .lower()
            )
            if show_dashboard in ["y", "yes"]:
                print("\n🚀 Launching dashboard...\n")
                run_twitter_dashboard(result["final_file"])
            else:
                print(
                    "💡 You can run the dashboard anytime using: python dashboard_twitter.py"
                )
            print("=" * 70 + "\n")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR OCCURRED")
        print("=" * 70)
        print(f"\n{str(e)}\n")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 70 + "\n")


# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    """Main program loop"""
    while True:
        print_banner()
        choice = get_main_choice()

        if choice == "1":
            run_facebook_scraper()
        elif choice == "2":
            run_instagram_scraper()
        elif choice == "3":
            run_twitter_scraper()
        elif choice == "4":
            print("\n👋 Thank you for using the Unified Social Media Scraper!")
            print("=" * 70 + "\n")
            break
        else:
            print("\n❌ Invalid choice. Please select 1-4.\n")

        input("\n Press Enter to continue...")


if __name__ == "__main__":
    main()
