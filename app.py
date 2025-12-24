import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from scraper import (
    InstagramScraperPipeline,
    TwitterScraperPipeline,
    FacebookScraperPipeline,
    run_instagram_dashboard,
    run_twitter_dashboard,
)
from sentiment_facebook import SentimentAnalyzer

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Unified Social Media Scraper", layout="wide")
st.title("🌐 Unified Social Media Scraper")

# -----------------------
# API Token
# -----------------------
API_TOKEN = os.getenv("APIFY_API_TOKEN")
if not API_TOKEN:
    API_TOKEN = st.text_input("Enter your Apify API token:", type="password")

if not API_TOKEN:
    st.warning("Please provide an Apify API token to start scraping.")
    st.stop()

# -----------------------
# Platform Selector
# -----------------------
platform = st.selectbox("Select Platform:", ["Instagram", "Twitter", "Facebook"])

# ═══════════════════════════════════════════════════════════════════
# INSTAGRAM SCRAPER
# ═══════════════════════════════════════════════════════════════════
if platform == "Instagram":
    st.subheader("📊 Instagram Scraper")

    # Initialize scraper
    scraper = InstagramScraperPipeline(API_TOKEN)
    cookies_path = "cookies.txt"

    # Scraping Mode
    mode = st.radio(
        "Select Scraping Mode:", ["Profile", "Keyword/Hashtag", "Post URL(s)"]
    )

    # Scraping Function
    def scrape_and_visualize(df, output_file):
        if df.empty:
            st.warning("No data found.")
            return

        st.success(f"✅ Scraping complete! Data saved at `{output_file}`")

        # 1️⃣ POST SENTIMENT PIE CHART
        if "post_sentiment_label" in df.columns:
            st.write("### 🥧 Post Sentiment Distribution")
            sentiment_counts = df["post_sentiment_label"].value_counts()
            st.plotly_chart(
                {
                    "data": [
                        {
                            "labels": sentiment_counts.index,
                            "values": sentiment_counts.values,
                            "type": "pie",
                        }
                    ],
                    "layout": {"title": "Post Sentiment"},
                },
                width="stretch",
            )

        # 2️⃣ COMMENT SENTIMENTS
        if "comment_sentiment_label" in df.columns:
            comment_df = df[df["comment_sentiment_label"].notna()]
            total_comments = len(comment_df)
            if total_comments > 0:
                st.write(f"### 💬 Comment Sentiments ({total_comments} comments)")
                comment_counts = comment_df["comment_sentiment_label"].value_counts()
                for label, count in comment_counts.items():
                    percent = round((count / total_comments) * 100, 1)
                    st.write(f"• {label}: {count} ({percent}%)")

        # 3️⃣ DATA PREVIEW
        st.write("### 📄 Data Preview")
        st.dataframe(df.head(50))

        # 4️⃣ DOWNLOAD BUTTON
        with open(output_file, "rb") as f:
            st.download_button(
                label="📂 Download CSV with Sentiment",
                data=f,
                file_name=Path(output_file).name,
                mime="text/csv",
            )

        # 5️⃣ DASHBOARD BUTTON
        if st.button("Open Instagram Dashboard"):
            run_instagram_dashboard(output_file)

    # PROFILE SCRAPING
    if mode == "Profile":
        st.subheader("Scrape by Instagram Profile")
        username = st.text_input("Enter Instagram username (without @):")
        max_posts = st.number_input(
            "Number of recent posts to scrape:", min_value=1, max_value=200, value=12
        )
        include_comments = st.checkbox("Include comments?", value=True)
        max_comments = st.number_input(
            "Max comments per post:", min_value=1, max_value=500, value=50
        )

        if st.button("Scrape Profile"):
            if not username:
                st.error("Please provide a username.")
            else:
                with st.spinner(f"Scraping profile @{username}..."):
                    df = scraper.scrape_profile(
                        username=username,
                        max_posts=max_posts,
                        include_comments=include_comments,
                        max_comments=max_comments,
                        cookies_path=cookies_path,
                    )
                    output_file = scraper.save_final_data(df, f"profile_{username}")
                    scrape_and_visualize(df, output_file)

    # KEYWORD/HASHTAG SCRAPING
    elif mode == "Keyword/Hashtag":
        st.subheader("Scrape by Keyword or Hashtag")
        keyword = st.text_input("Enter keyword or hashtag (# optional):")
        max_posts = st.number_input(
            "Number of posts to scrape:", min_value=1, max_value=500, value=50
        )
        include_comments = st.checkbox("Include comments?", value=True)
        max_comments = st.number_input(
            "Max comments per post:", min_value=1, max_value=500, value=50
        )

        if st.button("Scrape Keyword"):
            if not keyword:
                st.error("Please provide a keyword or hashtag.")
            else:
                with st.spinner(f"Scraping posts for #{keyword}..."):
                    df = scraper.scrape_keyword(
                        keyword=keyword,
                        max_posts=max_posts,
                        include_comments=include_comments,
                        max_comments=max_comments,
                        cookies_path=cookies_path,
                    )
                    output_file = scraper.save_final_data(df, f"keyword_{keyword}")
                    scrape_and_visualize(df, output_file)

    # POST URL SCRAPING
    elif mode == "Post URL(s)":
        st.subheader("Scrape by Post URLs")
        urls_input = st.text_area("Enter Instagram post URLs (comma-separated):")
        include_comments = st.checkbox("Include comments?", value=True)
        max_comments = st.number_input(
            "Max comments per post:", min_value=1, max_value=500, value=100
        )

        if st.button("Scrape Post URLs"):
            post_urls = [u.strip() for u in urls_input.split(",") if u.strip()]
            if not post_urls:
                st.error("Please enter at least one URL.")
            else:
                with st.spinner(f"Scraping {len(post_urls)} post(s)..."):
                    df = scraper.scrape_post_urls(
                        post_urls=post_urls,
                        include_comments=include_comments,
                        max_comments=max_comments,
                        cookies_path=cookies_path,
                    )
                    output_file = scraper.save_final_data(df, "post_urls")
                    scrape_and_visualize(df, output_file)

# ═══════════════════════════════════════════════════════════════════
# TWITTER SCRAPER
# ═══════════════════════════════════════════════════════════════════
elif platform == "Twitter":
    st.subheader("🐦 Twitter Scraper")

    # Initialize scraper
    scraper = TwitterScraperPipeline(API_TOKEN)

    # INPUTS
    user_input = st.text_input(
        "Enter Twitter username or profile URL (example: @elonmusk or https://x.com/elonmusk):"
    )

    tweet_ids_input = st.text_area(
        "Enter Tweet ID(s), separated by commas:",
        placeholder="Example: 1234567890, 9876543210",
    )

    max_replies = st.number_input(
        "Max replies per tweet:", min_value=20, max_value=2000, value=200
    )

    max_retweets = st.number_input(
        "Max retweeters per tweet:", min_value=10, max_value=2000, value=200
    )

    # SCRAPE BUTTON
    if st.button("Scrape Twitter"):
        if not user_input:
            st.error("Please enter a username or profile URL.")
            st.stop()

        if not tweet_ids_input.strip():
            st.error("Please enter at least one Tweet ID.")
            st.stop()

        tweet_ids = [t.strip() for t in tweet_ids_input.split(",") if t.strip()]

        with st.spinner(f"Scraping Twitter data for {user_input} ..."):
            result = scraper.scrape_from_user(
                user_input=user_input,
                tweet_ids=tweet_ids,
                max_replies=max_replies,
                max_retweets=max_retweets,
            )

        st.success("🎉 Scraping Completed Successfully!")

        # Display summary
        st.write("### 📊 Summary")
        st.write(f"**Tweets scraped:** {result['tweets']}")
        st.write(f"**Replies scraped:** {result['replies']}")
        st.write(f"**Retweeters scraped:** {result['retweeters']}")

        if result["final_file"]:
            st.success(f"✅ Final data saved at: `{result['final_file']}`")

            # Load CSV for visualization
            df = pd.read_csv(result["final_file"])

            st.subheader("📈 Sentiment Analysis Overview")

            # CHECK SENTIMENT COLUMNS
            sentiment_cols = [
                "tweet_sentiment_label",
                "tweet_sentiment_score",
                "interaction_sentiment_label",
                "interaction_sentiment_score",
            ]

            available_cols = [c for c in sentiment_cols if c in df.columns]

            if not available_cols:
                st.warning("⚠️ No sentiment columns found in CSV.")
                st.dataframe(df.head())
            else:
                st.success("Sentiment data loaded successfully!")

                # 1️⃣ TWEET SENTIMENT PIE CHART
                if "tweet_sentiment_label" in df.columns:
                    st.write("### 🥧 Sentiment Distribution (Tweet Text)")
                    sentiment_counts = df["tweet_sentiment_label"].value_counts()

                    st.plotly_chart(
                        {
                            "data": [
                                {
                                    "labels": sentiment_counts.index,
                                    "values": sentiment_counts.values,
                                    "type": "pie",
                                }
                            ],
                            "layout": {"title": "Tweet Sentiment"},
                        },
                        width="stretch",
                    )

                # 2️⃣ REPLY SENTIMENTS (TEXT ONLY, exclude retweeters)
                if (
                    "interaction_sentiment_label" in df.columns
                    and "interaction_type" in df.columns
                ):
                    reply_df = df[
                        (df["interaction_sentiment_label"].notna())
                        & (df["interaction_type"] == "reply")
                    ]
                    total_replies = len(reply_df)

                    st.write(f"### 💬 Reply Sentiments ({total_replies} replies)")

                    interaction_counts = reply_df[
                        "interaction_sentiment_label"
                    ].value_counts()

                    for label, count in interaction_counts.items():
                        percent = round((count / total_replies) * 100, 1)
                        st.write(f"• {label}: {count} ({percent}%)")

                # 3️⃣ DATA PREVIEW
                st.write("### 📄 Data Preview")
                st.dataframe(df.head(50))

            # 📥 DOWNLOAD BUTTON
            with open(result["final_file"], "rb") as f:
                st.download_button(
                    label="📂 Download CSV with Sentiment",
                    data=f,
                    file_name=Path(result["final_file"]).name,
                    mime="text/csv",
                )

            # Button to view Dashboard
            if st.button("Open Twitter Dashboard"):
                run_twitter_dashboard(result["final_file"])
        else:
            st.warning("No final CSV generated. Please check logs for errors.")

# ═══════════════════════════════════════════════════════════════════
# FACEBOOK SCRAPER
# ═══════════════════════════════════════════════════════════════════
elif platform == "Facebook":
    st.subheader("📘 Facebook Scraper + Sentiment Analysis")

    # MODE SELECTOR
    mode = st.selectbox("Select Scraping Mode:", ["Page", "Group", "Keyword"])
    mode_map = {"Page": "page", "Group": "group", "Keyword": "keyword"}
    mode_value = mode_map[mode]

    # TARGET INPUT
    target = st.text_input(f"Enter {mode} URL or keyword:")

    # MAX COUNTS
    max_posts = st.number_input("Max Posts", min_value=3, max_value=1000, value=10)
    max_comments = st.number_input(
        "Max Comments per Post", min_value=0, max_value=500, value=20
    )

    # RUN SCRAPING
    if st.button("🚀 Start Scraping"):
        if not target:
            st.warning("Please enter a valid URL or keyword.")
        else:
            st.info("Starting scraping... This may take a few minutes ⏳")
            scraper = FacebookScraperPipeline(API_TOKEN)
            try:
                result = scraper.scrape_from_url(
                    page_url=target,
                    max_posts=max_posts,
                    max_comments_per_post=max_comments,
                    mode=mode_value,
                )
                if result["posts"] == 0:
                    st.warning(
                        "No posts/comments found. Check the URL/keyword and mode."
                    )
                else:
                    st.success("✅ Scraping Completed!")

                    # Display summary
                    st.write("### 📊 Summary")
                    st.write(f"**Posts scraped:** {result['posts']}")
                    st.write(f"**Comments scraped:** {result['comments']}")
                    st.write(f"**Estimated cost:** ${result['cost']}")

                    # SENTIMENT ANALYSIS
                    if result["final_file"]:
                        df_path = Path(result["final_file"])
                        df = pd.read_csv(df_path)
                        st.subheader("🧠 Sentiment Analysis")
                        # Run sentiment
                        analyzer = SentimentAnalyzer()
                        df = analyzer.analyze_posts_and_comments(
                            df, post_col="post_message", comment_col="comment_text"
                        )

                        # 1️⃣ POST SENTIMENT PIE
                        if "post_sentiment_label" in df.columns:
                            st.write("### 🥧 Post Sentiment Distribution")
                            post_counts = df["post_sentiment_label"].value_counts()
                            st.plotly_chart(
                                {
                                    "data": [
                                        {
                                            "labels": post_counts.index,
                                            "values": post_counts.values,
                                            "type": "pie",
                                        }
                                    ],
                                    "layout": {"title": "Post Sentiment"},
                                },
                                use_container_width=True,
                            )

                        # 2️⃣ COMMENT SENTIMENT (TEXT ONLY)
                        if "comment_sentiment_label" in df.columns:
                            st.write("### 💬 Comment Sentiment Distribution")
                            comment_counts = df[
                                "comment_sentiment_label"
                            ].value_counts()
                            for sentiment, count in comment_counts.items():
                                st.write(f"• **{sentiment.upper()}**: {count}")

                        # 3️⃣ DATA PREVIEW
                        st.write("### 📄 Data Preview")
                        st.dataframe(df.head(50))

                        # Save updated CSV
                        df.to_csv(df_path, index=False)

                        # 📥 DOWNLOAD BUTTON
                        with open(df_path, "rb") as f:
                            st.download_button(
                                label="📂 Download CSV with Sentiment",
                                data=f,
                                file_name=df_path.name,
                                mime="text/csv",
                            )
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
