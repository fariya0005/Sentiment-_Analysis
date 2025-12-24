Unified Social Media Scraper & Sentiment Analysis Platform
A Python-based, multi-platform system for scraping Facebook, Instagram, and Twitter (X) data and performing sentiment analysis with optional visualization.
✨ Features
•	Multi-platform scraping: Facebook, Instagram, Twitter (X)
•	Multiple scraping modes per platform
•	Extracts posts, comments, replies, hashtags, profiles, reactions, retweeters
•	Built-in sentiment analysis for posts, comments, and replies
•	Emoji and engagement metrics analysis
•	Automatic data cleaning and preprocessing
•	CSV / JSON export support
•	Modular and extensible architecture
•	Optional Streamlit-based GUI
🛠 System Requirements
•	Python: 3.9+
Required Libraries
pip install pandas apify-client python-dotenv streamlit


Environment Setup
Create a .env file in the project root:
APIFY_API_TOKEN=your_apify_token_here



📁 Project Structure
project_root/
│
├─ Data/
│  ├─ Facebook/
│  │  ├─ preprocessing/
│  │  └─ final/
│  ├─ Instagram/
│  │  ├─ preprocessing/
│  │  └─ final/
│  └─ Twitter/
│     ├─ preprocessing/
│     └─ final/
│
├─ facebook_scraper.py
├─ instagram_scraper.py
├─ twitter_scraper.py
├─ sentiment_facebook.py
├─ sentiment_insta.py
├─ sentiment_twitter.py
├─ dashboard_facebook.py
├─ dashboard_insta.py
├─ dashboard_twitter.py
├─ scraper.py        # CLI entry point
├─ app.py            # Streamlit UI entry point
└─ .env

🧱 Architecture Overview
The system follows a modular, class-based architecture where each platform has its own pipeline.
Component	Description
FacebookScraperPipeline	Scrapes Facebook posts, comments, reactions, sentiment
InstagramScraperPipeline	Scrapes profiles, hashtags, posts, comments
TwitterScraperPipeline	Scrapes tweets, replies, retweeters
Sentiment Modules	Platform-specific sentiment analysis
Dashboards	Visualization & analytics
Streamlit App	Unified user interface

🤖 Apify Actors Used
Platform	Actor	Purpose
Facebook	powerai/facebook-post-search-scraper	Page & keyword posts
Facebook	apify/facebook-comments-scraper	Nested comments
Facebook	apify/facebook-groups-scraper	Group posts
Instagram	apify/instagram-profile-scraper	Profile scraping
Instagram	apify/instagram-post-scraper	Post scraping
Instagram	apify/instagram-hashtag-scraper	Hashtag-based posts
Instagram	louisdeconinck/instagram-comments-scraper	Comment scraping
Twitter	web.harvester/twitter-scraper	Tweets & profiles
Twitter	kaitoeasyapi/twitter-reply	Tweet replies
Twitter	kaitoeasyapi/tweet-reweet-userlist	Retweeters


🔄 Data Flow
1.	User provides input (URL, username, hashtag, keyword, IDs)
2.	Corresponding Apify actor is triggered
3.	Posts, comments, replies, reactions are extracted
4.	Data is cleaned and normalized
5.	Engagement and sentiment metrics are computed
6.	Raw and processed files are saved


▶️ How to Run
Run via Terminal
python scraper.py
Run with Streamlit UI
streamlit run app.py


⚠️ Error Handling
•	Missing or invalid API tokens are prompted
•	Invalid URLs or IDs are skipped with warnings
•	Actor failures retry automatically
•	Sentiment analysis errors do not stop execution

