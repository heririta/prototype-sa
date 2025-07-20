import streamlit as st
from PIL import Image
import base64
import os
import pandas as pd



st.set_page_config(page_title="Portal Berita", layout="wide")


# -------------------------------
# Login form
# -------------------------------

# -------------------------------
# Setelah login
# -------------------------------
st.sidebar.title("📚 Menu")
menu = st.sidebar.radio("Pilih", ["Crawling", "Sentiment", "Dashboard", "Logout", ])

if menu == "Logout":
    st.session_state.token = None
    st.success("Berhasil logout")
    st.rerun()

# -------------------------------
# Daftar berita dengan filter & pagination
# -------------------------------

elif menu == "Sentiment":
    title_filter = st.text_input("🔍 Filter judul berita")
    skip = st.sidebar.number_input("⏭️ Skip", min_value=0, step=1)
    limit = st.sidebar.slider("🧮 Limit", 1, 20, 6)

    st.title("📋 Daftar Berita")
    # news_list = api.get_news(st.session_state.token, title_filter, skip, limit)


    # Yerel placeholder görüntüyü yüklemek için fonksiyon
    def get_base64_encoded_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    # Placeholder görüntü yolu - kendi dosya yolunuzu buraya yazın
    PLACEHOLDER_IMAGE_PATH = "s_positive.jpg"  # Bu dosyanın Python kodunuzla aynı dizinde olduğunu varsayıyorum
    PLACEHOLDER_IMAGE_PATH2 = "s_negative.jpg"

    # Görüntüyü base64 formatına dönüştür (eğer dosya mevcutsa)
    if os.path.exists(PLACEHOLDER_IMAGE_PATH):
        img_base64 = get_base64_encoded_image(PLACEHOLDER_IMAGE_PATH)
        PLACEHOLDER_IMAGE = f"data:image/jpeg;base64,{img_base64}"
    else:
        # Dosya bulunamazsa yedek olarak online bir görsel kullan
        PLACEHOLDER_IMAGE = "https://img.freepik.com/free-vector/artificial-intelligence-ai-robot-server-room-digital-technology-banner_39422-794.jpg"
        st.warning(f"Placeholder image not found at {PLACEHOLDER_IMAGE_PATH}. Using fallback image.")

    # Görüntüyü base64 formatına dönüştür (eğer dosya mevcutsa)
    if os.path.exists(PLACEHOLDER_IMAGE_PATH2):
        img_base642 = get_base64_encoded_image(PLACEHOLDER_IMAGE_PATH2)
        PLACEHOLDER_IMAGE2 = f"data:image/jpeg;base64,{img_base642}"
    else:
        # Dosya bulunamazsa yedek olarak online bir görsel kullan
        PLACEHOLDER_IMAGE2 = "https://img.freepik.com/free-vector/artificial-intelligence-ai-robot-server-room-digital-technology-banner_39422-794.jpg"
        st.warning(f"Placeholder image not found at {PLACEHOLDER_IMAGE_PATH2}. Using fallback image.")



    # Apply CSS styling for cards
    st.markdown("""
    <style>
    .news-card {
        border-radius: 20px;
        padding: 0;
        margin-bottom: 20px;
        background-color: #E8E8E8;
        height: 480px;
        overflow: hidden;
        position: relative;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
    }
    .news-image-container {
        width: 100%;
        height: 220px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px 10px 0 10px;
    }
    .news-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 12px;
    }
    .news-content {
        padding: 12px 15px;
    }
    .news-title {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 10px;
        color: #000;
        line-height: 1.3;
        max-height: 105px;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .news-meta {
        display: flex;
        justify-content: space-between;
        margin-bottom: 12px;
        align-items: center;
        border-bottom: 1px solid #ddd;
        padding-bottom: 8px;
    }
    .news-source {
        color: #555;
        font-size: 12px;
        font-style: italic;
    }
    .news-date {
        color: #555;
        font-size: 12px;
        text-align: right;
        font-style: italic;
    }
    .news-description {
        color: #333;
        font-size: 13px;
        padding-bottom: 10px;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 5;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 90px;
    }
    </style>
    """, unsafe_allow_html=True)

    # for news in news_list:
    #     with st.expander(f"📰 {news['title']} (ID: {news['id']})"):
    #         st.markdown(news["content"])
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             if st.button(f"✏️ Update Judul - {news['id']}"):
    #                 new_title = st.text_input(f"Update judul (ID {news['id']})", key=f"edit-{news['id']}")
    #                 if new_title:
    #                     res = api.update_news(st.session_state.token, news["id"], new_title)
    #                     st.rerun()
    #         with col2:
    #             if st.button(f"🗑️ Hapus - {news['id']}"):
    #                 api.delete_news(st.session_state.token, news["id"])
    #                 st.warning("Berita dihapus.")
    #                 st.rerun()


    

    # df_filtered = pd.DataFrame(news_list)

    # buatkan variable untuk menampung data dari file news.json
    df_filtered = pd.read_json('news.json')

    # if not df_filtered.empty:
    #     st.dataframe(df_filtered, use_container_width=True) 
    # else:
    #     st.warning("Tidak ada berita yang ditemukan.")  

    # Display results
    if len(df_filtered) > 0:
        
        # Apply CSS styling for cards
        st.markdown("""
        <style>
        .news-card {
            border-radius: 20px;
            padding: 0;
            margin-bottom: 20px;
            background-color: #E8E8E8;
            height: 480px;
            overflow: hidden;
            position: relative;
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
        }
        .news-image-container {
            width: 100%;
            height: 220px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px 10px 0 10px;
        }
        .news-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 12px;
        }
        .news-content {
            padding: 12px 15px;
        }
        .news-title {
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 10px;
            color: #000;
            line-height: 1.3;
            max-height: 105px;
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .news-meta {
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
            align-items: center;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }
        .news-source {
            color: #555;
            font-size: 12px;
            font-style: italic;
        }
        .news-date {
            color: #555;
            font-size: 12px;
            text-align: right;
            font-style: italic;
        }
        .news-description {
            color: #333;
            font-size: 13px;
            padding-bottom: 10px;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 5;
            -webkit-box-orient: vertical;
            overflow: hidden;
            height: 90px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create 3-column layout
        num_cols = 3
        
        # Process news items in groups of 3 for the grid
        for i in range(0, len(df_filtered), num_cols):
            cols = st.columns(num_cols)
            
            # Get the current batch of news items
            current_batch = df_filtered.iloc[i:i+num_cols]
            
            # Display each news item in its column
            for j, (_, row) in enumerate(current_batch.iterrows()):
                if j < len(cols):  # Ensure we have a column for this item
                    with cols[j]:
                        # Eğer kaynak deeplearning.ai ise veya geçerli bir görüntü yoksa, placeholder kullan
                        # if row['is_sentiment'] == "false" or not pd.notna(row.get('Image')) or row.get('Image') is None:
                        if row['is_sentiment'] == "1" :
                            image_url = PLACEHOLDER_IMAGE2
                        else:
                            image_url = PLACEHOLDER_IMAGE
                        
                        # Format the date
                        date_str = row['publish_date']
                        
                        # Truncate description if it's too long
                        description = row['content'][:150] + "..." if len(row['content']) > 150 else row['content']
                        
                        # Display card with HTML
                        html_content = f"""
                        <a href="{row['link']}" target="_blank" style="text-decoration: none; color: inherit;">
                            <div class="news-card">
                                <div class="news-image-container">
                                    <img src="{image_url}" class="news-image" onerror="this.onerror=null;this.src='{PLACEHOLDER_IMAGE}';">
                                </div>
                                <div class="news-content">
                                    <div class="news-title">{row['content']}</div>
                                    <div class="news-meta">
                                        <div class="news-source">{row['content']}</div>
                                        <div class="news-date">{date_str}</div>
                                    </div>
                                    <div class="news-description">{description}</div>
                                </div>
                            </div>
                        </a>
                        """
                        st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.warning("No news found with the selected filters. Please adjust your date range or source selection.")


# -------------------------------
# Crawling
# -------------------------------

elif menu == "Crawling":
    import streamlit as st
    import pandas as pd
    import requests
    import urllib.parse
    from tqdm import tqdm
    from datetime import datetime, timedelta
    import base64
    import time
    import re
    import altair as alt
    from pytrends.request import TrendReq
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # ============ STREAMLIT CONFIG ============
    st.set_page_config(page_title="News Crawler PRO++", page_icon="📰", layout="wide")

    # CSS
    st.markdown("""
    <style>
    body {background:#f9fafc;}
    .result-card {
        background:#fff; border-radius:12px; padding:12px; margin-bottom:8px;
        box-shadow:0 2px 8px rgba(0,0,0,0.05);
    }
    .keyword-highlight {color:#e74c3c; font-weight:bold;}
    .stButton>button {
        background: linear-gradient(90deg,#1e90ff,#00bfff);
        color:white;font-weight:bold;border-radius:8px;padding:8px 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    # HEADER
    st.title("📰 News Crawler")
    # st.caption("Multi-keyword • Highlight • Dashboard • Google Trends • Wordcloud • Auto-refresh • Filter Toggle")

    # ============ INPUT ============
    keyword_input = st.text_area("Masukkan daftar kata kunci (pisahkan dengan koma)", "LPS, OJK, BI, Bank")
    keywords = [k.strip() for k in keyword_input.split(",") if k.strip()]

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Tanggal Mulai", datetime.now().date() - timedelta(days=7))
    with col2:
        end_date = st.date_input("Tanggal Akhir", datetime.now().date())

    filter_date_on = st.sidebar.checkbox("Aktifkan Filter Tanggal", value=True)
    auto_refresh = st.sidebar.checkbox("Aktifkan Auto-refresh")
    interval = st.sidebar.number_input("Interval refresh (detik)", 10, 300, 30) if auto_refresh else None

    crawl_btn = st.sidebar.button("🔍 Mulai Crawling")

    # Buat dataframe objek untuk RSS
    objects_df = pd.DataFrame([{"code": f"KWD{i+1}", "name": kw} for i, kw in enumerate(keywords)])

    # ============ FUNGSI CRAWLER (Google RSS) ============
    def get_object_from_articles(objects_df, start_date, end_date, filter_on=True):
        news_data = []

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        for _, row in tqdm(objects_df.iterrows(), total=len(objects_df)):
            keyword = row["name"]
            params = {"q": keyword, "hl": 'id'}
            url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(params)

            try:
                response = requests.get(url, timeout=10)
                rss_feed = response.text

                for item in rss_feed.split("<item>")[1:]:
                    try:
                        link = item.split("<link>")[1].split("</link>")[0]
                        title = item.split("<title>")[1].split("</title>")[0]
                        desc = item.split("<description>")[1].split("</description>")[0]
                        date_str = item.split("<pubDate>")[1].split("</pubDate>")[0]

                        try:
                            date_obj = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
                        except:
                            continue

                        if not filter_on or (start_date <= date_obj <= end_date):
                            news_data.append({
                                'Judul': title,
                                'Deskripsi': desc,
                                'Link': link,
                                'Tanggal Posting': date_obj,
                                'Sumber': 'Google News',
                                'Keyword': keyword
                            })
                    except:
                        continue
            except:
                continue

        return pd.DataFrame(news_data)

    # ============ Fungsi Highlight ============
    def highlight_text(text, keywords):
        for kw in keywords:
            text = re.sub(f"({kw})", r'<span class="keyword-highlight">\1</span>', text, flags=re.IGNORECASE)
        return text

    # ============ Google Trends ============
    def get_google_trends(keywords):
        pytrends = TrendReq(hl='id', tz=360)
        pytrends.build_payload(keywords, cat=0, timeframe='now 7-d', geo='ID', gprop='')
        df_trends = pytrends.interest_over_time().reset_index()
        df_trends.rename(columns={'date': 'Tanggal'}, inplace=True)
        return df_trends

    # ============ Wordcloud ============
    def generate_wordcloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # ============ Main Process ============
    def run_process():
        df = get_object_from_articles(objects_df, start_date, end_date, filter_date_on)

        # Fallback jika kosong dan filter aktif
        if df.empty and filter_date_on:
            st.warning("⚠ Tidak ada berita dalam rentang tanggal. Menampilkan berita terbaru (tanpa filter)...")
            df = get_object_from_articles(objects_df, start_date, end_date, filter_on=False)

        if not df.empty:
            st.success(f"✅ {len(df)} berita ditemukan.")

            # Tampilkan berita
            st.subheader("📰 Hasil Crawling")
            for _, row in df.iterrows():
                title_highlighted = highlight_text(row['Judul'], keywords)
                st.markdown(f"""
                <div class="result-card">
                    <h4>{title_highlighted}</h4>
                    <p>{row['Tanggal Posting'].strftime("%Y-%m-%d %H:%M")} | Keyword: {row['Keyword']}</p>
                    <a href="{row['Link']}" target="_blank">Baca Selengkapnya</a>
                </div>
                """, unsafe_allow_html=True)

            # Dashboard
            st.subheader("📊 Dashboard Analitik")
            col1, col2 = st.columns(2)

            with col1:
                date_count = df['Tanggal Posting'].dt.date.value_counts().reset_index()
                date_count.columns = ['Tanggal', 'Jumlah']
                chart = alt.Chart(date_count).mark_bar().encode(
                    x='Tanggal:T', y='Jumlah:Q', tooltip=['Tanggal', 'Jumlah']
                )
                st.altair_chart(chart, use_container_width=True)

            with col2:
                st.write("Wordcloud Judul Berita")
                generate_wordcloud(" ".join(df['Judul']))

            # Google Trends
            st.subheader("📈 Google Trends Comparison")
            trends_df = get_google_trends(keywords)
            trends_melted = trends_df.melt('Tanggal', var_name='Keyword', value_name='Volume')
            chart_trend = alt.Chart(trends_melted).mark_line(point=True).encode(
                x='Tanggal:T', y='Volume:Q', color='Keyword:N', tooltip=['Tanggal','Keyword','Volume']
            )
            st.altair_chart(chart_trend, use_container_width=True)

            # Download CSV
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="berita_google_news.csv">📥 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

        else:
            st.error("Tidak ada berita ditemukan sama sekali.")

    # Jalankan
    if crawl_btn or auto_refresh:
        while True:
            with st.spinner("Mengambil data dari Google News..."):
                run_process()
            if auto_refresh:
                time.sleep(interval)
                st.rerun()
            else:
                break

# -------------------------------
# chart
# -------------------------------
elif menu == "Dashboard":
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from collections import Counter
    import re
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    import datetime

    # -----------------------------
    # Dummy Data for Demo
    # -----------------------------
    # data = {
    #     'date': pd.date_range(start='2025-01-01', periods=60, freq='D'),
    #     'sentiment': np.random.choice(['positive', 'negative', 'neutral'], 60),
    #     'text': np.random.choice([
    #         'Great service', 'Amazing experience', 'Worst product', 'Average quality',
    #         'I love this', 'Terrible support', 'Highly recommend', 'Not good at all'
    #     ], 60)
    # }
    # df = pd.DataFrame(data)
    import random

    # Generate 200 data dummy
    np.random.seed(42)
    random.seed(42)

    dates = pd.date_range(start='2025-01-01', periods=120).tolist()
    sentiments = ['positive', 'negative', 'neutral']
    texts_positive = [
        "Great product!", "Amazing experience", "Loved it", "Highly recommend", 
        "Excellent quality", "Fantastic service", "Very satisfied", "Will buy again",
        "Smooth transaction", "Best app ever"
    ]
    texts_negative = [
        "Terrible product", "Worst experience", "Not worth the price", "Poor quality",
        "Horrible service", "Very disappointed", "Never again", "App keeps crashing",
        "Late delivery", "Fake reviews"
    ]
    texts_neutral = [
        "It’s okay", "Average experience", "Neutral feeling", "Nothing special",
        "Mediocre service", "Could be better", "No strong opinion", "Just fine",
        "Normal delivery", "Standard product"
    ]

    data = []
    for i in range(200):
        date = random.choice(dates)
        sentiment = random.choice(sentiments)
        if sentiment == 'positive':
            text = random.choice(texts_positive)
        elif sentiment == 'negative':
            text = random.choice(texts_negative)
        else:
            text = random.choice(texts_neutral)
        data.append({'date': date, 'sentiment': sentiment, 'text': text})

    df = pd.DataFrame(data)

    # -----------------------------
    # Streamlit Page Config
    # -----------------------------
    st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
    st.title("📊 Sentiment Analysis Dashboard")
    st.markdown("**Analisis interaktif, insight otomatis, dan export PDF premium dengan branding.**")

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    st.sidebar.header("🔍 Filter Data")
    sentiment_filter = st.sidebar.multiselect("Pilih Sentiment:", options=df['sentiment'].unique(), default=df['sentiment'].unique())
    date_range = st.sidebar.date_input("Rentang Tanggal:", [df['date'].min(), df['date'].max()])

    # Filter Data
    df_filtered = df[(df['sentiment'].isin(sentiment_filter)) &
                    (df['date'] >= pd.to_datetime(date_range[0])) &
                    (df['date'] <= pd.to_datetime(date_range[1]))]

    # -----------------------------
    # Summary Metrics
    # -----------------------------
    st.markdown("### ✅ Ringkasan Data")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", len(df_filtered))
    col2.metric("Positive", int((df_filtered['sentiment'] == 'positive').sum()))
    col3.metric("Negative", int((df_filtered['sentiment'] == 'negative').sum()))
    col4.metric("Neutral", int((df_filtered['sentiment'] == 'neutral').sum()))

    if len(df_filtered) > 0:
        # -----------------------------
        # Visualisasi
        # -----------------------------
        st.subheader("📈 Analisis Visual")
        trend_data = df_filtered.groupby(['date', 'sentiment']).size().reset_index(name='count')
        fig_trend = px.line(trend_data, x='date', y='count', color='sentiment', markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.subheader("📊 Distribusi (Count)")
            dist_count = df_filtered['sentiment'].value_counts().reset_index()
            dist_count.columns = ['sentiment', 'count']
            fig_bar = px.bar(dist_count, x='sentiment', y='count', color='sentiment', text='count')
            st.plotly_chart(fig_bar, use_container_width=True)

        with colB:
            st.subheader("📌 Distribusi (%)")
            dist_count['percentage'] = (dist_count['count'] / dist_count['count'].sum() * 100).round(2)
            fig_pie = px.pie(dist_count, names='sentiment', values='percentage', color='sentiment', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        # -----------------------------
        # WordCloud Split
        # -----------------------------
        st.subheader("☁️ WordCloud per Sentiment")
        col_wc1, col_wc2 = st.columns(2)
        fig_pos, fig_neg = None, None

        with col_wc1:
            st.markdown("**Positive**")
            text_positive = " ".join(df_filtered[df_filtered['sentiment'] == 'positive']['text'].tolist())
            if text_positive.strip():
                wc_positive = WordCloud(width=500, height=300, background_color='white', colormap='Greens').generate(text_positive)
                fig_pos, ax_pos = plt.subplots(figsize=(5, 3))
                ax_pos.imshow(wc_positive, interpolation='bilinear')
                ax_pos.axis("off")
                st.pyplot(fig_pos)
            else:
                st.info("Tidak ada data positif.")

        with col_wc2:
            st.markdown("**Negative**")
            text_negative = " ".join(df_filtered[df_filtered['sentiment'] == 'negative']['text'].tolist())
            if text_negative.strip():
                wc_negative = WordCloud(width=500, height=300, background_color='white', colormap='Reds').generate(text_negative)
                fig_neg, ax_neg = plt.subplots(figsize=(5, 3))
                ax_neg.imshow(wc_negative, interpolation='bilinear')
                ax_neg.axis("off")
                st.pyplot(fig_neg)
            else:
                st.info("Tidak ada data negatif.")

        # -----------------------------
        # Analisis Insight Otomatis + Top Keywords
        # -----------------------------
        def get_top_keywords(df, sentiment, n=5):
            texts = df[df['sentiment'] == sentiment]['text'].str.lower().tolist()
            words = re.findall(r'\b\w+\b', ' '.join(texts))
            return [w for w, _ in Counter(words).most_common(n)]

        top_positive = get_top_keywords(df_filtered, 'positive')
        top_negative = get_top_keywords(df_filtered, 'negative')

        total = len(df_filtered)
        dist = df_filtered['sentiment'].value_counts(normalize=True).mul(100).round(2).to_dict()
        dominant_sentiment = max(dist, key=dist.get)
        first_week = df_filtered[df_filtered['date'] <= df_filtered['date'].min() + pd.Timedelta(days=6)]
        last_week = df_filtered[df_filtered['date'] >= df_filtered['date'].max() - pd.Timedelta(days=6)]
        trend = "meningkat" if last_week['sentiment'].value_counts().get('positive', 0) > first_week['sentiment'].value_counts().get('positive', 0) else "menurun"

        insights = {
            "dominant": dominant_sentiment.capitalize(),
            "trend": trend,
            "top_positive": ", ".join(top_positive),
            "top_negative": ", ".join(top_negative),
            "summary_text": f"Sentimen {dominant_sentiment} mendominasi {dist[dominant_sentiment]}%. Tren positif {trend}. Kata kunci populer positif: {', '.join(top_positive)}; negatif: {', '.join(top_negative)}."
        }

        st.markdown("### 🔍 Insight Otomatis")
        st.info(insights["summary_text"])

        # -----------------------------
        # Export PDF Premium
        # -----------------------------
        st.subheader("📄 Export PDF Premium")

        img_trend = BytesIO(); fig_trend.write_image(img_trend, format="png"); img_trend.seek(0)
        img_bar = BytesIO(); fig_bar.write_image(img_bar, format="png"); img_bar.seek(0)
        img_pie = BytesIO(); fig_pie.write_image(img_pie, format="png"); img_pie.seek(0)

        img_pos = BytesIO()
        if fig_pos: fig_pos.savefig(img_pos, format="png", bbox_inches='tight'); img_pos.seek(0)
        img_neg = BytesIO()
        if fig_neg: fig_neg.savefig(img_neg, format="png", bbox_inches='tight'); img_neg.seek(0)

        # ---- Fungsi PDF Branding Premium ----
        def add_header_footer(c, title, page_num):
            c.setFillColor(HexColor("#004080"))
            c.rect(0, 740, 612, 60, fill=1, stroke=0)
            try:
                logo = ImageReader("logo.png")
                c.drawImage(logo, 20, 750, width=50, height=40, mask='auto')
            except:
                c.setFillColor("white")
                c.drawString(30, 770, "[Logo]")
            c.setFillColor("white")
            c.setFont("Helvetica-Bold", 18)
            c.drawString(80, 765, title)
            c.setFillColor(HexColor("#004080"))
            c.rect(0, 0, 612, 30, fill=1, stroke=0)
            c.setFillColor("white")
            c.setFont("Helvetica", 9)
            c.drawString(20, 15, "© 2025 Your Company - Confidential")
            c.drawRightString(590, 15, f"Halaman {page_num}")

        def generate_premium_pdf(summary, distribution, insights):
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)

            # PAGE 1
            add_header_footer(c, "Sentiment Analysis Report", 1)
            c.translate(0, -40)

            c.setFillColor(HexColor("#004080"))
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 700, "Summary")

            c.setFillColor("black")
            c.setFont("Helvetica", 11)
            c.drawString(60, 680, f"Total Data: {summary['total']}")
            c.drawString(60, 665, f"Positive: {summary['positive']}")
            c.drawString(60, 650, f"Negative: {summary['negative']}")
            c.drawString(60, 635, f"Neutral: {summary['neutral']}")

            # Distribusi
            c.setFillColor(HexColor("#004080"))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, 610, "Distribusi Sentiment:")
            c.setFont("Helvetica", 10)
            c.setFillColor("black")
            y = 595
            for _, row in distribution.iterrows():
                c.drawString(60, y, f"{row['sentiment'].capitalize()}: {row['count']} ({row['percentage']}%)")
                y -= 15

            # Kotak Insight dengan Gradasi
            c.setFillColor(HexColor("#0066CC"))
            c.rect(40, y-90, 530, 80, fill=1, stroke=0)
            c.setFillColor("white")
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y-65, "🔍 Insight Otomatis")
            c.setFont("Helvetica", 10)
            c.drawString(50, y-80, insights['summary_text'])

            # Top Keywords
            c.setFillColor(HexColor("#004080"))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y-110, "Top Keywords:")
            c.setFont("Helvetica", 10)
            c.setFillColor("black")
            c.drawString(60, y-125, f"Positive: {insights['top_positive']}")
            c.drawString(60, y-140, f"Negative: {insights['top_negative']}")

            c.showPage()

            # PAGE 2 (Grafik)
            add_header_footer(c, "Visualisasi Sentiment", 2)
            c.translate(0, -40)
            c.setFont("Helvetica-Bold", 14)
            c.setFillColor(HexColor("#004080"))
            c.drawString(50, 700, "Grafik & WordCloud")
            c.drawImage(ImageReader(img_trend), 40, 450, width=250, height=150)
            c.drawImage(ImageReader(img_bar), 300, 450, width=250, height=150)
            c.drawImage(ImageReader(img_pie), 40, 260, width=250, height=150)
            if fig_pos: c.drawImage(ImageReader(img_pos), 300, 260, width=250, height=150)
            if fig_neg: c.drawImage(ImageReader(img_neg), 40, 70, width=250, height=150)

            c.showPage()
            c.save()
            return buffer.getvalue()

        summary_info = {'total': total, 'positive': int((df_filtered['sentiment'] == 'positive').sum()), 'negative': int((df_filtered['sentiment'] == 'negative').sum()), 'neutral': int((df_filtered['sentiment'] == 'neutral').sum())}
        pdf_data = generate_premium_pdf(summary_info, dist_count, insights)

        st.download_button("⬇️ Download PDF Premium", data=pdf_data, file_name="sentiment_report_premium.pdf", mime="application/pdf")

    else:
        st.warning("Tidak ada data sesuai filter.")
