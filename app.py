import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import plotly.express as px
from bs4 import BeautifulSoup

# --- Utility functions ---
def get_price_value(product):
    try:
        return float(str(product.get('price', '')).replace('$', '').strip())
    except:
        return 0.0

def get_rating_value(product):
    try:
        rating_str = str(product.get('rating', '')).split()[0]
        return float(rating_str)
    except:
        return 0.0

def get_review_count(product):
    try:
        return int(str(product.get('reviews', '0')).replace(',', '').strip())
    except:
        return 0

def estimate_monthly_sales(bsr):
    # Very rough estimate based on Best Seller Rank
    if bsr < 1000:
        return random.randint(3000, 5000)
    elif bsr < 5000:
        return random.randint(1000, 3000)
    elif bsr < 20000:
        return random.randint(300, 1000)
    else:
        return random.randint(50, 300)

# --- Import your classes from earlier code ---
# (AmazonDataCollector, AIAnalyzer, TrendTracker, etc.)
# For brevity, I assume you keep those definitions exactly as in your file above.

# Initialize components
@st.cache_resource
def init_components():
    from importlib import reload
    return {
        'collector': AmazonDataCollector(),
        'analyzer': AIAnalyzer(),
        'tracker': TrendTracker()
    }

# --- Pages ---
def keyword_research_page(components):
    st.markdown('<div class="main-header">🔑 Keyword Research</div>', unsafe_allow_html=True)
    st.markdown("Discover profitable keywords for your Amazon KDP books")

    keyword = st.text_input("Enter a keyword idea:", placeholder="e.g., gratitude journal, fitness planner")
    if st.button("Find Keywords"):
        if keyword:
            with st.spinner("Analyzing keywords..."):
                products = components['collector'].search_products(keyword)
                time.sleep(0.5)
                if products:
                    st.success(f"Found {len(products)} products for keyword '{keyword}'")
                    df = pd.DataFrame(products)
                    st.dataframe(df[['title','price','rating','reviews']])
                else:
                    st.warning("No products found. Try another keyword.")

def competitor_analysis_page(components, ai_service):
    st.markdown('<div class="main-header">👥 Competitor Analysis</div>', unsafe_allow_html=True)
    asin = st.text_input("Enter competitor ASIN:")
    if st.button("Analyze Competitor"):
        if asin:
            product = components['collector'].get_product_details(asin)
            st.write("### Product Details")
            st.json(product)
            st.write("### AI Competitor Analysis")
            analysis = components['analyzer'].analyze_competitor(product, ai_service)
            st.markdown(analysis)

def trend_analysis_page(components):
    st.markdown('<div class="main-header">📈 Trend Analysis</div>', unsafe_allow_html=True)
    keyword = st.text_input("Enter keyword for trend analysis:")
    if st.button("Analyze Trend"):
        if keyword:
            trend_data = components['tracker'].get_trend_data(keyword)
            st.json(trend_data)
            df = components['tracker'].generate_trend_chart_data(keyword)
            fig = px.line(df, x="date", y="interest", title=f"Trend for {keyword}")
            st.plotly_chart(fig, use_container_width=True)

# --- Main App ---
def main():
    st.sidebar.title("📚 Amazon KDP Research Tool")
    page = st.sidebar.radio("Navigate", [
        "Category Finder",
        "Market Research",
        "Keyword Research",
        "Competitor Analysis",
        "Trend Analysis",
        "API Settings"
    ])

    components = init_components()
    ai_service = st.session_state.get('ai_service', 'gemini')

    if page == "Category Finder":
        category_finder_page(components)
    elif page == "Market Research":
        market_research_page(components, ai_service)
    elif page == "Keyword Research":
        keyword_research_page(components)
    elif page == "Competitor Analysis":
        competitor_analysis_page(components, ai_service)
    elif page == "Trend Analysis":
        trend_analysis_page(components)
    elif page == "API Settings":
        api_settings_page()

if __name__ == "__main__":
    main()
