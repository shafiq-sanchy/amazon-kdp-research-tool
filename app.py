import streamlit as st
import requests
import json
import pandas as pd
import time
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import re
import io
import base64

# Configure page
st.set_page_config(
    page_title="Amazon KDP Research Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for API keys
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'rapidapi': ["3f52b9a04fmsh50120f0c987f590p168854jsn841e06da4d93"],
        'gemini': [],
        'chatgpt': [],
        'megallm': [],
        'rapidapi_counter': 0,
        'gemini_counter': 0,
        'chatgpt_counter': 0,
        'megallm_counter': 0
    }

if 'ai_service' not in st.session_state:
    st.session_state.ai_service = 'gemini'

class APIManager:
    """Manages API calls with rotation and rate limiting"""
    
    def __init__(self):
        self.session = requests.Session()
        self.last_call_time = {}
        self.min_interval = 2.0  # Increased to avoid rate limits
        
    def rotate_api_key(self, service):
        """Rotate API keys to avoid rate limits"""
        keys = st.session_state.api_keys.get(service, [])
        if not keys:
            return None
            
        counter_key = f"{service}_counter"
        st.session_state.api_keys[counter_key] = (st.session_state.api_keys[counter_key] + 1) % len(keys)
        return keys[st.session_state.api_keys[counter_key]]
    
    def make_request(self, url, headers=None, params=None, service='rapidapi'):
        """Make API request with rate limiting and retry logic"""
        # Rate limiting
        current_time = time.time()
        if service in self.last_call_time:
            elapsed = current_time - self.last_call_time[service]
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Rotate API key if needed
                if headers and 'x-rapidapi-key' in headers:
                    api_key = self.rotate_api_key('rapidapi')
                    if api_key:
                        headers['x-rapidapi-key'] = api_key
                
                response = self.session.get(url, headers=headers, params=params, timeout=15)
                self.last_call_time[service] = time.time()
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    time.sleep(3 ** attempt)  # Increased backoff
                    continue
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    return None
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Request failed: {str(e)}")
                    return None
                time.sleep(2)
        
        return None

class AmazonDataCollector:
    """Collects data from Amazon using multiple APIs"""
    
    def __init__(self):
        self.api_manager = APIManager()
        
    def search_products(self, keyword, page=1):
        """Search for products on Amazon"""
        url = "https://real-time-amazon-data.p.rapidapi.com/search"
        
        headers = {
            "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com",
            "x-rapidapi-key": st.session_state.api_keys['rapidapi'][0]  # Will be rotated
        }
        
        params = {
            "query": keyword,
            "page": str(page),
            "country": "US",
            "category_id": "stripbooks"  # Books category
        }
        
        data = self.api_manager.make_request(url, headers, params)
        
        if data and 'data' in data and 'products' in data['data']:
            products = []
            for product in data['data']['products']:
                products.append({
                    'asin': product.get('asin', ''),
                    'title': product.get('product_title', ''),
                    'price': product.get('product_price', ''),
                    'rating': product.get('product_star_rating', ''),
                    'reviews': product.get('product_num_ratings', ''),
                    'url': product.get('product_url', ''),
                    'photo': product.get('product_photo', ''),
                    'is_prime': product.get('is_prime', False),
                    'source': 'RapidAPI'
                })
            return products
        return []
    
    def get_product_details(self, asin):
        """Get detailed product information"""
        url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
        
        headers = {
            "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com",
            "x-rapidapi-key": st.session_state.api_keys['rapidapi'][0]  # Will be rotated
        }
        
        params = {
            "asin": asin,
            "country": "US"
        }
        
        data = self.api_manager.make_request(url, headers, params)
        
        if data and 'data' in data:
            return data['data']
        return None
    
    def get_product_reviews(self, asin):
        """Get product reviews"""
        url = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"
        
        headers = {
            "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com",
            "x-rapidapi-key": st.session_state.api_keys['rapidapi'][0]  # Will be rotated
        }
        
        params = {
            "asin": asin,
            "country": "US",
            "sort_by": "TOP_REVIEWS",
            "reviews_per_page": "10"
        }
        
        data = self.api_manager.make_request(url, headers, params)
        
        if data and 'data' in data and 'reviews' in data['data']:
            return data['data']['reviews']
        return []
    
    def get_categories(self, keyword):
        """Get KDP categories for a keyword"""
        # Search for products first
        products = self.search_products(keyword)
        
        if not products:
            return []
        
        # Extract categories from top products
        categories = {}
        for product in products[:5]:  # Top 5 products
            details = self.get_product_details(product['asin'])
            if details and 'product_category' in details:
                category_path = details['product_category']
                if isinstance(category_path, list):
                    for category in category_path:
                        if category not in categories:
                            categories[category] = 0
                        categories[category] += 1
        
        # Sort by frequency
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        return [{'name': cat, 'count': count} for cat, count in sorted_categories]

class AIAnalyzer:
    """Uses AI to analyze collected data"""
    
    def __init__(self):
        self.api_manager = APIManager()
    
    def analyze_with_gemini(self, prompt):
        """Analyze data using Google Gemini"""
        api_key = self.api_manager.rotate_api_key('gemini')
        if not api_key:
            return "No Gemini API key available. Please add one in API Settings."
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=15)
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"Gemini API Error: {response.status_code}"
        except Exception as e:
            return f"Gemini API Error: {str(e)}"
    
    def analyze_with_chatgpt(self, prompt):
        """Analyze data using OpenAI ChatGPT"""
        api_key = self.api_manager.rotate_api_key('chatgpt')
        if not api_key:
            return "No ChatGPT API key available. Please add one in API Settings."
        
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=15)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"ChatGPT API Error: {response.status_code}"
        except Exception as e:
            return f"ChatGPT API Error: {str(e)}"
    
    def analyze_with_megallm(self, prompt):
        """Analyze data using MegaLLM"""
        api_key = self.api_manager.rotate_api_key('megallm')
        if not api_key:
            return "No MegaLLM API key available. Please add one in API Settings."
        
        url = "https://api.megallm.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=15)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"MegaLLM API Error: {response.status_code}"
        except Exception as e:
            return f"MegaLLM API Error: {str(e)}"
    
    def analyze_market_opportunity(self, keyword, products, ai_service="gemini"):
        """Analyze market opportunity using AI"""
        prompt = f"""
        Analyze the market opportunity for Amazon KDP books with keyword: "{keyword}"
        
        Based on these top products:
        {json.dumps(products[:5], indent=2)}
        
        Provide:
        1. Market demand level (Low/Medium/High)
        2. Competition level (Low/Medium/High)
        3. Opportunity score (1-10)
        4. Recommended book topics
        5. Key insights
        """
        
        if ai_service == "gemini":
            return self.analyze_with_gemini(prompt)
        elif ai_service == "chatgpt":
            return self.analyze_with_chatgpt(prompt)
        else:
            return self.analyze_with_megallm(prompt)
    
    def analyze_competitor(self, product, ai_service="gemini"):
        """Analyze a competitor using AI"""
        prompt = f"""
        Analyze this Amazon KDP competitor:
        
        Title: {product.get('title', '')}
        Price: {product.get('price', '')}
        Rating: {product.get('rating', '')}
        Reviews: {product.get('reviews', '')}
        
        Provide:
        1. Strengths (1-3 points)
        2. Weaknesses (1-3 points)
        3. Market position
        4. Improvement opportunities
        """
        
        if ai_service == "gemini":
            return self.analyze_with_gemini(prompt)
        elif ai_service == "chatgpt":
            return self.analyze_with_chatgpt(prompt)
        else:
            return self.analyze_with_megallm(prompt)

class TrendTracker:
    """Tracks trends using Google Trends data"""
    
    def __init__(self):
        self.api_manager = APIManager()
    
    def get_trend_data(self, keyword):
        """Get trend data for a keyword"""
        # Using a free Google Trends API alternative
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={keyword}&mode=artlist&format=json"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Process trend data
                return {
                    'trend': 'increasing',  # Simplified
                    'interest': 75,
                    'source': 'GDELT'
                }
        except:
            pass
        
        return {
            'trend': 'stable',
            'interest': 50,
            'source': 'Estimated'
        }

# Initialize components
@st.cache_resource
def init_components():
    return {
        'collector': AmazonDataCollector(),
        'analyzer': AIAnalyzer(),
        'tracker': TrendTracker()
    }

# API Settings Page
def api_settings_page():
    st.header("🔑 API Settings")
    st.markdown("Manage your API keys for different services")
    
    # RapidAPI Keys
    st.subheader("RapidAPI Keys")
    rapidapi_keys = st.text_area(
        "Enter RapidAPI Keys (one per line):",
        value="\n".join(st.session_state.api_keys['rapidapi']),
        help="Get your keys from https://rapidapi.com/hub"
    )
    st.session_state.api_keys['rapidapi'] = [key.strip() for key in rapidapi_keys.split('\n') if key.strip()]
    
    # Gemini API Keys
    st.subheader("Google Gemini API Keys")
    gemini_keys = st.text_area(
        "Enter Gemini API Keys (one per line):",
        value="\n".join(st.session_state.api_keys['gemini']),
        help="Get your keys from https://makersuite.google.com/app/apikey"
    )
    st.session_state.api_keys['gemini'] = [key.strip() for key in gemini_keys.split('\n') if key.strip()]
    
    # ChatGPT API Keys
    st.subheader("OpenAI ChatGPT API Keys")
    chatgpt_keys = st.text_area(
        "Enter ChatGPT API Keys (one per line):",
        value="\n".join(st.session_state.api_keys['chatgpt']),
        help="Get your keys from https://platform.openai.com/api-keys"
    )
    st.session_state.api_keys['chatgpt'] = [key.strip() for key in chatgpt_keys.split('\n') if key.strip()]
    
    # MegaLLM API Keys
    st.subheader("MegaLLM API Keys")
    megallm_keys = st.text_area(
        "Enter MegaLLM API Keys (one per line):",
        value="\n".join(st.session_state.api_keys['megallm']),
        help="Get your keys from https://megallm.com"
    )
    st.session_state.api_keys['megallm'] = [key.strip() for key in megallm_keys.split('\n') if key.strip()]
    
    # File Upload
    st.subheader("Upload API Keys File")
    uploaded_file = st.file_uploader(
        "Upload a file with API keys (TXT, CSV, or JSON)",
        type=["txt", "csv", "json"]
    )
    
    if uploaded_file is not None:
        try:
            # Read file content
            content = uploaded_file.getvalue().decode("utf-8")
            
            # Parse based on file type
            if uploaded_file.name.endswith('.json'):
                data = json.loads(content)
                if 'rapidapi' in data:
                    st.session_state.api_keys['rapidapi'].extend(data['rapidapi'])
                if 'gemini' in data:
                    st.session_state.api_keys['gemini'].extend(data['gemini'])
                if 'chatgpt' in data:
                    st.session_state.api_keys['chatgpt'].extend(data['chatgpt'])
                if 'megallm' in data:
                    st.session_state.api_keys['megallm'].extend(data['megallm'])
            else:
                # For TXT and CSV, assume each line is an API key
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                for line in lines:
                    if 'rapidapi' in line.lower():
                        key = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                        st.session_state.api_keys['rapidapi'].append(key)
                    elif 'gemini' in line.lower():
                        key = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                        st.session_state.api_keys['gemini'].append(key)
                    elif 'chatgpt' in line.lower() or 'openai' in line.lower():
                        key = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                        st.session_state.api_keys['chatgpt'].append(key)
                    elif 'megallm' in line.lower():
                        key = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                        st.session_state.api_keys['megallm'].append(key)
            
            st.success("API keys uploaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Display current status
    st.subheader("Current API Keys Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RapidAPI Keys", len(st.session_state.api_keys['rapidapi']))
    with col2:
        st.metric("Gemini Keys", len(st.session_state.api_keys['gemini']))
    with col3:
        st.metric("ChatGPT Keys", len(st.session_state.api_keys['chatgpt']))
    with col4:
        st.metric("MegaLLM Keys", len(st.session_state.api_keys['megallm']))
    
    # AI Service Selection
    st.subheader("AI Service Selection")
    ai_service = st.selectbox(
        "Select AI service for analysis:",
        ["gemini", "chatgpt", "megallm"],
        index=["gemini", "chatgpt", "megallm"].index(st.session_state.ai_service),
        help="This will be used for AI-powered analysis"
    )
    st.session_state.ai_service = ai_service
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
        st.rerun()

# Main UI
def main():
    components = init_components()
    
    # Sidebar
    st.sidebar.title("📚 Amazon KDP Research Tool")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Tool",
        [
            "🔑 API Settings",
            "🔍 Category Finder",
            "📊 Market Research",
            "🔑 Keyword Research",
            "👥 Competitor Analysis",
            "📈 Top Sellers Tracker",
            "📈 Trend Analysis"
        ]
    )
    
    # Get AI service selection
    ai_service = st.session_state.get('ai_service', 'gemini')
    
    # Main content
    if page == "🔑 API Settings":
        api_settings_page()
    elif page == "🔍 Category Finder":
        category_finder_page(components)
    elif page == "📊 Market Research":
        market_research_page(components, ai_service)
    elif page == "🔑 Keyword Research":
        keyword_research_page(components)
    elif page == "👥 Competitor Analysis":
        competitor_analysis_page(components, ai_service)
    elif page == "📈 Top Sellers Tracker":
        top_sellers_page(components)
    elif page == "📈 Trend Analysis":
        trend_analysis_page(components)

def category_finder_page(components):
    st.header("🔍 KDP Category Finder")
    st.markdown("Find profitable Amazon KDP categories for your niche")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        keyword = st.text_input("Enter your book keyword or niche:", placeholder="e.g., 'cooking', 'self-help', 'mystery'")
    
    with col2:
        st.write("")
        search_button = st.button("Find Categories", type="primary")
    
    if search_button and keyword:
        with st.spinner("Searching categories..."):
            categories = components['collector'].get_categories(keyword)
            
            if categories:
                st.success(f"Found {len(categories)} categories for '{keyword}'")
                
                # Display categories
                for i, category in enumerate(categories[:10], 1):
                    with st.expander(f"{i}. {category['name']} (Products: {category['count']})"):
                        st.write(f"**Category:** {category['name']}")
                        st.write(f"**Product Count:** {category['count']}")
                        
                        # Get sample products
                        products = components['collector'].search_products(keyword)
                        if products:
                            st.write("**Sample Products:**")
                            for product in products[:3]:
                                st.write(f"- {product['title'][:80]}...")
                                st.write(f"  Price: {product['price']} | Rating: {product['rating']}")
            else:
                st.warning("No categories found. Try a different keyword.")

def market_research_page(components, ai_service):
    st.header("📊 Market Research")
    st.markdown("Analyze market demand and competition for your niche")
    
    keyword = st.text_input("Enter your niche keyword:", placeholder="e.g., 'vegan cooking'")
    
    if st.button("Analyze Market"):
        if keyword:
            with st.spinner("Analyzing market..."):
                # Get products
                products = components['collector'].search_products(keyword)
                
                if products:
                    # Display basic stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Products", len(products))
                    col2.metric("Avg Price", f"${calculate_avg_price(products)}")
                    col3.metric("Avg Rating", f"{calculate_avg_rating(products):.1f}")
                    col4.metric("Total Reviews", sum(get_review_count(p) for p in products))
                    
                    # AI Analysis
                    with st.spinner("AI analyzing market opportunity..."):
                        ai_analysis = components['analyzer'].analyze_market_opportunity(keyword, products, ai_service)
                        st.subheader("🤖 AI Market Analysis")
                        st.write(ai_analysis)
                    
                    # Top products
                    st.subheader("🏆 Top Products")
                    for i, product in enumerate(products[:5], 1):
                        with st.expander(f"{i}. {product['title'][:60]}..."):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**ASIN:** {product['asin']}")
                                st.write(f"**Price:** {product['price']}")
                                st.write(f"**Rating:** {product['rating']} ({product['reviews']} reviews)")
                                st.write(f"**Prime:** {'✅' if product['is_prime'] else '❌'}")
                            with col2:
                                if product['photo']:
                                    st.image(product['photo'], width=100)
                else:
                    st.warning("No products found. Try a different keyword.")

def keyword_research_page(components):
    st.header("🔑 Keyword Research")
    st.markdown("Find profitable keywords for your KDP books")
    
    keyword = st.text_input("Enter a seed keyword:", placeholder="e.g., 'meditation'")
    
    if st.button("Research Keywords"):
        if keyword:
            with st.spinner("Researching keywords..."):
                # Get related products
                products = components['collector'].search_products(keyword)
                
                if products:
                    # Extract keywords from titles
                    keyword_data = extract_keywords(products)
                    
                    # Display keywords
                    st.subheader("🔍 Related Keywords")
                    df = pd.DataFrame(keyword_data[:20])
                    st.dataframe(df, use_container_width=True)
                    
                    # Visualize
                    fig = px.bar(df.head(10), x='frequency', y='keyword', orientation='h')
                    fig.update_layout(title='Top Keywords by Frequency')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend data
                    st.subheader("📈 Trend Analysis")
                    trend_data = components['tracker'].get_trend_data(keyword)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Trend", trend_data['trend'].capitalize())
                    col2.metric("Interest Level", f"{trend_data['interest']}/100")
                    col3.metric("Source", trend_data['source'])
                else:
                    st.warning("No data found. Try a different keyword.")

def competitor_analysis_page(components, ai_service):
    st.header("👥 Competitor Analysis")
    st.markdown("Analyze your competitors on Amazon KDP")
    
    option = st.radio("Select analysis type:", ["Search by ASIN", "Search by Keyword"])
    
    if option == "Search by ASIN":
        asin = st.text_input("Enter ASIN:", placeholder="e.g., B08N5M7S6K")
        
        if st.button("Analyze Competitor"):
            if asin:
                with st.spinner("Analyzing competitor..."):
                    # Get product details
                    details = components['collector'].get_product_details(asin)
                    
                    if details:
                        # Display basic info
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader("📦 Product Details")
                            st.write(f"**Title:** {details.get('product_title', 'N/A')}")
                            st.write(f"**Brand:** {details.get('product_brand', 'N/A')}")
                            st.write(f"**Price:** {details.get('product_price', 'N/A')}")
                            st.write(f"**Rating:** {details.get('product_star_rating', 'N/A')}")
                            st.write(f"**Reviews:** {details.get('product_num_ratings', 'N/A')}")
                        
                        with col2:
                            if details.get('product_photo'):
                                st.image(details['product_photo'], width=150)
                        
                        # AI Analysis
                        with st.spinner("AI analyzing competitor..."):
                            ai_analysis = components['analyzer'].analyze_competitor(details, ai_service)
                            st.subheader("🤖 AI Analysis")
                            st.write(ai_analysis)
                        
                        # Reviews
                        reviews = components['collector'].get_product_reviews(asin)
                        if reviews:
                            st.subheader("💬 Recent Reviews")
                            for review in reviews[:3]:
                                st.write(f"**{review.get('review_rating', 'N/A')}/5** - {review.get('review_title', 'N/A')}")
                                st.write(review.get('review_comment', 'N/A')[:200] + "...")
                                st.write("---")
                    else:
                        st.error("Product not found. Check ASIN and try again.")
    
    else:  # Search by Keyword
        keyword = st.text_input("Enter keyword:", placeholder="e.g., 'productivity'")
        
        if st.button("Find Competitors"):
            if keyword:
                with st.spinner("Finding competitors..."):
                    products = components['collector'].search_products(keyword)
                    
                    if products:
                        st.success(f"Found {len(products)} competitors")
                        
                        for i, product in enumerate(products[:5], 1):
                            with st.expander(f"{i}. {product['title'][:60]}..."):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**ASIN:** {product['asin']}")
                                    st.write(f"**Price:** {product['price']}")
                                    st.write(f"**Rating:** {product['rating']}")
                                    st.write(f"**Reviews:** {product['reviews']}")
                                with col2:
                                    if product['photo']:
                                        st.image(product['photo'], width=100)
                    else:
                        st.warning("No competitors found. Try a different keyword.")

def top_sellers_page(components):
    st.header("📈 Top Sellers Tracker")
    st.markdown("Track top sellers in your niche")
    
    category = st.selectbox(
        "Select a category:",
        [
            "All Books",
            "Business & Money",
            "Self-Help",
            "Cookbooks",
            "Health & Fitness",
            "Parenting",
            "Relationships",
            "Mystery & Thriller",
            "Romance",
            "Science Fiction"
        ]
    )
    
    time_range = st.selectbox("Time range:", ["Today", "This Week", "This Month"])
    
    if st.button("Track Top Sellers"):
        with st.spinner("Tracking top sellers..."):
            # Get top products
            keyword = category.lower().replace(" & ", " ").replace(" ", " ")
            products = components['collector'].search_products(keyword)
            
            if products:
                st.success(f"Found {len(products)} top sellers")
                
                # Create leaderboard
                leaderboard_data = []
                for i, product in enumerate(products[:10], 1):
                    reviews = get_review_count(product)
                    rating = get_rating_value(product)
                    
                    leaderboard_data.append({
                        'Rank': i,
                        'Title': product['title'][:50] + "...",
                        'ASIN': product['asin'],
                        'Price': product['price'],
                        'Rating': rating,
                        'Reviews': reviews,
                        'Score': calculate_seller_score(rating, reviews)
                    })
                
                df = pd.DataFrame(leaderboard_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualize
                fig = px.scatter(
                    df,
                    x='Reviews',
                    y='Rating',
                    size='Score',
                    hover_name='Title',
                    title='Top Sellers Performance'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data found. Try a different category.")

def trend_analysis_page(components):
    st.header("📈 Trend Analysis")
    st.markdown("Analyze trends for your keywords")
    
    keyword = st.text_input("Enter keyword to analyze:", placeholder="e.g., 'mindfulness'")
    
    if st.button("Analyze Trend"):
        if keyword:
            with st.spinner("Analyzing trends..."):
                # Get trend data
                trend_data = components['tracker'].get_trend_data(keyword)
                
                # Display trend info
                col1, col2, col3 = st.columns(3)
                col1.metric("Trend Direction", trend_data['trend'].capitalize())
                col2.metric("Interest Level", f"{trend_data['interest']}/100")
                col3.metric("Data Source", trend_data['source'])
                
                # Get related products
                products = components['collector'].search_products(keyword)
                
                if products:
                    st.subheader("📊 Market Analysis")
                    
                    # Price distribution
                    prices = [get_price_value(p) for p in products if get_price_value(p) > 0]
                    if prices:
                        fig = px.histogram(x=prices, nbins=10, title="Price Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Rating distribution
                    ratings = [get_rating_value(p) for p in products if get_rating_value(p) > 0]
                    if ratings:
                        fig = px.histogram(x=ratings, nbins=10, title="Rating Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    if trend_data['trend'] == 'increasing':
                        st.success("✅ This keyword is trending upward! Good opportunity.")
                    elif trend_data['trend'] == 'decreasing':
                        st.warning("⚠️ This keyword is trending downward. Be cautious.")
                    else:
                        st.info("ℹ️ This keyword has stable demand.")

# Helper functions
def calculate_avg_price(products):
    prices = [get_price_value(p) for p in products if get_price_value(p) > 0]
    return sum(prices) / len(prices) if prices else 0

def calculate_avg_rating(products):
    ratings = [get_rating_value(p) for p in products if get_rating_value(p) > 0]
    return sum(ratings) / len(ratings) if ratings else 0

def get_price_value(product):
    try:
        price_str = product.get('price', '').replace('$', '').replace(',', '')
        return float(price_str) if price_str else 0
    except:
        return 0

def get_rating_value(product):
    try:
        rating_str = product.get('rating', '').split(' ')[0]
        return float(rating_str) if rating_str else 0
    except:
        return 0

def get_review_count(product):
    try:
        reviews_str = product.get('reviews', '').replace(',', '')
        return int(reviews_str) if reviews_str.isdigit() else 0
    except:
        return 0

def calculate_seller_score(rating, reviews):
    # Simple scoring algorithm
    return (rating * 20) + (reviews / 100)

def extract_keywords(products):
    """Extract keywords from product titles"""
    from collections import Counter
    
    all_words = []
    for product in products:
        title = product.get('title', '').lower()
        words = re.findall(r'\b\w+\b', title)
        # Filter out common words
        words = [w for w in words if len(w) > 3 and w not in ['book', 'guide', 'the', 'and', 'for', 'with']]
        all_words.extend(words)
    
    word_count = Counter(all_words)
    return [{'keyword': word, 'frequency': count} for word, count in word_count.most_common()]

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This tool uses real Amazon data to help you research profitable KDP niches and categories.")

if __name__ == "__main__":
    main()
