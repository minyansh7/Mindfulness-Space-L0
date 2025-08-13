# Required libraries
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json

def get_time_colors(hour):
    if 5 <= hour < 8:
        return {'primary': '#FF6B6B', 'secondary': '#FFB347', 'bg_gradient': 'linear-gradient(135deg, #FF6B6B 0%, #FFB347 100%)', 'ripple_color': 'rgba(255, 107, 107, 0.3)', 'text_color': 'black'}
    elif 8 <= hour < 12:
        return {'primary': '#4ECDC4', 'secondary': '#45B7D1', 'bg_gradient': 'linear-gradient(135deg, #4ECDC4 0%, #45B7D1 100%)', 'ripple_color': 'rgba(78, 205, 196, 0.3)', 'text_color': 'black'}
    elif 12 <= hour < 17:
        return {'primary': '#45B7D1', 'secondary': '#96CEB4', 'bg_gradient': 'linear-gradient(135deg, #45B7D1 0%, #96CEB4 100%)', 'ripple_color': 'rgba(69, 183, 209, 0.3)', 'text_color': 'black'}
    elif 17 <= hour < 20:
        return {'primary': '#A29BFE', 'secondary': '#FD79A8', 'bg_gradient': 'linear-gradient(135deg, #A29BFE 0%, #FD79A8 100%)', 'ripple_color': 'rgba(162, 155, 254, 0.3)', 'text_color': 'white'}
    else:
        return {'primary': '#6C5CE7', 'secondary': '#2D3436', 'bg_gradient': 'linear-gradient(135deg, #6C5CE7 0%, #2D3436 100%)', 'ripple_color': 'rgba(108, 92, 231, 0.3)', 'text_color': 'white'}

def run():
    st.set_page_config(page_title="🌦️ Meditation Weather Map", layout="wide")
    colors = get_time_colors(datetime.datetime.now().hour)

    # Enhanced Global Styling
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}
    .main > div {{
        padding-top: 1rem;
    }}
    
    /* SIDEBAR WITH DEFAULT STREAMLIT STYLING */
    .stSidebar {{
        background: #f0f2f6 !important;
        border-right: 1px solid rgba(49, 51, 63, 0.2) !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
    }}
    
    .stSidebar > div {{
        background: transparent !important;
        border: none !important;
    }}
    
    /* SIDEBAR TITLE WITH DEFAULT DARK COLOR */
    .sidebar-title {{
        color: #262730 !important;
        font-size: 24px !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 5px !important;
        background: none !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: unset !important;
        background-clip: unset !important;
        text-shadow: none !important;
    }}
    
    /* ENHANCED TIME NAVIGATION HEADER - COMPACT VERSION */
    .stSidebar .stRadio > label > div[data-testid="stMarkdownContainer"] p {{
        font-size: 12px !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 6px !important;
        letter-spacing: -0.5px !important;
        background: linear-gradient(45deg, #ffd700, #ff6b6b) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    
    /* FALLBACK FOR RADIO LABEL STYLING - COMPACT */
    .stSidebar .stRadio > label {{
        font-size: 12px !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 6px !important;
        letter-spacing: -0.5px !important;
        background: linear-gradient(45deg, #ffd700, #ff6b6b) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-family: 'DM Sans', sans-serif !important;
        display: block !important;
    }}
    
    /* RESPONSIVE QUARTER NAVIGATION - ULTRA COMPACT 80% WIDTH */
    .stSidebar .stRadio {{
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 8px 10px 15px 10px !important;
        text-align: center;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.08) !important;
        animation: slideInLeft 0.8s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        width: 100% !important;
        box-sizing: border-box !important;
        margin: 8px 0 !important;
        min-width: 240px;
        flex: 1;
        max-width: 280px;
        background: rgba(255,255,255,0.05) !important;
    }}
    
    .stSidebar .stRadio:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    
    .stSidebar .stRadio label {{
        color: #262730 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: clamp(10px, 2.5vw, 12px) !important;
        font-weight: 400 !important;
        margin-bottom: 8px !important;
        display: block !important;
    }}
    
    /* OVERRIDE FOR RADIO BUTTON INDIVIDUAL LABELS (NOT THE MAIN HEADER) */
    .stSidebar .stRadio > div > label {{
        background: rgba(0, 0, 0, 0.1) !important;
        border: 1.5px solid rgba(0, 0, 0, 0.2) !important;
        width: clamp(14px, 4vw, 18px) !important;
        height: clamp(14px, 4vw, 18px) !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        position: relative !important;
        z-index: 2 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        flex: none !important;
        flex-shrink: 0 !important;
        min-width: clamp(14px, 18vw, 18px) !important;
        /* RESET GRADIENT FOR BUTTON DOTS */
        background-image: none !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: unset !important;
        background-clip: unset !important;
        color: transparent !important;
        font-size: 0 !important;
    }}
    
    /* RESPONSIVE CONTAINER FOR RADIO BUTTONS - ULTRA COMPACT */
    .stSidebar .stRadio > div {{
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        align-items: center !important;
        gap: 0 !important;
        position: relative !important;
        padding-bottom: 12px !important;
        padding: 8px 2px 6px 2px !important;
        width: 100% !important;
        margin: 0 auto !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        min-height: 24px !important;
    }}
    
    /* RESPONSIVE BACKGROUND LINE - COMPACT */
    .stSidebar .stRadio > div::before {{
        content: "" !important;
        position: absolute !important;
        top: 52% !important;
        left: 3% !important;
        right: 3% !important;
        height: 3px !important;
        background: linear-gradient(135deg, rgba(160, 160, 160, 0.3), rgba(120, 120, 120, 0.2)) !important;
        border: 1px solid rgba(140, 140, 140, 0.25) !important;
        border-radius: 8px !important;
        box-shadow: 
            0 1px 2px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(180, 180, 180, 0.2) !important;
        z-index: 1 !important;
        transform: translateY(-50%) !important;
    }}
    
    /* RESPONSIVE START LABEL - COMPACT */
    .stSidebar .stRadio::before {{
        content: "2024Q1" !important;
        position: absolute !important;
        bottom: 3px !important;
        left: 8px !important;
        color: #262730 !important;   
        font-family: 'DM Sans', sans-serif !important;
        font-size: 7px !important;
        font-weight: 400 !important;
        z-index: 10 !important;
    }}
    
    /* RESPONSIVE END LABEL - COMPACT */
    .stSidebar .stRadio::after {{
        content: "2025Q2" !important;
        position: absolute !important;
        bottom: 3px !important;
        right: 8px !important;
        color: #262730 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 7px !important;
        font-weight: 400 !important;
        z-index: 10 !important;
    }}
    
    
    /* RESPONSIVE RADIO BUTTON HOVER STATES */
    .stSidebar .stRadio > div > label:hover {{
        transform: scale(1.05) !important;
        background: rgba(0, 0, 0, 0.15) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
        border: 1.5px solid rgba(0, 0, 0, 0.3) !important;
    }}
    
    .stSidebar .stRadio > div > label > div {{
        display: none !important;
    }}
    
    .stSidebar .stRadio > div > label input {{
        display: none !important;
    }}
    
    /* RESPONSIVE SELECTED STATE - COMPACT */
    .stSidebar .stRadio > div > label[data-checked="true"] {{
        background: rgba(0, 0, 0, 0.6) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.6) !important;
        border: 1.5px solid rgba(0, 0, 0, 0.4) !important;
        transform: scale(1.3) !important;
    }}
    
    /* RESPONSIVE CURRENT QUARTER LABEL - COMPACT */
    .stSidebar .stRadio > div > label[data-checked="true"]::after {{
        content: attr(data-quarter) !important;
        position: absolute !important;
        top: 120% !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        color: #262730 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 7px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 1px 4px !important;
        border-radius: 4px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08) !important;
    }}
    
    /* QUARTER HEADER - CALCULATED FIXED PIXEL POSITION */
    .quarter-header-bottom {{
        position: absolute !important;
        bottom: 9px !important;  /* Calculated: (display height ~18px / 2) = 9px centers on border */
        left: 115px !important;  /* Adjusted further left: true center of time slider box */
        transform: translateX(-50%) !important;
        
        color: #262730 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        text-align: center !important;
        
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(0, 0, 0, 0.15) !important;
        border-radius: 6px !important;
        padding: 3px 10px !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15) !important;
        
        white-space: nowrap !important;
        z-index: 15 !important;
        animation: slideInLeft 0.8s cubic-bezier(0.4, 0, 0.2, 1) 0.2s both !important;
    }}
    
    @keyframes slideInLeft {{
        from {{ transform: translateX(-100px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    
    /* CONTENT STYLING - COMPACT STAT CARDS */
    .stat-card {{
        background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 6px 10px;
        text-align: center;
        color: white;
        font-family: 'DM Sans', sans-serif;
        min-width: 100px;
        flex: 1;
        max-width: 180px;
    }}
    
    .footer-text {{
        text-align: center;
        font-size: 1rem;
        font-weight: 600;
        color: #666;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #eee;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    /* MOBILE RESPONSIVE ADJUSTMENTS */
    @media (max-width: 768px) {{
        .stSidebar .stRadio {{
            padding: 20px 25px 40px 25px !important;
            margin: 5px 0 !important;
        }}
        
        .stSidebar .stRadio > div {{
            padding: 15px 3px 12px 3px !important;
            padding-bottom: 30px !important;
            min-height: 45px !important;
        }}
        
        .stSidebar .stRadio > div::before {{
            left: 3% !important;
            right: 3% !important;
        }}
        
        .stSidebar .stRadio::before {{
            left: 2% !important;
            font-size: 8px !important;
        }}
        
        .stSidebar .stRadio::after {{
            right: 2% !important;
            font-size: 8px !important;
        }}
        
        .stSidebar .stRadio > div > label {{
            width: 14px !important;
            height: 14px !important;
            min-width: 14px !important;
        }}
        
        .quarter-header-bottom {{
            bottom: -10px !important;
            font-size: 14px !important;
            padding: 5px 12px !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

    # --- Load Real Data ---
    @st.cache_data
    def load_data():
        try:
            df = pd.read_parquet("precomputed/pain_points_clusters.parquet")
            
            # Ensure sentiment_score is numeric
            if 'sentiment_score' in df.columns:
                df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
            
            # Handle date column properly
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['quarter'] = df['date'].dt.to_period("Q").astype(str)
            else:
                df['quarter'] = 'All'
            
            # Keep pain_topic_label as text data
            df = df.dropna(subset=['sentiment_score', 'cluster_name'])
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            # Return empty dataframe with proper structure as fallback
            return pd.DataFrame({
                'cluster_name': ['Meditation & Mindfulness'],
                'sentiment_score': [0.3],
                'pain_topic_label': ['Difficulty concentrating'],
                'quarter': ['2024Q3']
            })

    data = load_data()

    # --- Helper Functions ---
    def get_weather_type(sentiment):
        if sentiment >= 0.395: return 'sunny'
        elif sentiment >= 0.295: return 'partly-cloudy'
        elif sentiment >= -0.3: return 'cloudy'
        elif sentiment >= -0.6: return 'rainy'
        else: return 'stormy'

    def get_sentiment_description(sentiment):
        if sentiment >= 0.395: return "Sunny & Positive"
        elif sentiment >= 0.295: return "Mostly Positive"
        elif sentiment >= -0.2: return "Mixed Conditions"
        elif sentiment >= -0.6: return "Challenging"
        else: return "Alarming"

    def get_trend_description(sentiment):
        if sentiment > 0.5: return "Thriving Community"
        elif sentiment > 0: return "Growing Positivity"
        elif sentiment > -0.5: return "Stable Discussions"
        else: return "Needs Support"

    # --- Enhanced Data Processing ---
    @st.cache_data
    def process_weather_data(df, sentiment_column):
        # Calculate volume as actual count of discussions per cluster
        volume_stats = df.groupby('cluster_name').size().reset_index(name='volume')

        # Calculate average sentiment per cluster  
        sentiment_stats = df.groupby('cluster_name')[sentiment_column].mean().reset_index()

        # Merge volume and sentiment data
        topic_stats = volume_stats.merge(sentiment_stats, on='cluster_name')
        topic_stats['sentiment'] = topic_stats[sentiment_column]
        topic_stats['weather_type'] = topic_stats['sentiment'].apply(get_weather_type)
        topic_stats['weather_emoji'] = topic_stats['weather_type'].map({
            'sunny': '☀️', 'partly-cloudy': '⛅', 'cloudy': '☁️',
            'rainy': '🌧️', 'stormy': '⛈️'
        })
        
        vol_min, vol_max = topic_stats['volume'].min(), topic_stats['volume'].max()
        if vol_max > vol_min:
            vol_norm = (topic_stats['volume'] - vol_min) / (vol_max - vol_min)
            topic_stats['region_size'] = (vol_norm * 150 + 100).fillna(100).clip(upper=300).astype(int)
        else:
            topic_stats['region_size'] = 100
        
        return topic_stats.reset_index()

    # --- Enhanced Region Positions ---
    def get_region_positions():
        return {
            'Meditation & Mindfulness': {'left': '42%', 'top': '51%'},
            'Self-Regulation': {'left': '42%', 'top': '15%'},
            'Anxiety & Mental Health': {'left': '10%', 'top': '35%'},
            'Awareness': {'left': '15%', 'top': '60%'},
            'Buddhism & Spirituality': {'left': '70%', 'top': '30%'},
            'Concentration & Flow': {'left': '69%', 'top': '62%'},
            'Practice, Retreat, & Meta': {'left': '42%', 'top': '86%'}
        }

    # --- Enhanced Color Mapping for Topics ---
    def get_topic_colors(topic_name):
        if 'Meditation' in topic_name and 'Mindfulness' in topic_name:
            return {'primary': '#00CED1', 'secondary': '#20B2AA', 'border': '#008B8B'}
        elif 'Self-Regulation' in topic_name:
            return {'primary': '#1E90FF', 'secondary': '#4169E1', 'border': '#0000CD'}
        elif 'Anxiety' in topic_name or 'Mental Health' in topic_name:
            return {'primary': '#32CD32', 'secondary': '#228B22', 'border': '#006400'}
        elif 'Awareness' in topic_name:
            return {'primary': '#9ACD32', 'secondary': '#7FFF00', 'border': '#556B2F'}
        elif 'Buddhism' in topic_name or 'Spirituality' in topic_name:
            return {'primary': '#FFD700', 'secondary': '#FFA500', 'border': '#FF8C00'}
        elif 'Concentration' in topic_name or 'Flow' in topic_name:
            return {'primary': '#FF4500', 'secondary': '#FF6347', 'border': '#DC143C'}
        elif 'Practice' in topic_name or 'Retreat' in topic_name or 'Meta' in topic_name:
            return {'primary': '#FF1493', 'secondary': '#C71585', 'border': '#B22222'}
        else:
            return {'primary': '#87CEEB', 'secondary': '#4682B4', 'border': '#4682B4'}

    # --- ENHANCED SIDEBAR WITH QUARTER NAVIGATION ---
    expected_quarters = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2']
    available_quarters = [q for q in expected_quarters if q in data['quarter'].unique()]
    
    if len(available_quarters) < 6:
        available_quarters = expected_quarters
    
    default_q = '2025Q2' if '2025Q2' in available_quarters else available_quarters[-1]

    with st.sidebar:
        radio_container = st.container()
        
        with radio_container:
            selected_quarter = st.radio(
                "🌤️ Time Travel",
                options=available_quarters,
                index=available_quarters.index(default_q) if default_q in available_quarters else -1,
                key="enhanced_quarter_selector",
                horizontal=True
            )
            
            # Process data for current quarter to determine weather emoji
            sentiment_col = 'sentiment_score'
            quarter_data = data[data['quarter'] == selected_quarter].copy()
            current_quarter_processed = process_weather_data(quarter_data, sentiment_col)
            current_avg_sentiment = current_quarter_processed['sentiment'].mean()
            
            # Determine weather emoji
            if current_avg_sentiment >= 0.395:
                weather_emoji = "☀️"
            elif current_avg_sentiment >= 0.295:
                weather_emoji = "⛅"
            elif current_avg_sentiment >= -0.295:
                weather_emoji = "☁️"
            elif current_avg_sentiment >= -0.6:
                weather_emoji = "🌧️"
            else:
                weather_emoji = "⛈️"
            
            st.markdown(f"""
            <div class="quarter-header-bottom">
                {weather_emoji} {selected_quarter}
            </div>
            """, unsafe_allow_html=True)

    # --- MAIN CONTENT AREA ---
    
    # Process data for all quarters
    all_quarter_data = {}
    for quarter in available_quarters:
        quarter_data = data[data['quarter'] == quarter].copy() if quarter != 'All' else data.copy()
        all_quarter_data[quarter] = process_weather_data(quarter_data, sentiment_col)

    # Get current quarter data
    current_weather_data = all_quarter_data[selected_quarter]
    
    # Calculate total quarter volume for percentage calculations
    total_quarter_volume = current_weather_data['volume'].sum()
    
    # Weather mapping for emoji and descriptions
    weather_mapping = {
        'sunny': ('☀️', 'Sunny and Positive'),
        'partly-cloudy': ('⛅', 'Clearing Up'), 
        'cloudy': ('☁️', 'Overcast'),
        'rainy': ('🌧️', 'Light Showers'),
        'stormy': ('⛈️', 'Storm Warning')
    }
    
    weather_data = current_weather_data.copy()
    
    # Apply weather emojis and descriptions
    for weather_type, (emoji, desc) in weather_mapping.items():
        mask = weather_data['weather_type'] == weather_type
        weather_data.loc[mask, 'weather_emoji'] = emoji
        weather_data.loc[mask, 'weather_desc'] = desc

    # Calculate enhanced metrics
    sunny_regions = len(weather_data[weather_data['weather_type'] == 'sunny'])
    cloudy_regions = len(weather_data[weather_data['weather_type'] == 'cloudy'])
    rainy_regions = len(weather_data[weather_data['weather_type'] == 'rainy'])
    stormy_regions = len(weather_data[weather_data['weather_type'] == 'stormy'])
    avg_sentiment = weather_data['sentiment'].mean()
    unique_topics = weather_data['cluster_name'].nunique()
    display_total = weather_data['volume'].sum()

    if avg_sentiment >= 0.395: weather_emoji_main, weather_desc_main = "☀️", "Sunny"
    elif avg_sentiment >= 0.295: weather_emoji_main, weather_desc_main = "⛅", "Partly Cloudy"
    elif avg_sentiment >= -0.295: weather_emoji_main, weather_desc_main = "☁️", "Cloudy"
    elif avg_sentiment >= -0.6: weather_emoji_main, weather_desc_main = "🌧️", "Rainy"
    else: weather_emoji_main, weather_desc_main = "⛈️", "Stormy"

    # Title Section
    st.markdown("""
    <div style="text-align: center; color: white; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(45deg, #ffd700, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            🌦️ Meditation Weather Map</h1>
        <h3 style="font-size: 1.5rem; font-weight: 500; color:#333">How people feel</h3>
        <p style="font-size: 1rem; color: #888; max-width: 800px; margin: auto;">
            This weather map shows how people's sentiment change over time across topics of meditations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Calculate quarterly sentiment journey for trend visualization
    quarterly_sentiments = []
    for quarter in available_quarters:
        if quarter in all_quarter_data and len(all_quarter_data[quarter]) > 0:
            quarter_avg = all_quarter_data[quarter]['sentiment'].mean()
            quarterly_sentiments.append(quarter_avg)
        else:
            quarterly_sentiments.append(0.0)
    
    # Calculate growth vs previous quarter
    current_quarter_index = available_quarters.index(selected_quarter) if selected_quarter in available_quarters else -1
    growth_text = selected_quarter
    
    if current_quarter_index > 0:
        prev_quarter = available_quarters[current_quarter_index - 1]
        if prev_quarter in all_quarter_data and len(all_quarter_data[prev_quarter]) > 0:
            prev_volume = all_quarter_data[prev_quarter]['volume'].sum()
            current_volume = display_total
            if prev_volume > 0:
                growth_pct = ((current_volume - prev_volume) / prev_volume) * 100
                if growth_pct > 0:
                    growth_text = f"+{growth_pct:.0f}% QoQ"
                else:
                    growth_text = f"{growth_pct:.0f}% QoQ"
            else:
                growth_text = f"QoQ"
    
    # Normalize sentiment values for sparkline
    if len(quarterly_sentiments) > 0:
        min_sent = min(quarterly_sentiments)
        max_sent = max(quarterly_sentiments)
        if max_sent > min_sent:
            normalized_sentiments = [(s - min_sent) / (max_sent - min_sent) for s in quarterly_sentiments]
        else:
            normalized_sentiments = [0.5] * len(quarterly_sentiments)
    else:
        normalized_sentiments = [0.5] * 6
    
    # Enhanced Cardinal Spline implementation
    def create_sparkline_path(values, width=65, height=22, tension=0.4):
        """Create smooth Cardinal spline path for sparkline visualization."""
        if len(values) < 2:
            return f"M 0,{height/2} L {width},{height/2}"
        
        points = []
        for i, val in enumerate(values):
            x = (i / (len(values) - 1)) * width
            y = height - (val * height)
            points.append((x, y))
        
        if len(points) < 3:
            return f"M {points[0][0]},{points[0][1]} L {points[1][0]},{points[1][1]}"
        
        def get_cardinal_point(p0, p1, p2, p3, t, tension):
            t2 = t * t
            t3 = t2 * t
            
            c0 = -tension * t3 + 2 * tension * t2 - tension * t
            c1 = (2 - tension) * t3 + (tension - 3) * t2 + 1
            c2 = (tension - 2) * t3 + (3 - 2 * tension) * t2 + tension * t
            c3 = tension * t3 - tension * t2
            
            x = c0 * p0[0] + c1 * p1[0] + c2 * p2[0] + c3 * p3[0]
            y = c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1]
            
            return (x, y)
        
        path = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
        
        num_segments_per_span = 8
        
        for i in range(len(points) - 1):
            p0 = points[max(0, i - 1)]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[min(len(points) - 1, i + 2)]
            
            for j in range(1, num_segments_per_span + 1):
                t = j / num_segments_per_span
                curve_point = get_cardinal_point(p0, p1, p2, p3, t, tension)
                path += f" L {curve_point[0]:.2f},{curve_point[1]:.2f}"
        
        return path

    sparkline_path = create_sparkline_path(normalized_sentiments, tension=0.4)

    # Find current quarter position for emphasis
    current_quarter_index = available_quarters.index(selected_quarter) if selected_quarter in available_quarters else -1
    if current_quarter_index >= 0:
        t_param = current_quarter_index / (len(normalized_sentiments) - 1)
        current_x = t_param * 65
        current_y = 22 - (normalized_sentiments[current_quarter_index] * 22)
    else:
        current_x, current_y = 65, 11

    # Calculate sentiment color
    if avg_sentiment >= 0.395:
        sentiment_color = "lime"
    elif avg_sentiment >= 0.295:
        sentiment_color = "yellow"
    else:
        sentiment_color = "orange"

    # Ultra-Clean Statistics Bar
    st.markdown(f"""
    <div style="background: {colors['bg_gradient']}; border-radius: 15px; padding: {int(8*1.15)}px {int(12*1.15)}px; margin: 2px 0 0px 0; display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; width: 100%; box-sizing: border-box;">
        <div class="stat-card" style="padding: {int(6*1.15)}px {int(10*1.15)}px; min-width: 90px; flex: 1; display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div style="height: 37px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 24px; font-weight: 700; line-height: 1;">{display_total:,}</div>
            </div>
            <div style="height: 21px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 11px; opacity: 0.9; font-weight: 500;">Discussions</div>
            </div>
            <div style="height: 16px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 9px; opacity: 0.7;">{growth_text}</div>
            </div>
        </div>
        <div class="stat-card" style="padding: {int(6*1.15)}px {int(10*1.15)}px; min-width: 90px; flex: 1; display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div style="height: 37px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 24px; font-weight: 700; line-height: 1;">{unique_topics}</div>
            </div>
            <div style="height: 21px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 11px; opacity: 0.9; font-weight: 500;">Regions</div>
            </div>
            <div style="height: 16px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 9px; opacity: 0.7;">Main Topics</div>
            </div>
        </div>
        <div class="stat-card" style="padding: {int(6*1.15)}px {int(10*1.15)}px; min-width: 90px; flex: 1; display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div style="height: 37px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 24px; font-weight: 700; line-height: 1;">{sunny_regions}</div>
            </div>
            <div style="height: 21px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 11px; opacity: 0.9; font-weight: 500;">Sunny</div>
            </div>
            <div style="height: 16px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 9px; opacity: 0.7;">vs {cloudy_regions} cloudy</div>
            </div>
        </div>
        <div class="stat-card" style="padding: {int(6*1.15)}px {int(10*1.15)}px; min-width: 90px; flex: 1; display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div style="height: 37px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 18px; font-weight: 700; line-height: 1;">{weather_desc_main}</div>
            </div>
            <div style="height: 21px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 11px; opacity: 0.9; font-weight: 500;">Climate</div>
            </div>
            <div style="height: 16px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 9px; opacity: 0.9; color: {sentiment_color}; font-weight: 600;">{avg_sentiment:+.2f}</div>
            </div>
        </div>
        <div class="stat-card" style="padding: {int(6*1.15)}px {int(10*1.15)}px; min-width: 90px; flex: 1; display: flex; flex-direction: column; align-items: center; text-align: center;">
            <div style="height: 37px; display: flex; align-items: center; justify-content: center;">
                <svg width="65" height="22" style="overflow: visible;">
                    <path d="{sparkline_path}" stroke="rgba(255,255,255,0.9)" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="{current_x}" cy="{current_y}" r="3" fill="rgba(255,255,255,1)" stroke="rgba(255,255,255,0.3)" stroke-width="5"/>
                </svg>
            </div>
            <div style="height: 21px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 11px; opacity: 0.9; font-weight: 500;">Trend</div>
            </div>
            <div style="height: 16px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 9px; opacity: 0.7;">Sentiment</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- ENHANCED CUSTOM WEATHER MAP COMPONENT ---
    region_positions = get_region_positions()
    
    # Eliminate gaps between components
    st.markdown("""
    <style>
        .stHtml > div {
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        iframe {
            width: 100% !important;
            border: none !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .element-container {
            margin-bottom: 0 !important;
            margin-top: 0 !important;
        }
        
        .stMarkdown + .stHtml,
        .stMarkdown + .stHtml > div {
            margin-top: -10px !important;
        }
        
        .stMarkdown {
            margin-bottom: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Generate HTML for each region with dynamic sizing
    regions_html = ""
    for _, region in weather_data.iterrows():
        region_name = region['cluster_name']
        position = region_positions.get(region_name, {'left': '50%', 'top': '50%'})
        colors_topic = get_topic_colors(region_name)
        
        size_px = int(region['region_size'] * 1.2)
        volume_percentage = (region['volume'] / total_quarter_volume) * 100
        
        # Format region name for display
        words = region_name.split()
        if len(words) > 2:
            mid_point = len(words) // 2
            line1 = ' '.join(words[:mid_point])
            line2 = ' '.join(words[mid_point:])
            formatted_name = f"{line1}<br>{line2}"
        else:
            formatted_name = region_name
        
        # Smart tooltip positioning
        tooltip_position_style = ""
        if 'Practice' in region_name and 'Meta' in region_name:
            tooltip_position_style = """
                top: auto !important;
                bottom: 100% !important;
                left: 20px !important;
                transform: translateY(-8px) !important;
            """
        elif position['left'] and float(position['left'].replace('%', '')) > 70:
            tooltip_position_style = """
                left: auto !important;
                right: -10px !important;
            """
        
        regions_html += f"""
        <div class="weather-region {region['weather_type']}" 
             data-weather="{region['weather_type']}"
             data-sentiment="{region['sentiment']:.2f}"
             data-volume="{region['volume']}"
             style="
                position: absolute;
                left: {position['left']};
                top: {position['top']};
                width: {size_px}px;
                height: {size_px * 0.7}px;
                background: linear-gradient(135deg, {colors_topic['primary']}, {colors_topic['secondary']});
                border: 3px solid {colors_topic['border']};
                border-radius: 20px;
                transform: translate(-50%, -50%);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                color: white;
                font-family: 'DM Sans', sans-serif;
                text-align: center;
                padding: 15px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                backdrop-filter: blur(10px);
             ">
            <div class="weather-emoji" style="font-size: {min(24, size_px//8)}px; margin-bottom: 8px;">
                {region['weather_emoji']}
            </div>
            <div class="region-title" style="font-size: {min(14, size_px//12)}px; font-weight: 800; line-height: 1.2; margin-bottom: 5px;">
                {formatted_name}
            </div>
            <div class="weather-desc" style="font-size: {min(9, size_px//18)}px; opacity: 0.9; font-weight: 500;">
                {region['weather_desc']}
            </div>
            
            <div class="region-tooltip" style="
                position: absolute;
                top: 100%;
                left: -10px;
                transform: translateY(8px);
                background: rgba(0,0,0,0.75);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.4);
                color: white;
                padding: 6px 8px;
                border-radius: 6px;
                font-size: 9px;
                opacity: 0;
                pointer-events: none;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                z-index: 2000;
                box-shadow: 
                    0 4px 15px rgba(0,0,0,0.3),
                    inset 0 1px 0 rgba(255,255,255,0.2);
                min-width: 180px;
                max-width: 220px;
                text-shadow: 0 1px 3px rgba(0,0,0,0.8);
                text-align: left;
                white-space: normal;
                line-height: 1.1;
                {tooltip_position_style}
            ">
                <div style="font-weight: 700; font-size: 10px; margin-bottom: 3px; line-height: 1.1; text-align: left;">
                    {region['weather_emoji']} {region_name}
                </div>
                <div style="font-size: 8px; line-height: 1.2; text-align: left;">
                    <span style="opacity: 0.9;">Sentiment:</span> <strong style="color: {('lime' if region['sentiment'] >= 0.395 else 'yellow' if region['sentiment'] >= 0.295 else 'orange')};">{region['sentiment']:.2f}</strong><br>
                    <span style="opacity: 0.9;">Volume:</span> <strong>{volume_percentage:.1f}%</strong><br>
                    <span style="opacity: 0.9;">Trend:</span> <strong style="font-size: 7px;">{get_trend_description(region['sentiment']).split(' ')[0]}</strong>
                </div>
            </div>
        </div>
        """

    # Enhanced Weather Map with Custom Components
    st.components.v1.html(f"""
    <!DOCTYPE html>
    <html style="margin: 0; padding: 0;">
    <head>
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                margin: 0;
                padding: 0;
                overflow-x: hidden;
            }}
            .weather-map-container {{
                position: relative;
                width: 100%;
                height: 568px;
                background: {colors['bg_gradient']};
                border-radius: 20px;
                overflow: hidden;
                margin: -2px 0 2px 0;
                box-sizing: border-box;
            }}
            
            /* WEATHER-CONDITION-BASED ANIMATIONS */
            .sunny {{
                animation: sunny-pulse 3s ease-in-out infinite !important;
            }}
            
            @keyframes sunny-pulse {{
                0%, 100% {{ 
                    transform: translate(-50%, -50%) scale(1);
                    filter: brightness(1) drop-shadow(0 0 15px rgba(255, 215, 0, 0.3));
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15), 0 0 20px rgba(255, 215, 0, 0.2);
                }}
                50% {{ 
                    transform: translate(-50%, -50%) scale(1.05);
                    filter: brightness(1.15) drop-shadow(0 0 30px rgba(255, 215, 0, 0.6));
                    box-shadow: 0 12px 35px rgba(0,0,0,0.25), 0 0 35px rgba(255, 215, 0, 0.4);
                }}
            }}
            
            .partly-cloudy {{
                animation: hope-breathe 4s ease-in-out infinite !important;
            }}
            
            @keyframes hope-breathe {{
                0%, 100% {{ 
                    transform: translate(-50%, -50%) translateY(0px) scale(1);
                    filter: brightness(1);
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                }}
                25% {{ 
                    transform: translate(-50%, -50%) translateY(-5px) scale(1.02);
                    filter: brightness(1.08);
                    box-shadow: 0 15px 30px rgba(0,0,0,0.2);
                }}
                75% {{ 
                    transform: translate(-50%, -50%) translateY(3px) scale(0.98);
                    filter: brightness(0.95);
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                }}
            }}
            
            .cloudy {{
                animation: contemplative-drift 8s ease-in-out infinite !important;
            }}
            
            @keyframes contemplative-drift {{
                0%, 100% {{ 
                    transform: translate(-50%, -50%) translateX(0px) translateY(0px) scale(1);
                    filter: brightness(0.95);
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                }}
                25% {{ 
                    transform: translate(-50%, -50%) translateX(3px) translateY(-2px) scale(1.01);
                    filter: brightness(1.02);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
                }}
                50% {{ 
                    transform: translate(-50%, -50%) translateX(-2px) translateY(4px) scale(0.995);
                    filter: brightness(0.88);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
                }}
                75% {{ 
                    transform: translate(-50%, -50%) translateX(2px) translateY(2px) scale(1.005);
                    filter: brightness(0.98);
                    box-shadow: 0 9px 28px rgba(0,0,0,0.16);
                }}
            }}
            
            .rainy {{
                animation: gentle-rain-shake 5s ease-in-out infinite !important;
            }}
            
            @keyframes gentle-rain-shake {{
                0%, 100% {{ 
                    transform: translate(-50%, -50%) translateX(0px) scale(1);
                    filter: brightness(0.85);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
                }}
                25% {{ 
                    transform: translate(-50%, -50%) translateX(-2px) scale(0.99);
                    filter: brightness(0.8);
                    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
                }}
                75% {{ 
                    transform: translate(-50%, -50%) translateX(2px) scale(1.01);
                    filter: brightness(0.9);
                    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
                }}
            }}
            
            .stormy {{
                animation: storm-alert 2s ease-in-out infinite !important;
            }}
            
            @keyframes storm-alert {{
                0%, 100% {{ 
                    transform: translate(-50%, -50%) scale(1);
                    filter: brightness(0.7) contrast(1.2);
                    box-shadow: 0 8px 25px rgba(0,0,0,0.3), 0 0 15px rgba(255, 0, 0, 0.3);
                }}
                50% {{ 
                    transform: translate(-50%, -50%) scale(1.03);
                    filter: brightness(1.1) contrast(1.5);
                    box-shadow: 0 12px 35px rgba(0,0,0,0.4), 0 0 25px rgba(255, 0, 0, 0.5);
                }}
            }}
            
            /* Enhanced Hover effects */
            .weather-region {{
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }}
            
            .weather-region:hover {{
                transform: translate(-50%, -50%) scale(1.1) !important;
                z-index: 1000 !important;
                filter: brightness(1.05) saturate(1.05) !important;
                box-shadow: 
                    0 6px 16px rgba(0,0,0,0.3),
                    0 0 10px rgba(255,255,255,0.05),
                    inset 0 0 20px rgba(255,255,255,0.1) !important;
                animation-play-state: paused !important;
                border-width: 4px !important;
            }}
            
            .weather-region:hover .region-tooltip {{
                opacity: 1 !important;
                transform: translateY(5px) !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }}
            
            .weather-region:hover .weather-emoji {{
                transform: scale(1.2) !important;
                transition: transform 0.3s ease !important;
            }}
            
            .weather-region:hover .region-title {{
                text-shadow: 0 0 10px rgba(255,255,255,0.8) !important;
                transition: text-shadow 0.3s ease !important;
            }}
            
            /* Hover ripple effect */
            .weather-region::before {{
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 30%);
                transform: translate(-50%, -50%);
                transition: all 0.6s ease;
                pointer-events: none;
                z-index: -1;
            }}
            
            .weather-region:hover::before {{
                width: 200%;
                height: 200%;
            }}
            
            /* Enhanced Legend */
            .weather-legend {{
                position: absolute;
                top: 15px;
                right: 20px;
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 15px;
                padding: 12px 12px;
                color: white;
                font-family: 'DM Sans', sans-serif;
                font-size: 11px;
                min-width: 140px;
                max-width: 180px;
            }}
            
            .legend-title {{
                font-size: 12px;
                font-weight: 800;
                margin-bottom: 4px;
                text-align: Left;
            }}
            
            .legend-item {{
                margin-bottom: 4px;
                display: flex;
                align-items: center;
                gap: 8px;
                line-height: 1.1;
            }}
            
            .legend-item:last-of-type {{
                margin-bottom: 0;
            }}
            
            .legend-emoji {{
                font-size: 14px;
                flex-shrink: 0;
            }}
            
            .legend-text {{
                flex: 1;
                font-size: 10px;
                font-weight: 600;
            }}
            
            /* Accessibility */
            @media (prefers-reduced-motion: reduce) {{
                .sunny, .partly-cloudy, .cloudy, .rainy, .stormy {{
                    animation: none !important;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="weather-map-container">
            <!-- Weather Regions -->
            {regions_html}
            
            <!-- Enhanced Legend -->
            <div class="weather-legend">
                <div class="legend-title">🌦️ Weather Legend</div>
                <div class="legend-item">
                    <span class="legend-emoji">☀️</span>
                    <span class="legend-text">Sunny and Positive (0.4+)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-emoji">⛅</span>
                    <span class="legend-text">Clearing Up (0.3+)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-emoji">☁️</span>
                    <span class="legend-text">Overcast (-0.3/0.3)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-emoji">🌧️</span>
                    <span class="legend-text">Light Showers (-0.6)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-emoji">⛈️</span>
                    <span class="legend-text">Storm Warning (-0.6+)</span>
                </div>
                <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.2); font-size: 9px; color: #ffffff;">
                    📅 <strong>{selected_quarter}</strong><br>
                    <em><strong>Region Size = Discussion Volume</strong><br><strong>>>Hover for more details</strong></em><br>
                </div>
            </div>
        </div>
        
        <script>
            // Enhanced interaction logging
            const regions = document.querySelectorAll('.weather-region');
            const weatherCount = {{'sunny': 0, 'partly-cloudy': 0, 'cloudy': 0}};
            let totalVolume = 0;
            
            // Add enhanced hover event listeners
            regions.forEach((region, index) => {{
                const weather = region.getAttribute('data-weather');
                const sentiment = parseFloat(region.getAttribute('data-sentiment'));
                const volume = parseInt(region.getAttribute('data-volume'));
                const tooltip = region.querySelector('.region-tooltip');
                const regionTitle = region.querySelector('.region-title');
                const regionName = regionTitle ? regionTitle.textContent.replace(/\s+/g, ' ').trim() : 'Unknown';
                
                weatherCount[weather]++;
                totalVolume += volume;
                
                // Enhanced hover events
                region.addEventListener('mouseenter', function() {{
                    console.log(`🎯 HOVER: ${{regionName}} | ${{weather}} (${{sentiment.toFixed(2)}}) | ${{volume.toLocaleString()}}`);
                }});
                
                region.addEventListener('mouseleave', function() {{
                    console.log(`⬅️ HOVER OUT: ${{regionName}}`);
                }});
                
                region.addEventListener('click', function() {{
                    console.log(`🖱️ CLICKED: ${{regionName}} - ${{weather}} region with ${{volume.toLocaleString()}} discussions`);
                }});
                
                console.log(`${{regionName}}: vol=${{volume.toLocaleString()}}, sentiment=${{sentiment.toFixed(3)}} → ${{weather}} motion`);
            }});
              
            // Keyboard navigation support
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'h' || e.key === 'H') {{
                    console.log('🔍 HELP: Hover over weather regions for sentiment and volume info');
                    console.log('💡 TIP: Tooltips show volume as % of quarter total + trend info');
                    console.log('🌦️ Weather: ☀️ Sunny (0.4+), ⛅ Partly Cloudy (0.3+), ☁️ Cloudy (-0.3/0.3), 🌧️ Rainy (-0.6), ⛈️ Stormy (-0.6+)');
                    console.log('📅 Real data from meditation communities');
                }}
            }});
            
            console.log('⌨️ Press "H" for help | 🖱️ Hover for enhanced tooltips with volume percentages');
        </script>
    </body>
    </html>
    """, height=640, width=None)

    # --- ENHANCED SENTIMENT RIVER FLOW ANALYSIS ---
    
    # Function to get top challenge per region per quarter
    def get_top_challenges_by_region(df, selected_quarter):
        """Get top challenge for each region in the selected quarter with quarter-wide percentages"""
        region_challenges = {}
        
        # Filter data for selected quarter
        quarter_data = df[df['quarter'] == selected_quarter].copy()
        
        # Get total quarter volume for percentage calculation
        total_quarter_volume = len(quarter_data)
        
        # For each unique region/cluster
        for region in quarter_data['cluster_name'].unique():
            region_data = quarter_data[quarter_data['cluster_name'] == region]
            
            # Get top challenge using pain_topic_label
            if 'pain_topic_label' in region_data.columns and len(region_data) > 0:
                # Remove null values
                non_null_data = region_data['pain_topic_label'].dropna()
                
                if len(non_null_data) > 0:
                    # Direct value_counts
                    top_challenges = non_null_data.value_counts().head(3)
                    
                    # Store with quarter-wide percentages
                    region_challenges[region] = [
                        (challenge, count, (count / total_quarter_volume) * 100) 
                        for challenge, count in top_challenges.items()
                    ]
                else:
                    region_challenges[region] = []
            else:
                region_challenges[region] = []
        
        return region_challenges

    def calculate_trend_momentum(all_quarter_data, available_quarters, selected_quarter, region_challenges):
        """Calculate QoQ sentiment trends and momentum analysis"""
        current_index = available_quarters.index(selected_quarter)
        
        # Get current quarter data
        current_data = all_quarter_data[selected_quarter]
        
        if current_index > 0:
            # Has previous quarter - calculate QoQ changes
            prev_quarter = available_quarters[current_index - 1]
            prev_data = all_quarter_data[prev_quarter]
            
            # Merge current and previous data
            comparison_df = current_data[['cluster_name', 'sentiment', 'volume']].merge(
                prev_data[['cluster_name', 'sentiment', 'volume']], 
                on='cluster_name', 
                how='outer', 
                suffixes=('_current', '_prev')
            ).fillna(0)
            
            # Calculate sentiment change
            comparison_df['sentiment_change'] = comparison_df['sentiment_current'] - comparison_df['sentiment_prev']
            comparison_df['volume_change'] = comparison_df['volume_current'] - comparison_df['volume_prev']
            comparison_df['volume_change_pct'] = (comparison_df['volume_change'] / comparison_df['volume_prev'].replace(0, 1)) * 100
            
            # Identify meaningful changes
            rising_stars = comparison_df[comparison_df['sentiment_change'] >= 0.05].sort_values('sentiment_change', ascending=False)
            declining_regions = comparison_df[comparison_df['sentiment_change'] <= -0.05].sort_values('sentiment_change')
            
        else:
            # No previous quarter - fallback to best/worst regions
            comparison_df = current_data.copy()
            rising_stars = current_data.sort_values('sentiment', ascending=False)
            declining_regions = current_data.sort_values('sentiment')
            comparison_df['sentiment_change'] = 0
        
        return rising_stars, declining_regions, comparison_df, current_index > 0

    # Get challenge data for current quarter    
    # --- SENTIMENT RIVER FLOW ANALYSIS (Replacing Trend Momentum) ---
    
    # Get challenge data for current quarter
    region_challenges = get_top_challenges_by_region(data, selected_quarter)
    
    # Calculate trend momentum for river flow
    rising_stars, declining_regions, comparison_df, has_previous = calculate_trend_momentum(
        all_quarter_data, available_quarters, selected_quarter, region_challenges
    )
    
    # Determine if we're in first quarter (no previous data)
    is_first_quarter = selected_quarter == '2024Q1' or not has_previous
    
    # Create River Flow HTML Component
    def create_river_flow_html():
        # Generate challenge bubbles and river paths for each topic
        river_topics_html = ""
        
        if is_first_quarter:
            # For 2024Q1, show active challenges only - integrated header
            # For quarters with previous data, show river flow
            st.markdown("""
            <style>
            .custom-subtitle {
                font-size: 24px;
                color: #333333;
                margin: 0px 0 5px 0; 
                font-weight: 600;
                font-family: 'DM Sans', sans-serif;
            }
            </style>
            """, unsafe_allow_html=True)

            # Render the styled subtitle
            st.markdown(f"""
            <h4 class="custom-subtitle" style="text-align: center; margin-top: -40px;">
                Regional Topic Trends and Active Challenges – {selected_quarter}
            </h4>
            """, unsafe_allow_html=True)

            # Define baseline variables for first quarter
            trend_emoji = "📍"
            trend_text = "Establishing baseline. QoQ trends available from next quarter.."
            trend_color = "#64748b"  # Grey for baseline

            header_section = f"""
                <div class="header-content" style ="text-align: center;">
                    <div class="trend-summary baseline" style="background: linear-gradient(135deg, {trend_color}15, {trend_color}05); border-color: {trend_color}40;">
                        <span class="trend-emoji">{trend_emoji}</span>
                        <span class="trend-text" style="color: {trend_color};">{trend_text}</span>
                    </div>
                </div>
            """
            
            # Process current quarter data for active challenges display
            for _, topic_data in current_weather_data.iterrows():
                topic_name = topic_data['cluster_name']
                sentiment = topic_data['sentiment']
                volume = topic_data['volume']
                weather_emoji = topic_data['weather_emoji']
                
                # Get challenges for this topic
                topic_challenges = region_challenges.get(topic_name, [])
                
                # Create challenge bubbles HTML
                challenge_bubbles_html = ""
                challenge_items_html = ""
                
                if topic_challenges:
                    for i, (challenge_text, count, percentage) in enumerate(topic_challenges[:3]):
                        # Determine challenge severity
                        if percentage >= 2.0:
                            severity = "high"
                            icon = "H"
                        elif percentage >= 1.0:
                            severity = "medium" 
                            icon = "M"
                        else:
                            severity = "low"
                            icon = "L"
                        
                        # Position bubbles along a horizontal line (no previous quarter to compare)
                        bubble_x = 20 + (i * 25)  # Spread bubbles horizontally
                        
                        challenge_bubbles_html += f"""
                        <div class="challenge-bubble {severity}" style="left: {bubble_x}%; top: 50%;">{icon}</div>
                        <div class="challenge-tooltip" style="left: {bubble_x}%; top: 50%;">
                            <strong>{challenge_text} ({percentage:.1f}%)</strong><br>
                            Challenge intensity: {severity.title()}
                        </div>
                        """
                        
                        challenge_items_html += f"""
                        <div class="challenge-item {severity}">
                            <div class="challenge-severity challenge-{severity}"></div>
                            <div class="challenge-content">
                                <div class="challenge-text">{challenge_text}</div>
                                <div class="challenge-stats">{percentage:.1f}% of discussions • {severity.title()} intensity</div>
                            </div>
                        </div>
                        """
                
                # Create single-point "river" for baseline
                river_topics_html += f"""
                <div class="topic-flow baseline">
                    <div class="topic-header">
                        <span class="topic-emoji">{weather_emoji}</span>
                        <span class="topic-name">{topic_name}</span>
                    </div>
                    
                    <div class="flow-svg-container">
                        <svg class="flow-svg" viewBox="0 0 800 120">
                            <!-- Single baseline point -->
                            <circle cx="400" cy="60" r="15" fill="#94a3b8" opacity="0.8"/>
                            <text x="400" y="85" text-anchor="middle" font-size="12" fill="#475569" font-weight="600">{sentiment:.2f}</text>
                            <text x="400" y="35" text-anchor="middle" font-size="10" fill="#64748b" opacity="0.8">Baseline</text>
                        </svg>
                        {challenge_bubbles_html}
                    </div>
                    
                    <div class="metrics-glass-bar">
                        <div class="metrics-section">
                            <div class="metric-item">
                                <span class="metric-label">Change:</span>
                                <span class="metric-value">{sentiment:.2f}</span>
                            </div>
                            <div class="discussions-pill">{volume:,} discussions</div>
                        </div>
                        <div class="metrics-section">
                            <div class="status-badge stable">
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Active Challenges:</span>
                                <span class="challenges-count">{len(topic_challenges)}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="challenge-insights">
                        <div class="insights-header">
                            <span>⚡</span>
                            <span>Active Challenge Analysis</span>
                        </div>
                        <div class="challenge-list">
                            {challenge_items_html if challenge_items_html else '<div class="challenge-item low"><div class="challenge-severity challenge-low"></div><div class="challenge-content"><div class="challenge-text">No significant challenges detected</div><div class="challenge-stats">Healthy baseline state</div></div></div>'}
                        </div>
                    </div>
                </div>
                """
        
        else:
            # For quarters with previous data, show river flow with integrated header
            prev_quarter = available_quarters[available_quarters.index(selected_quarter) - 1]
            overall_change = comparison_df['sentiment_change'].mean()
            
            if overall_change > 0.02:
                trend_emoji = "📈"
                trend_text = f"Rising (+{overall_change:.2f}) vs {prev_quarter}"
                trend_color = "#10b981"
            elif overall_change < -0.02:
                trend_emoji = "📉" 
                trend_text = f"Declining ({overall_change:.2f}) vs {prev_quarter}"
                trend_color = "#f97316"
            else:
                trend_emoji = "➡️" 
                trend_text = f"Steady ({overall_change:+.2f}) vs {prev_quarter}"
                trend_color = "#64748b"
            
            # For quarters with previous data, show river flow
            st.markdown("""
            <style>
            .custom-subtitle {
                font-size: 24px;
                color: #333333;
                margin: 0px 0 5px 0; 
                font-weight: 600;
                font-family: 'DM Sans', sans-serif;
            }
            </style>
            """, unsafe_allow_html=True)

            # Render the styled subtitle
            st.markdown(f"""
            <h4 class="custom-subtitle" style="text-align: center; margin-top: -40px;">
                Regional Topic Trends and Active Challenges – {selected_quarter}
            </h4>
            """, unsafe_allow_html=True)

            header_section = f"""
                <div class="header-content" style="text-align: center;">
                    <div class="trend-summary" style="margin-top: -80px;background: linear-gradient(135deg, {trend_color}15, {trend_color}05); border-color: {trend_color}40;">
                        <span class="trend-emoji">{trend_emoji}</span>
                        <span class="trend-text" style="color: {trend_color};">{trend_text}</span>
                        <br></br>
                    </div>
                </div>
            """
            
            # Process all topics for river flow
            all_topics = comparison_df.merge(
                current_weather_data[['cluster_name', 'weather_emoji']], 
                on='cluster_name', 
                how='left'
            )
            
            for _, topic_data in all_topics.iterrows():
                topic_name = topic_data['cluster_name']
                sentiment_current = topic_data['sentiment_current']
                sentiment_prev = topic_data['sentiment_prev'] 
                sentiment_change = topic_data['sentiment_change']
                volume_change_pct = topic_data['volume_change_pct']
                weather_emoji = topic_data.get('weather_emoji', '☁️')
                
                # SINGLE COLOR TREND LINES - No gradients, consistent styling
                if sentiment_change >= 0.05:
                    flow_type = "rising"
                    flow_color = "#10b981"  # Green for rising - SINGLE COLOR
                    path_data = "M 50 80 Q 300 50 750 20"  # Rising curve
                    left_y, right_y = 80, 20
                elif sentiment_change <= -0.05:
                    flow_type = "declining" 
                    flow_color = "#f97316"  # Orange for declining - SINGLE COLOR
                    path_data = "M 50 40 Q 300 70 750 100"  # Declining curve
                    left_y, right_y = 40, 100
                else:
                    flow_type = "stable"
                    flow_color = "#94a3b8"  # Grey for stable - SINGLE COLOR
                    path_data = "M 50 60 Q 400 45 750 60"  # Nearly flat with slight variation
                    left_y, right_y = 60, 60

                # ✅ Add this line after setting flow_type
                glow_filter = "glow-strong" if flow_type == "stable" else "glow"
                
                # Always use purple for right dot (footer color) and grey for left dot
                right_dot_color = "#6C5CE7"  # Purple footer color for all trends
                
                # Get challenges for this topic
                topic_challenges = region_challenges.get(topic_name, [])
                
                # Create challenge bubbles along the river path
                challenge_bubbles_html = ""
                challenge_items_html = ""
                
                if topic_challenges:
                    for i, (challenge_text, count, percentage) in enumerate(topic_challenges[:3]):
                        # Determine challenge severity
                        if percentage >= 2.0:
                            severity = "high"
                            icon = "H"
                        elif percentage >= 1.0:
                            severity = "medium"
                            icon = "M"
                        else:
                            severity = "low"
                            icon = "L"
                        
                        # Position bubbles along the sentiment path based on flow type
                        if flow_type == "rising":
                            bubble_positions = [(25, 70), (50, 55), (75, 35)]
                        elif flow_type == "declining":
                            bubble_positions = [(25, 50), (50, 65), (75, 85)]
                        else:  # stable
                            bubble_positions = [(25, 60), (50, 58), (75, 60)]
                        
                        if i < len(bubble_positions):
                            pos_x, pos_y = bubble_positions[i]
                            challenge_bubbles_html += f"""
                            <div class="challenge-bubble {severity}" style="left: {pos_x}%; top: {pos_y}%;">{icon}</div>
                            <div class="challenge-tooltip" style="left: {pos_x}%; top: {pos_y}%;">
                                <strong>{challenge_text} ({percentage:.1f}%)</strong><br>
                                Challenge intensity: {severity.title()}
                            </div>
                            """
                        
                        challenge_items_html += f"""
                        <div class="challenge-item {severity}">
                            <div class="challenge-severity challenge-{severity}"></div>
                            <div class="challenge-content">
                                <div class="challenge-text">{challenge_text}</div>
                                <div class="challenge-stats">{percentage:.1f}% of discussions • {severity.title()} intensity</div>
                            </div>
                        </div>
                        """
                
                # Add challenge rapids for visual turbulence
                rapids_html = ""
                if topic_challenges and len(topic_challenges) > 0:
                    avg_challenge_intensity = sum(p[2] for p in topic_challenges[:3]) / len(topic_challenges[:3])
                    if avg_challenge_intensity > 1.5:
                        rapids_html = """
                        <path class="rapid" d="M 200 50 Q 250 40 300 45" stroke="#ef4444" opacity="0.5"/>
                        <path class="rapid" d="M 450 55 Q 500 65 550 60" stroke="#ef4444" opacity="0.4"/>
                        """
                    elif avg_challenge_intensity > 0.8:
                        rapids_html = """
                        <path class="rapid" d="M 350 52 Q 400 48 450 50" stroke="#f97316" opacity="0.3"/>
                        """
                
                river_topics_html += f"""
                <div class="topic-flow {flow_type}">
                    <div class="topic-header">
                        <span class="topic-emoji">{weather_emoji}</span>
                        <span class="topic-name">{topic_name}</span>
                    </div>
                    
                    <div class="flow-svg-container">
                        <svg class="flow-svg" viewBox="0 0 800 120">
                            <defs>
                                <!-- Glow effect only - no gradients -->
                                <filter id="glow">
                                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                                    <feMerge> 
                                        <feMergeNode in="coloredBlur"/>
                                        <feMergeNode in="SourceGraphic"/>
                                    </feMerge>
                                </filter>
                                <filter id="glow-strong">
                                    <feGaussianBlur stdDeviation="10" result="coloredBlur"/>
                                    <feMerge>
                                        <feMergeNode in="coloredBlur"/>
                                        <feMergeNode in="SourceGraphic"/>
                                    </feMerge>
                                    </filter>
                            </defs>
                            
                            <!-- SINGLE COLOR TREND LINES: Same thickness and style for all -->
                            <path d="{path_data}" 
                                  stroke="{flow_color}" 
                                  stroke-width="8" 
                                  fill="none" 
                                  stroke-linecap="round"
                                  stroke-linejoin="round"
                                  filter="url(#{glow_filter})"
                                  opacity="0.9"/>
                            
                            <!-- Challenge rapids -->
                            {rapids_html}
                            
                            <!-- Sentiment value points: LEFT ALWAYS GREY, RIGHT ALWAYS PURPLE -->
                            <circle cx="50" cy="{left_y}" r="10" fill="#94a3b8" opacity="0.8" stroke="rgba(148,163,184,0.5)" stroke-width="1"/>
                            <circle cx="750" cy="{right_y}" r="12" fill="{right_dot_color}" opacity="0.9" stroke="rgba(139,92,246,0.5)" stroke-width="1"/>
                            
                            <!-- Quarter labels with proper positioning -->
                            <text x="50" y="{left_y + (20 if left_y < 60 else -15)}" text-anchor="middle" font-size="12" fill="#475569" font-weight="600">{sentiment_prev:.2f}</text>
                            <text x="750" y="{right_y + (20 if right_y < 60 else -15)}" text-anchor="middle" font-size="12" fill="#475569" font-weight="600">{sentiment_current:.2f}</text>
                            
                            <!-- Flow direction arrow -->
                            <text x="400" y="15" text-anchor="middle" font-size="10" fill="{flow_color}" font-weight="600" opacity="0.8">
                                {prev_quarter} → {selected_quarter}
                            </text>
                        </svg>
                        
                        {challenge_bubbles_html}
                    </div>
                    
                    <div class="metrics-glass-bar">
                        <div class="metrics-section">
                            <div class="metric-item">
                                <span class="metric-label">Sentiment Trend:</span>
                                <span class="metric-value" style="color: {flow_color};">
                                    {sentiment_change:+.2f} {'↗' if flow_type == 'rising' else '↘' if flow_type == 'declining' else ''}
                                </span>
                            </div>
                            <div class="discussions-pill" style="background: linear-gradient(135deg, {flow_color}, {flow_color}CC);">
                                {volume_change_pct:+.0f}% discussions
                            </div>
                            <div class="status-badge" style="background: linear-gradient(135deg, {flow_color}, {flow_color}DD);">
                                <span>{'🚀 Strong Growth' if flow_type == 'rising' and volume_change_pct > 30 else '🤝 Support Needed' if flow_type == 'declining' else '📈 Growing' if flow_type == 'rising' else '⚠️ Monitor Closely' if flow_type == 'declining' else '➡️ Stable'}</span>
                            </div>
                        </div>
                        <div class="metrics-section">
                            <div class="metric-item">
                                <span class="metric-label">Active Challenges:</span>
                                <span class="challenges-count">{len(topic_challenges)}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="challenge-insights">
                        <div class="insights-header">
                            <span>⚡</span>
                            <span>Top Challenges</span>
                        </div>
                        <div class="challenge-list">
                            {challenge_items_html if challenge_items_html else '<div class="challenge-item low"><div class="challenge-severity challenge-low"></div><div class="challenge-content"><div class="challenge-text">No significant challenges detected</div><div class="challenge-stats">Smooth waters</div></div></div>'}
                        </div>
                    </div>
                </div>
                """
        
        # Create the full HTML component
        return f"""
        <!DOCTYPE html>
        <html style="margin: 0; padding: 0;">
        <head>
            <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'DM Sans', sans-serif;
                    background: #ffffff;
                    padding: 20px;
                    color: #1e293b;
                    min-height: 100vh;
                }}
                
                .river-container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                
                /* CUSTOM PAGE HEADER */
                .page-header {{
                    background: rgba(248, 250, 252, 0.8);
                    backdrop-filter: blur(15px);
                    border: 1px solid rgba(226, 232, 240, 0.8);
                    border-radius: 20px;
                    padding: 4px;
                    margin-bottom: 4px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                    animation: slideInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                }}
                
                .page-title {{
                    font-size: 1.75rem;
                    font-weight: 700;
                    color: #1e293b;
                    margin-bottom: 4px;
                    font-family: 'DM Sans', sans-serif;
                    line-height: 1.2;
                }}
                
                .trend-summary {{
                    display: inline-flex;
                    align-items: center;
                    gap: 12px;
                    padding: 12px 20px;
                    border-radius: 12px;
                    border: 1px solid rgba(226, 232, 240, 0.6);
                    font-size: 1rem;
                    font-weight: 600;
                    backdrop-filter: blur(10px);
                }}
                
                .trend-summary.baseline {{
                    background: linear-gradient(135deg, rgba(148, 163, 184, 0.1), rgba(148, 163, 184, 0.05));
                    border-color: rgba(148, 163, 184, 0.4);
                    color: #64748b;
                }}
                
                .trend-emoji {{
                    font-size: 1.25rem;
                }}
                
                .trend-text {{
                    font-family: 'DM Sans', sans-serif;
                }}
                
                @keyframes slideInUp {{
                    from {{
                        transform: translateY(30px);
                        opacity: 0;
                    }}
                    to {{
                        transform: translateY(0);
                        opacity: 1;
                    }}
                }}
                
                /* TIMELINE QUARTERS */
                .timeline-quarters {{
                    display: flex;
                    justify-content: space-between;
                    margin-top: 24px;        /* Add this line */
                    margin-bottom: 32px;
                    padding: 0 40px;
                    position: relative;
                }}
                
                .timeline-quarters::before {{
                    content: '';
                    position: absolute;
                    top: 50%;
                    left: 40px;
                    right: 40px;
                    height: 2px;
                    background: linear-gradient(90deg, #e2e8f0, #cbd5e1);
                    transform: translateY(-50%);
                    z-index: 1;
                }}
                
                .quarter-marker {{
                    text-align: center;
                    position: relative;
                    z-index: 2;
                }}
                
                .quarter-label {{
                    font-size: 0.875rem;
                    font-weight: 600;
                    color: #475569;
                    margin-bottom: 8px;
                }}
                
                .quarter-dot {{
                    width: 12px;
                    height: 12px;
                    background: #cbd5e1;
                    border-radius: 50%;
                    margin: 0 auto;
                }}
                
                .quarter-marker.current .quarter-dot {{
                    background: #6366f1;
                    box-shadow: 0 0 10px rgba(99, 102, 241, 0.3);
                    animation: pulse 2s ease-in-out infinite;
                }}
                
                .quarter-marker.current .quarter-label {{
                    color: #6366f1;
                    font-weight: 700;
                }}
                
                @keyframes pulse {{
                    0%, 100% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.2); }}
                }}
                
                /* CHALLENGE LEGEND */
                .challenge-legend {{
                    background: rgba(248,250,252,0.8);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(226,232,240,0.8);
                    border-radius: 16px;
                    padding: 16px;
                    margin-bottom: 24px;
                    display: flex;
                    justify-content: center;
                    gap: 24px;
                    flex-wrap: wrap;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }}
                
                .legend-item {{
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 0.875rem;
                    color: #475569;
                    font-weight: 500;
                }}
                
                .legend-icon {{
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                }}
                
                .challenge-high {{ background: #ef4444; }}
                .challenge-medium {{ background: #f97316; }}
                .challenge-low {{ background: #22c55e; }}
                
                /* TOPIC FLOW CONTAINERS */
                .topic-flow {{
                    background: rgba(248, 250, 252, 0.8);
                    backdrop-filter: blur(15px);
                    border: 1px solid rgba(226, 232, 240, 0.8);
                    border-radius: 20px;
                    padding: 24px;
                    margin-bottom: 32px;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                }}
                
                .topic-flow:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
                }}
                
                .topic-header {{
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 20px;
                }}
                
                .topic-emoji {{
                    font-size: 1.5rem;
                    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
                }}
                
                .topic-name {{
                    font-size: 1.25rem;
                    font-weight: 600;
                    color: #1e293b;
                }}
                
                .flow-svg-container {{
                    position: relative;
                    margin-bottom: 20px;
                }}
                
                .flow-svg {{
                    width: 100%;
                    height: 120px;
                }}
                
                /* CHALLENGE BUBBLES */
                .challenge-bubble {{
                    position: absolute;
                    width: 20px;
                    height: 20px;
                    background: rgba(255,255,255,0.95);
                    border: 2px solid;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 0.75rem;
                    font-weight: 700;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    z-index: 10;
                    transform: translate(-50%, -50%);
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                
                .challenge-bubble:hover {{
                    transform: translate(-50%, -50%) scale(1.5);
                    z-index: 20;
                }}
                
                .challenge-bubble.high {{
                    border-color: #ef4444;
                    color: #ef4444;
                    animation: bubble-urgent 1.5s ease-in-out infinite;
                }}
                
                .challenge-bubble.medium {{
                    border-color: #f97316;
                    color: #f97316;
                    animation: bubble-moderate 2s ease-in-out infinite;
                }}
                
                .challenge-bubble.low {{
                    border-color: #22c55e;
                    color: #22c55e;
                    animation: bubble-calm 3s ease-in-out infinite;
                }}
                
                @keyframes bubble-urgent {{
                    0%, 100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }}
                    50% {{ box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }}
                }}
                
                @keyframes bubble-moderate {{
                    0%, 100% {{ box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.3); }}
                    50% {{ box-shadow: 0 0 0 4px rgba(249, 115, 22, 0); }}
                }}
                
                @keyframes bubble-calm {{
                    0%, 100% {{ box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.2); }}
                    50% {{ box-shadow: 0 0 0 3px rgba(34, 197, 94, 0); }}
                }}
                
                /* ENHANCED CHALLENGE TOOLTIPS - Better handling for long text */
                .challenge-tooltip {{
                    position: absolute;
                    background: rgba(0,0,0,0.9);
                    color: white;
                    padding: 10px 14px;
                    border-radius: 8px;
                    font-size: 0.8rem;
                    opacity: 0;
                    pointer-events: none;
                    transition: all 0.3s ease;
                    z-index: 30;
                    max-width: 320px;
                    white-space: normal;
                    line-height: 1.4;
                    transform: translate(-50%, -100%);
                    margin-top: -10px;
                    word-wrap: break-word;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                }}
                
                .challenge-bubble:hover + .challenge-tooltip {{
                    opacity: 1;
                }}
                
                /* CHALLENGE RAPIDS */
                .rapid {{
                    fill: none;
                    stroke-width: 2;
                    animation: rapid-flow 2s ease-in-out infinite;
                }}
                
                @keyframes rapid-flow {{
                    0%, 100% {{ stroke-dasharray: 4,4; stroke-dashoffset: 0; }}
                    50% {{ stroke-dashoffset: 8; }}
                }}
                
                /* SINGLE GLASS METRICS BAR */
                .metrics-glass-bar {{
                    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(248,250,252,0.8));
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(226,232,240,0.6);
                    border-radius: 16px;
                    padding: 16px 20px;
                    margin-bottom: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 20px;
                    flex-wrap: wrap;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
                }}
                
                .metrics-section {{
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    flex-wrap: wrap;
                }}
                
                .metric-item {{
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    font-size: 0.875rem;
                }}
                
                .metric-label {{
                    color: #64748b;
                    font-weight: 500;
                }}
                
                .metric-value {{
                    font-weight: 700;
                    font-size: 0.9rem;
                }}
                
                .discussions-pill {{
                    color: white;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                }}
                
                .status-badge {{
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                    color: white;
                }}
                
                .challenges-count {{
                    color: #6366f1;
                    font-weight: 700;
                }}
                
                /* CHALLENGE INSIGHTS */
                .challenge-insights {{
                    background: linear-gradient(135deg, rgba(99,102,241,0.05), rgba(139,92,246,0.05));
                    border: 1px solid rgba(99,102,241,0.1);
                    border-radius: 12px;
                    padding: 16px;
                }}
                
                .insights-header {{
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: 12px;
                    font-size: 0.875rem;
                    font-weight: 600;
                    color: #6366f1;
                }}
                
                .challenge-list {{
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }}
                
                .challenge-item {{
                    display: flex;
                    align-items: flex-start;
                    gap: 12px;
                    padding: 10px;
                    background: rgba(255,255,255,0.8);
                    border-radius: 8px;
                    border-left: 3px solid;
                }}
                
                .challenge-item.high {{
                    border-left-color: #ef4444;
                }}
                
                .challenge-item.medium {{
                    border-left-color: #f97316;
                }}
                
                .challenge-item.low {{
                    border-left-color: #22c55e;
                }}
                
                .challenge-severity {{
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    flex-shrink: 0;
                    margin-top: 4px;
                }}
                
                .challenge-content {{
                    flex: 1;
                }}
                
                .challenge-text {{
                    font-size: 0.85rem;
                    color: #1e293b;
                    line-height: 1.5;
                    margin-bottom: 4px;
                    word-wrap: break-word;
                }}
                
                .challenge-stats {{
                    font-size: 0.75rem;
                    color: #64748b;
                }}
                
                /* MOBILE RESPONSIVE */
                @media (max-width: 768px) {{
                    .page-header {{
                        padding: 24px;
                    }}
                    
                    .page-title {{
                        font-size: 1.5rem;
                    }}
                    
                    .timeline-quarters {{
                        padding: 0 20px;
                    }}
                    
                    .topic-flow {{
                        padding: 16px;
                    }}
                    
                    .metrics-glass-bar {{
                        flex-direction: column;
                        gap: 12px;
                        align-items: flex-start;
                    }}
                    
                    .metrics-section {{
                        width: 100%;
                        justify-content: space-between;
                    }}
                    
                    .challenge-legend {{
                        flex-direction: column;
                        gap: 12px;
                        align-items: center;
                    }}
                    
                    .challenge-tooltip {{
                        max-width: 280px;
                        font-size: 0.75rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="river-container">
                <!-- INTEGRATED HEADER -->
                {header_section}
                
                <!-- Timeline -->
                <div class="timeline-quarters">
                    {'<div class="quarter-marker"><div class="quarter-label">' + prev_quarter + '</div><div class="quarter-dot"></div></div>' if not is_first_quarter else ''}
                    <div class="quarter-marker current">
                        <div class="quarter-label">{selected_quarter}</div>
                        <div class="quarter-dot"></div>
                    </div>
                </div>
                
                <!-- Challenge Legend -->
                <div class="challenge-legend">
                    <div class="legend-item">
                        <span><strong>Challenges   </strong></span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon challenge-high"></div>
                        <span>High Intensity (>2%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon challenge-medium"></div>
                        <span>Moderate (1-2%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon challenge-low"></div>
                        <span>Manageable (<1%)</span>
                    </div>
                </div>
                
                <!-- River Topics -->
                {river_topics_html}
            </div>
            
            <script>
                // Animate SVG paths on load
                document.addEventListener('DOMContentLoaded', function() {{
                    const paths = document.querySelectorAll('.flow-svg path:not(.rapid)');
                    
                    paths.forEach(path => {{
                        const length = path.getTotalLength();
                        if (length) {{
                            path.style.strokeDasharray = length;
                            path.style.strokeDashoffset = length;
                            
                            setTimeout(() => {{
                                path.style.transition = 'stroke-dashoffset 2s ease-in-out';
                                path.style.strokeDashoffset = 0;
                            }}, 300);
                        }}
                    }});
                    
                    // Animate challenge bubbles
                    setTimeout(() => {{
                        document.querySelectorAll('.challenge-bubble').forEach((bubble, index) => {{
                            bubble.style.opacity = 0;
                            bubble.style.transform = 'translate(-50%, -50%) scale(0)';
                            
                            setTimeout(() => {{
                                bubble.style.transition = 'all 0.4s ease';
                                bubble.style.opacity = 1;
                                bubble.style.transform = 'translate(-50%, -50%) scale(1)';
                            }}, index * 200 + 800);
                        }});
                    }}, 100);
                }});
                
                // Enhanced hover interactions
                document.querySelectorAll('.challenge-bubble').forEach(bubble => {{
                    bubble.addEventListener('mouseenter', function() {{
                        document.querySelectorAll('.rapid').forEach(rapid => {{
                            rapid.style.animationPlayState = 'paused';
                        }});
                    }});
                    
                    bubble.addEventListener('mouseleave', function() {{
                        document.querySelectorAll('.rapid').forEach(rapid => {{
                            rapid.style.animationPlayState = 'running';
                        }});
                    }});
                }});
            </script>
        </body>
        </html>
        """
    
    # Render the River Flow component with 4800px height
    st.components.v1.html(create_river_flow_html(), height=4800, width=None)

    # Footer
    st.markdown("""
    <div class="footer-text">
        Powered by Terramare ᛘ𓇳     ©2025
    </div>
    """, unsafe_allow_html=True)

# Execute app
if __name__ == "__main__":
    run()