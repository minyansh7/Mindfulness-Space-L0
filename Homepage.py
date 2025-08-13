import streamlit as st
import datetime
import importlib
import base64

# Function to encode image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# Page configuration with custom icon
st.set_page_config(
    page_title="MindSpace OS",
    layout="wide",
    page_icon="assets/narrative_web_clean.png",  # Using your existing narrative_web.png
    initial_sidebar_state="collapsed"
)

# Get time
current_time = datetime.datetime.now().hour

# Dynamic color scheme with text color
def get_time_colors(hour):
    if 5 <= hour < 8:
        return {
            'primary': '#FF6B6B',
            'secondary': '#FFB347',
            'bg_gradient': 'linear-gradient(135deg, #FF6B6B 0%, #FFB347 100%)',
            'ripple_color': 'rgba(255, 107, 107, 0.3)',
            'text_color': 'black'
        }
    elif 8 <= hour < 12:
        return {
            'primary': '#4ECDC4',
            'secondary': '#45B7D1',
            'bg_gradient': 'linear-gradient(135deg, #4ECDC4 0%, #45B7D1 100%)',
            'ripple_color': 'rgba(78, 205, 196, 0.3)',
            'text_color': 'black'
        }
    elif 12 <= hour < 17:
        return {
            'primary': '#45B7D1',
            'secondary': '#96CEB4',
            'bg_gradient': 'linear-gradient(135deg, #45B7D1 0%, #96CEB4 100%)',
            'ripple_color': 'rgba(69, 183, 209, 0.3)',
            'text_color': 'black'
        }
    elif 17 <= hour < 20:
        return {
            'primary': '#A29BFE',
            'secondary': '#FD79A8',
            'bg_gradient': 'linear-gradient(135deg, #A29BFE 0%, #FD79A8 100%)',
            'ripple_color': 'rgba(162, 155, 254, 0.3)',
            'text_color': 'white'
        }
    else:
        return {
            'primary': '#6C5CE7',
            'secondary': '#2D3436',
            'bg_gradient': 'linear-gradient(135deg, #6C5CE7 0%, #2D3436 100%)',
            'ripple_color': 'rgba(108, 92, 231, 0.3)',
            'text_color': 'white'
        }

colors = get_time_colors(current_time)

# Get base64 encoded image
image_base64 = get_base64_image("assets/narrative_web_clean.png")

# Inject CSS
st.markdown(f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .stApp {{
        background: {colors['bg_gradient']};
        color: {colors['text_color']};
        font-family: 'Inter', sans-serif;
    }}

    .main .block-container {{
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }}

    .nav-bar {{
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 0rem 0rem;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 500;
        position: relative;
        top: 0;
        z-index: 999;
    }}

    .nav-right {{
        display: flex;
        gap: 1.5rem;
    }}

    .nav-bar a {{
        position: relative;
        display: inline-block;
        color: transparent;
        background: linear-gradient(45deg, white, rgba(255, 255, 255, 0.8));
        -webkit-background-clip: text;
        background-clip: text;
        text-decoration: none;
        padding: 0.2rem 0;
        transition: color 0.3s ease;
    }}

    .nav-bar a::before {{
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 0%;
        height: 6px;
        background-color: #FFFF00;
        transition: width 0.4s ease-in-out;
        z-index: -1;
    }}

    .nav-bar a:hover::before {{
        width: 100%;
    }}

    .hero-title {{
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, white, rgba(255, 255, 255, 0.8));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
    }}

    .hero-subtitle {{
        font-size: 1.2rem;
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        margin: 0 0 0.8rem 0;
        line-height: 1.6;
    }}

    .meditation-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 300px;
        margin: 0.5rem 0;
    }}

    .meditation-circle {{
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: {colors['ripple_color']};
        animation: pulse 3s ease-in-out infinite;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
    }}

    .meditation-circle::before {{
        content: '';
        position: absolute;
        top: -20px;
        left: -20px;
        right: -20px;
        bottom: -20px;
        border: 2px solid {colors['primary']};
        border-radius: 50%;
        animation: ripple 2s ease-out infinite;
    }}

    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); opacity: 0.8; }}
        50% {{ transform: scale(1.1); opacity: 1; }}
    }}

    @keyframes ripple {{
        0% {{ transform: scale(1); opacity: 1; }}
        100% {{ transform: scale(1.5); opacity: 0; }}
    }}

    .theme-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);  /* Changed from 5 to 4 */
        gap: 1.8rem;  /* Increased from 1.2rem */
        padding: 0 3rem;  /* Increased from 2rem */
        margin: 1rem 0;  /* Increased from 0.5rem */
    }}

    /* Responsive breakpoints */
    @media (max-width: 1200px) {{
        .theme-grid {{
            grid-template-columns: repeat(2, 1fr);  /* Changed from 3 to 2 */
            gap: 2rem;  /* Increased gap */
            padding: 0 2rem;
        }}
    }}

    @media (max-width: 768px) {{
        .theme-grid {{
            grid-template-columns: 1fr;
            gap: 1.5rem;  /* Increased gap */
            padding: 0 1.5rem;
        }}
    }}

    .theme-card {{
        position: relative;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;  /* Increased from 12px */
        padding: 2rem;  /* Increased from 1.6rem */
        text-align: center;
        transition: transform 0.05s ease, background 0.2s ease;
        cursor: pointer;
        overflow: hidden;
        min-height: 140px;  /* Increased from 120px */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}

    .theme-card:hover {{
        transform: translateY(-6px) scale(1.03);
        background: rgba(255, 255, 255, 0.22);
        box-shadow: 0 20px 40px rgba(255, 255, 255, 0.25);
    }}

    .theme-title {{
        font-size: 1.4rem;  /* Increased from 1.2rem */
        font-weight: 600;
        margin-bottom: 0.8rem;  /* Increased from 0.5rem */
        line-height: 1.3;  /* Increased from 1.2 */
    }}

    .theme-card a, .theme-card a:visited {{
        color: {colors['text_color']} !important;
        text-decoration: none;
        display: block;
        width: 100%;
        height: 100%;
    }}

    .theme-description {{
        line-height: 1.5;
    }}

    .theme-title, .theme-description {{
        background: linear-gradient(45deg, white, rgba(255, 255, 255, 0.8));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
    }}

    .theme-card::after {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.4);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: none;
        opacity: 0;
    }}

    .theme-card:hover::after {{
        animation: organicBulb 1.1s ease-out forwards;
    }}

    @keyframes organicBulb {{
        0% {{
            width: 0;
            height: 0;
            opacity: 1;
            background: rgba(255, 255, 255, 0.4);
            border-radius: 50%;
            transform: translate(-50%, -50%) rotate(0deg);
        }}
        25% {{
            width: 64px;
            height: 48px;
            opacity: 0.8;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
            transform: translate(-50%, -50%) rotate(15deg);
        }}
        50% {{
            width: 128px;
            height: 96px;
            opacity: 0.5;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 30% 70% 70% 30% / 30% 40% 60% 70%;
            transform: translate(-50%, -50%) rotate(30deg);
        }}
        75% {{
            width: 192px;
            height: 144px;
            opacity: 0.2;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 70% 30% 40% 60% / 40% 70% 30% 60%;
            transform: translate(-50%, -50%) rotate(45deg);
        }}
        100% {{
            width: 256px;
            height: 192px;
            opacity: 0;
            background: rgba(255, 255, 255, 0);
            border-radius: 40% 60% 60% 40% / 60% 40% 60% 40%;
            transform: translate(-50%, -50%) rotate(60deg);
        }}
    }}
</style>
""", unsafe_allow_html=True)

# Navigation header
st.markdown(f"""
<div class="nav-bar">
    <div class="nav-right">
        <a href="https://terramare.substack.com/subscribe?params=%5Bobject%20Object%5D">Join Newsletter</a>
        <a href="mailto:contact@meditation-pain-map.com">Contact</a>
        <a href="https://www.linkedin.com/in/minyanshi/" target="_blank">LinkedIn</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-title">🧭 MindSpace OS</div>
<div class="hero-subtitle">
    Explore interlinked visualizations revealing emotions felt, themes emerged & intertwined, and challenges shared 
    across online meditation communities.
</div>
""", unsafe_allow_html=True)

# Meditation Circle
if image_base64:
    st.markdown(f"""
    <div class="meditation-container">
        <div class="meditation-circle">
            <img src="data:image/png;base64,{image_base64}" alt="MindSpace Icon" style="width: 80px; height: 80px; object-fit: contain; border-radius: 50%;">
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Fallback to emoji if image not found
    st.markdown(f"""
    <div class="meditation-container">
        <div class="meditation-circle">🧘‍♀️</div>
    </div>
    """, unsafe_allow_html=True)

# Theme Cards
st.markdown(f"""
<div class="theme-grid">
    <a href="/Emotion_Pulse_v" style="text-decoration: none; display: block;">
        <div class="theme-card">
            <div class="theme-title">Emotional Pulse</div>
        </div>
    </a>
    <a href="/Meditation_Weather_Report" style="text-decoration: none; display: block;">
        <div class="theme-card">
            <div class="theme-title">Weather Report</div>
        </div>
    </a>
    <a href="/Main_Topics_Sankey" style="text-decoration: none; display: block;">
        <div class="theme-card">
            <div class="theme-title">Main Narratives</div>
        </div>
    </a>
    <a href="/Cocurrence_Mapping_Over_Time" style="text-decoration: none; display: block;">
        <div class="theme-card">
            <div class="theme-title">Narrative Connections</div>
        </div>
    </a>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown(f"""
<br>
<div style="
    text-align: center;
    font-size: 1rem;
    font-weight: 600;
    background: linear-gradient(45deg, white, rgba(255, 255, 255, 0.8));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent;
">
    Powered By Terramare ᛘ𓇳 ©2025
</div>
""", unsafe_allow_html=True)

# Routing logic
query_params = st.query_params
page = query_params.get("page", "home")

if page == "Emotion_Pulse_v":
    module = importlib.import_module("pages.0_Emotion_Pulse")
    module.run()
elif page == "Meditation_Weather_report":
    module = importlib.import_module("pages.1_Meditation_Weather_Report")
    module.run()
elif page == "Main_Topics_Sankey":
    module = importlib.import_module("pages.2_Main_Topics_Sankey")
    module.run()
elif page == "Cocurrence_Mapping_Over_Time":
    module = importlib.import_module("pages.3_Cocurrence_Mapping_Over_Time")
    module.run()