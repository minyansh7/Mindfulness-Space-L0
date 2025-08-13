import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import json
import plotly.graph_objects as go
import pandas as pd
import streamlit.components.v1 as components
import json

st.set_page_config(
    page_title="🌊 The Emotional Pulse of the Meditation Landscape",
    layout="wide"
)

# --- Text color logic for hover popup ---
def ideal_text_color(bg_hex: str) -> str:
    bg_hex = bg_hex.lstrip("#")
    r, g, b = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 186 else 'white'

def run():
    # --- Styled CSS for footer and plot wrapper with responsive design ---
    st.markdown("""
    <style>
    /* Global responsive styles */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: none;
    }
    
    .footer-text {
        text-align: center;
        font-size: 1rem;
        font-weight: 600;
        color: #666;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #eee;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    @supports not (-webkit-background-clip: text) {
        .footer-text {
            color: #667eea !important;
            background: none !important;
        }
    }
    
    /* Responsive plot container */
    .plot-container {
        width: 100%;
        margin: auto;
        position: relative;
    }
    
    /* Responsive text and layout */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.5rem;
        font-weight: 500;
        margin-top: 0;
    }
    
    .description {
        font-size: 1rem;
        margin: 0.05rem auto 0;
        color: #888;
        max-width: 1400px;
        width: 95%;
        line-height: 1.5;
        text-wrap: pretty;
    }
    
    .annotation-container {
        text-align: left;
        margin: 0.1rem 0 0.1rem 0;
        margin-left: 2.5vw;
        margin-right: 2.5vw;
    }
    
    /* Mobile responsive breakpoints */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem !important;
        }
        .sub-title {
            font-size: 1.2rem !important;
        }
        .description {
            font-size: 0.9rem !important;
        }
        .annotation-container .flex-container {
            flex-direction: column !important;
            gap: 20px !important;
        }
        .footer-text {
            font-size: 0.9rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.5rem !important;
        }
        .sub-title {
            font-size: 1rem !important;
        }
        .description {
            font-size: 0.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
    <h1 class="main-title">🌊 Emotion Pulse</h1>
    <h3 class="sub-title">What people feel</h3>
    <p class="description">
        This emotional map reveals what emotions peole express through meditation practices, drawn from thousands of reddit posts and comments shared between January 2024 and June 2025.
    </p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: left; padding: 1rem;">
    <p style="
        font-size: 1rem;
        margin: 0.01rem auto 0;
        color: #444;
        max-width: 1400px;
        width: 100%;
        line-height: 1.5;
    ">
        Conducted advanced NLP processing using
        <strong>Google Research's <a href="https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/" 
        target="_blank" style="color: #4285f4; text-decoration: none; border-bottom: 1px dotted #4285f4;">GoEmotions</a> Model</strong> 
        with 27-category emotion classification system and the Fine-grained model trained on 58,000 human-annotated texts, to decode emotional contexts among Reddit's post and comments on Meditation. 
    </p>
</div>
""", unsafe_allow_html=True)
    
# Poetic invitation section
    st.markdown("""
    <div class="annotation-container">
        <div style="display: flex; justify-content: center;">
            <div style="max-width: 500px; text-align: center;">
                <p style="font-size: 16px; color: #718096; margin: 0; line-height: 1.8; font-weight: 300; opacity: 0.9;">
                    <strong>Each point</strong> marks a message shared on Reddit about meditation.<br>
                    <strong>Color</strong> reflects the emotional pulses uncovered for each.
                       <br> 
                </p>
            </div>
        </div>
</div>
""", unsafe_allow_html=True)


    
    @st.cache_data
    def load_emotion_clusters():
        return pd.read_parquet("precomputed/emotion_clusters.parquet")

    emotion_df = load_emotion_clusters()
    df = emotion_df.copy()

    cluster_colors = {
        'Reflective Caring': '#ff5e78',
        'Soothing Empathy': '#00c49a',
        'Tender Uncertainty': '#6a5acd',
        'Melancholic Confusion': '#9a32cd',
        'Anxious Concern': '#ffc300'
    }

    cluster_styles = {
        'Reflective Caring': {'symbol': 'circle', 'size': 8},
        'Soothing Empathy': {'symbol': 'diamond', 'size': 8},
        'Tender Uncertainty': {'symbol': 'triangle-nw', 'size': 9},
        'Melancholic Confusion': {'symbol': 'star-square', 'size': 8},
        'Anxious Concern': {'symbol': 'star-triangle-up', 'size': 8}
    }

    # Prepare data for JavaScript - keep Top emotions and Post/Comment, remove Emotional Pulse
    plot_data = []
    for _, row in df.iterrows():
        # Extract only "Top Emotions:" and "Post/Comment:" lines
        original_text = str(row['hover_text'])
        lines = original_text.split('\n')
        
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            # Keep lines that contain "top emotions" or "post" or "comment"
            if (line_lower.startswith('top emotions') or 
                'post' in line_lower or 
                'comment' in line_lower):
                filtered_lines.append(line_stripped)
        
        # If no lines found, show original text for debugging
        if not filtered_lines:
            simplified_text = original_text  # Show full text to see what's available
        else:
            simplified_text = '\n'.join(filtered_lines)
        
        plot_data.append({
            'x': float(row['umap_x']),
            'y': float(row['umap_y']),
            'cluster': row['archetype_label'],
            'hover_text': simplified_text,
            'color': cluster_colors[row['archetype_label']]
        })

    # Calculate centroids for labels
    centroids = df.groupby('archetype_label')[['umap_x', 'umap_y']].mean()
    offsets = {
        'Reflective Caring': [3.9, 1.2],
        'Soothing Empathy': [1.5, 4],
        'Tender Uncertainty': [-4.1, -2.5],
        'Melancholic Confusion': [0.5, -1.5],
        'Anxious Concern': [2.4, -0.3]
    }
    
    label_data = []
    for archetype, (x, y) in centroids.iterrows():
        dx, dy = offsets.get(archetype, [0, 0])
        label_data.append({
            'x': x + dx,
            'y': y + dy,
            'text': archetype,
            'color': cluster_colors[archetype]
        })

    # Calculate plot bounds
    y_min = df['umap_y'].min()
    y_max = df['umap_y'].max()
    y_range = y_max - y_min
    
    # Create the HTML with embedded plot and responsive functionality
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ 
                margin: 0; 
                padding: 0; 
                font-family: Arial, sans-serif; 
                background: white;
                overflow-x: hidden;
            }}
            
            .container {{
                position: relative;
                width: 100%;
                height: 100vh;
                min-height: 600px;
            }}
            
            #plotDiv {{ 
                width: 100%; 
                height: 100%; 
            }}
            
            #hoverBox {{
                position: absolute;
                top: 20px;
                left: 50px;
                right: 50px;
                background: rgba(255, 255, 255, 0.95);
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                font-size: 10px;
                line-height: 1;
                box-shadow: 0 4px 4px rgba(0,0,0,0.15);
                z-index: 1000;
                backdrop-filter: blur(2px);
                display: none;
                max-height: 200px;
                overflow-y: auto;
            }}
            
            #hoverBox.visible {{ 
                display: block; 
            }}
            
            #hoverBox h4 {{ 
                margin: 0 0 4px 0; 
                font-size: 10px; 
                font-weight: bold; 
                border-bottom: 1px solid #eee;
                padding-bottom: 4px;
            }}
            
            #hoverContent {{
                font-size: 10px;
                line-height: 1.1;
            }}
            
            /* Responsive hover box */
            @media (max-width: 1024px) {{
                #hoverBox {{
                    left: 100px;
                    right: 100px;
                }}
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    height: 70vh;
                    min-height: 500px;
                }}
                
                #hoverBox {{
                    left: 20px;
                    right: 20px;
                    top: 10px;
                    padding: 8px;
                    font-size: 8px;
                    max-height: 150px;
                }}
                
                #hoverBox h4 {{
                    font-size: 10px;
                }}
            }}
            
            @media (max-width: 480px) {{
                .container {{
                    height: 60vh;
                    min-height: 400px;
                }}
                
                #hoverBox {{
                    left: 10px;
                    right: 10px;
                    padding: 10px;
                    font-size: 10px;
                    max-height: 120px;
                }}
                
                #hoverBox h4 {{
                    font-size: 10px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div id="hoverBox">
                <h4 id="hoverTitle"></h4>
                <div id="hoverContent"></div>
            </div>
            <div id="plotDiv"></div>
        </div>
        
        <script>
            const plotData = {json.dumps(plot_data)};
            const labelData = {json.dumps(label_data)};
            const clusterColors = {json.dumps(cluster_colors)};
            const clusterStyles = {json.dumps(cluster_styles)};
            
            // Fixed index for consistent default point selection
            const DEFAULT_POINT_INDEX = 33;
            
            let plotDiv = document.getElementById('plotDiv');
            let currentLayout = null;
            let currentTraces = null;
            
            // Function to calculate hover box height
            function getHoverBoxHeight() {{
                const containerWidth = window.innerWidth;
                
                if (containerWidth <= 480) {{
                    return 120; // max-height from CSS
                }} else if (containerWidth <= 768) {{
                    return 150; // max-height from CSS
                }} else {{
                    return 200; // max-height from CSS
                }}
            }}
            
            // Function to calculate responsive dimensions and layout
            function getResponsiveLayout() {{
                const containerWidth = window.innerWidth;
                const containerHeight = window.innerHeight;
                
                // Calculate responsive margins with reduced bottom margin
                let leftMargin, rightMargin, topMargin, bottomMargin;
                
                if (containerWidth <= 480) {{
                    leftMargin = rightMargin = 20;
                    topMargin = 28;
                    bottomMargin = 5; // Reduced from 10
                }} else if (containerWidth <= 768) {{
                    leftMargin = rightMargin = 50;
                    topMargin = 38;
                    bottomMargin = 8; // Reduced from 15
                }} else if (containerWidth <= 1024) {{
                    leftMargin = rightMargin = 100;
                    topMargin = 50;
                    bottomMargin = 10; // Reduced from 20
                }} else {{
                    leftMargin = rightMargin = 200;
                    topMargin = 60;
                    bottomMargin = 10; // Reduced from 20
                }}
                
                // Calculate responsive height (1.1 times the original)
                let plotHeight;
                if (containerWidth <= 480) {{
                    plotHeight = Math.max(330, containerHeight * 0.495); // 1.1 * (300, 0.45)
                }} else if (containerWidth <= 768) {{
                    plotHeight = Math.max(412.5, containerHeight * 0.5775); // 1.1 * (375, 0.525)
                }} else {{
                    plotHeight = Math.max(495, containerHeight * 0.78375); // 1.1 * (450, 0.7125)
                }}
                
                // Calculate responsive font size for labels
                const labelFontSize = containerWidth <= 768 ? 12 : 16;
                
                return {{
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white',
                    font: {{ color: 'black' }},
                    showlegend: false,
                    xaxis: {{
                        showgrid: false,
                        zeroline: false,
                        showticklabels: false,
                        scaleanchor: 'y',
                        scaleratio: 1
                    }},
                    yaxis: {{
                        showgrid: false,
                        zeroline: false,
                        showticklabels: false,
                        range: [{y_min - 0.01 * y_range}, {y_max + 0.13 * y_range}]
                    }},
                    hovermode: 'closest',
                    hoverdistance: 100,
                    autosize: true,
                    height: plotHeight,
                    margin: {{ 
                        t: topMargin, 
                        b: bottomMargin, 
                        l: leftMargin, 
                        r: rightMargin 
                    }},
                    responsive: true,
                    labelFontSize: labelFontSize
                }};
            }}
            
            // Function to create responsive traces
            function getResponsiveTraces(layout) {{
                // Group data by cluster
                const clusters = Object.keys(clusterColors);
                const traces = [];
                let allDataPoints = []; // To store all points for hover events
                
                clusters.forEach(cluster => {{
                    const clusterPoints = plotData.filter(d => d.cluster === cluster);
                    const style = clusterStyles[cluster];
                    
                    // Add to all data points for hover events
                    allDataPoints = allDataPoints.concat(clusterPoints);
                    
                    // Responsive marker size
                    const markerSize = window.innerWidth <= 768 ? style.size * 0.8 : style.size;
                    
                    traces.push({{
                        x: clusterPoints.map(d => d.x),
                        y: clusterPoints.map(d => d.y),
                        mode: 'markers',
                        name: cluster,
                        marker: {{
                            color: clusterColors[cluster],
                            opacity: 0.6,
                            symbol: style.symbol,
                            size: markerSize
                        }},
                        customdata: clusterPoints.map(d => [d.hover_text, d.cluster, d.color]),
                        hoverinfo: 'none',
                        showlegend: false
                    }});
                }});
                
                // Add cluster labels with responsive font size
                labelData.forEach(label => {{
                    traces.push({{
                        x: [label.x],
                        y: [label.y],
                        mode: 'text',
                        text: [`<b>${{label.text}}</b>`],
                        textfont: {{ 
                            size: layout.labelFontSize, 
                            color: label.color, 
                            family: 'sans-serif' 
                        }},
                        showlegend: false,
                        hoverinfo: 'none'
                    }});
                }});
                
                window.allDataPoints = allDataPoints; // Store globally for hover events
                return traces;
            }}
            
            // Function to create the plot
            function createPlot() {{
                currentLayout = getResponsiveLayout();
                currentTraces = getResponsiveTraces(currentLayout);
                
                const config = {{
                    displayModeBar: true,
                    toImageButtonOptions: {{
                        format: 'png',
                        filename: 'emotion_pulse',
                        height: currentLayout.height,
                        width: Math.min(1200, window.innerWidth),
                        scale: 2
                    }},
                    modeBarButtonsToRemove: [
                        'pan2d', 
                        'lasso2d', 
                        'select2d',
                        'zoom2d',
                        'zoomIn2d', 
                        'zoomOut2d',
                        'autoScale2d',
                        'resetScale2d'
                    ],
                    scrollZoom: false,
                    doubleClick: false,
                    responsive: true
                }};
                
                Plotly.newPlot('plotDiv', currentTraces, currentLayout, config);
                setupEventListeners();
            }}
            
            // Function to resize the plot
            function resizePlot() {{
                if (plotDiv && plotDiv._fullLayout) {{
                    const newLayout = getResponsiveLayout();
                    const newTraces = getResponsiveTraces(newLayout);
                    
                    // Update layout and relayout
                    Plotly.react('plotDiv', newTraces, newLayout);
                }}
            }}
            
            // Function to setup event listeners
            function setupEventListeners() {{
                const hoverBox = document.getElementById('hoverBox');
                const hoverTitle = document.getElementById('hoverTitle');
                const hoverContent = document.getElementById('hoverContent');
                
                function showHoverForPoint(text, cluster, color) {{
                    hoverTitle.innerHTML = `<span style="color: ${{color}};">● ${{cluster}}</span>`;
                    hoverContent.innerHTML = text;
                    hoverBox.classList.add('visible');
                }}
                
                plotDiv.on('plotly_hover', function(data) {{
                    const point = data.points[0];
                    if (point.customdata && point.customdata.length >= 3) {{
                        const [text, cluster, color] = point.customdata;
                        showHoverForPoint(text, cluster, color);
                    }}
                }});
                
                plotDiv.on('plotly_unhover', function(data) {{
                    hoverBox.classList.remove('visible');
                }});
                
                // Show default point after 0.1 seconds
                setTimeout(() => {{
                    if (window.allDataPoints && window.allDataPoints.length > DEFAULT_POINT_INDEX) {{
                        const defaultPoint = window.allDataPoints[DEFAULT_POINT_INDEX];
                        showHoverForPoint(defaultPoint.hover_text, defaultPoint.cluster, defaultPoint.color);
                    }}
                }}, 100);
            }}
            
            // Debounced resize function
            let resizeTimeout;
            function debouncedResize() {{
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(resizePlot, 300);
            }}
            
            // Add event listeners
            window.addEventListener('resize', debouncedResize);
            window.addEventListener('orientationchange', function() {{
                setTimeout(debouncedResize, 500);
            }});
            
            // Initial plot creation
            createPlot();
            
            // Force initial resize after a short delay to ensure proper sizing
            setTimeout(() => {{
                if (plotDiv) {{
                    Plotly.Plots.resize('plotDiv');
                }}
            }}, 100);
        </script>
    </body>
    </html>
    """

    # --- Render in responsive container ---
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    # Calculate responsive height for the component with 1.1x increase
    base_height = 1045  # 1.1 * 950 = 1045
    hover_box_height = 240  # 200px hover box + 40px padding
    component_height = base_height - hover_box_height  # Reduce by hover box space
    
    # Use Streamlit's HTML component with adjusted height
    components.html(html_code, height=component_height, scrolling=False)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-text">
        Powered By Terramare ᛘ𓇳     ©2025
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for responsive tracking
if 'plot_resized' not in st.session_state:
    st.session_state.plot_resized = False

run()