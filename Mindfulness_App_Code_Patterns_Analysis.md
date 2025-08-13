# Mindfulness Space Application - Code Patterns Analysis

## Overview
This analysis examines the Python scripts in the `/pages` directory of the Mindfulness Space application, which appears to be a comprehensive Streamlit-based data visualization platform for analyzing meditation-related Reddit discussions using advanced NLP and emotion analysis.

## File Structure and Purpose

The application consists of 7 main visualization scripts:

1. **0_Emotion_Pulse.py** - Emotional sentiment mapping using UMAP clustering
2. **0_Emotion_Pulse_v.py** - Alternative version with simplified hover functionality
3. **1_Meditation_Weather_Report.py** - Time-based sentiment analysis with weather metaphors
4. **2_Main_Topics_Sankey.py** - Topic flow visualization using Sankey diagrams
5. **3_Cocurrence_Mapping_Over_Time.py** - Time-series co-occurrence network analysis
6. **4_Cocurrence_Mapping_complex.py** - Professional river flow interface
7. **5_Cocurrence_Mapping.py** - Static co-occurrence network visualization

## Core Architectural Patterns

### 1. Streamlit Application Structure

All scripts follow a consistent Streamlit application pattern:

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Application Title",
    layout="wide"
)

def run():
    # Application logic
    pass

if __name__ == "__main__":
    run()
```

### 2. Data Loading and Caching Pattern

Consistent use of Streamlit's caching decorator for performance optimization:

```python
@st.cache_data
def load_data():
    return pd.read_parquet("precomputed/data_file.parquet")

# Example from Main Topics Sankey
@st.cache_data
def load_main_topics():
    return pd.read_parquet("precomputed/main_topics.parquet")

# Example from Emotion Pulse
@st.cache_data
def load_emotion_clusters():
    return pd.read_parquet("precomputed/emotion_clusters.parquet")
```

### 3. Responsive CSS and Styling

Each script includes comprehensive responsive CSS with mobile-first design:

```css
/* Mobile responsive breakpoints */
@media (max-width: 1200px) {
    .annotation-container {
        padding: 0 100px;
    }
}

@media (max-width: 768px) {
    .main-title {
        font-size: 2rem !important;
    }
    .sub-title {
        font-size: 1.2rem !important;
    }
}

@media (max-width: 480px) {
    .main-title {
        font-size: 1.5rem !important;
    }
}
```

## Data Processing Patterns

### 1. Coordinate Transformation

Multiple scripts apply 90-degree counterclockwise rotation for visualization layout:

```python
# Pattern found in multiple files
nodes_q['x_rot'] = -nodes_q['y']
nodes_q['y_rot'] = nodes_q['x']
edges_q['x0_rot'] = -edges_q['y0']
edges_q['y0_rot'] = edges_q['x0']
edges_q['x1_rot'] = -edges_q['y1']
edges_q['y1_rot'] = edges_q['x1']
```

### 2. Connected Node Filtering

Pattern for identifying nodes that participate in network connections:

```python
# Build set of coordinates that have edges
node_coords = set()
for _, edge in edges_q.iterrows():
    node_coords.add((edge['x0_rot'], edge['y0_rot']))
    node_coords.add((edge['x1_rot'], edge['y1_rot']))

def has_edge(row):
    node_coord = (row['x_rot'], row['y_rot'])
    return node_coord in node_coords

connected_nodes = df_nodes[df_nodes.apply(has_edge, axis=1)]
```

### 3. Sentiment Analysis Integration

Consistent pattern for handling sentiment data with fallbacks:

```python
# Check if sentiment column exists
if 'sentiment' not in df_nodes.columns:
    df_nodes['sentiment'] = 0.0
    st.warning("⚠️ Sentiment data not found. Using default values.")

# Safe sentiment value extraction
try:
    sentiment_value = float(node['sentiment']) if pd.notna(node['sentiment']) else 0.0
except:
    sentiment_value = 0.0
```

## Visualization Patterns

### 1. Plotly Integration with Custom HTML

Hybrid approach combining Plotly's power with custom HTML for enhanced interactivity:

```python
# Create HTML with embedded Plotly visualization
html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* Custom responsive CSS */
    </style>
</head>
<body>
    <div id="plotDiv"></div>
    <script>
        const plotData = {json.dumps(data)};
        // Custom JavaScript for enhanced interactions
        Plotly.newPlot('plotDiv', traces, layout, config);
    </script>
</body>
</html>
"""

# Render with Streamlit components
components.html(html_code, height=600, scrolling=False)
```

### 2. Dynamic Color Mapping

Consistent color scheme management across visualizations:

```python
# Topic mapping with colors and icons
topic_mapping = {
    'Self-Regulation': {'color': '#1f77b4', 'icon': '🎯'},
    'Awareness': {'color': '#84cc16', 'icon': '🌿'},
    'Buddhism & Spirituality': {'color': '#f59e0b', 'icon': '🕉️'},
    'Concentration & Flow': {'color': '#ef4444', 'icon': '🎯'},
    'Practice, Retreat, & Meta': {'color': '#a855f7', 'icon': '🏛️'},
    'Anxiety & Mental Health': {'color': '#22c55e', 'icon': '💚'},
    'Meditation & Mindfulness': {'color': '#17becf', 'icon': '🧘'}
}

# Cluster color mapping
cluster_color_map = {cluster: topic_mapping.get(cluster, {}).get('color', '#64748b') 
                    for cluster in unique_clusters}
```

### 3. Smart Hover System

Advanced hover functionality with contextual information:

```python
def ideal_text_color(bg_hex: str) -> str:
    """Calculate contrasting text color for readability"""
    bg_hex = bg_hex.lstrip("#")
    r, g, b = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 186 else 'white'

# Generate hover text with rich formatting
hover_text = f"""
<b>Topic:</b> {cluster_name}<br>
<b>Theme:</b> {theme}<br>
<b>Engagement Score:</b> {engagement}<br>
<b>Sentiment:</b> {sentiment:.2f}
"""
```

## Time-Based Analysis Patterns

### 1. Quarter-Based Navigation

Sophisticated time navigation system with visual feedback:

```python
# Quarter processing
quarters = sorted(df_nodes['quarter'].unique())
quarter_labels = [f"{q[:4]}Q{q[-1]}" for q in quarters]
reverse_label_map = {f"{q[:4]}Q{q[-1]}": q for q in quarters}

# Session state management
if 'slider_index' not in st.session_state:
    st.session_state.slider_index = len(quarter_labels) - 1  # Default to latest

# Dynamic quarter selection
selected_quarter = reverse_label_map[quarter_labels[st.session_state.slider_index]]
```

### 2. Time-Based Color Schemes

Dynamic theming based on time of day:

```python
def get_time_colors(hour):
    """Return color scheme based on current hour"""
    if 5 <= hour < 8:  # Dawn
        return {
            'primary': '#FF6B6B',
            'secondary': '#FFB347',
            'bg_gradient': 'linear-gradient(135deg, #FF6B6B 0%, #FFB347 100%)',
            'text_color': 'black'
        }
    elif 8 <= hour < 12:  # Morning
        return {
            'primary': '#4ECDC4',
            'secondary': '#45B7D1',
            'bg_gradient': 'linear-gradient(135deg, #4ECDC4 0%, #45B7D1 100%)',
            'text_color': 'black'
        }
    # ... more time periods
```

## Advanced Interaction Patterns

### 1. Multi-Layer Hover Detection

Creating invisible interaction layers for enhanced UX:

```python
# Create edge hover points along paths
edgeData.forEach((edge) => {
    const dx = edge.x1 - edge.x0;
    const dy = edge.y1 - edge.y0;
    const edgeLength = Math.sqrt(dx * dx + dy * dy);
    const numPoints = Math.max(5, Math.floor(edgeLength * 20));
    
    for (let i = 0; i < numPoints; i++) {
        const t = i / (numPoints - 1);
        const x = edge.x0 + t * dx;
        const y = edge.y0 + t * dy;
        
        edgeHoverX.push(x);
        edgeHoverY.push(y);
        edgeHoverTexts.push(edge.hover_text);
    }
});

// Add invisible markers for hover detection
traces.push({
    x: edgeHoverX,
    y: edgeHoverY,
    mode: 'markers',
    marker: {
        size: 12,
        color: 'rgba(0,0,0,0)',  // Completely invisible
        line: { width: 0 }
    },
    hoverinfo: 'text',
    hovertext: edgeHoverTexts
});
```

### 2. Responsive Layout Calculations

Dynamic layout adjustment based on screen size:

```python
function getResponsiveLayout() {
    const containerWidth = window.innerWidth;
    const containerHeight = window.innerHeight;
    
    let leftMargin, rightMargin, topMargin, bottomMargin;
    
    if (containerWidth <= 480) {
        leftMargin = rightMargin = 20;
        topMargin = 0;
        bottomMargin = 0;
    } else if (containerWidth <= 768) {
        leftMargin = rightMargin = 50;
        topMargin = 0;
        bottomMargin = 0;
    } else {
        leftMargin = rightMargin = 200;
        topMargin = 0;
        bottomMargin = 0;
    }
    
    let plotHeight = containerWidth <= 480 ? 
        Math.max(380, containerHeight * 0.8664) :
        Math.max(524, containerHeight * 0.8664);
    
    return {
        // Layout configuration
        height: plotHeight,
        margin: { t: topMargin, b: bottomMargin, l: leftMargin, r: rightMargin }
    };
}
```

## Data Pipeline Patterns

### 1. Text Processing and Wrapping

Intelligent text processing for hover displays:

```python
def wrap_hover_text(text):
    """Smart wrapping with different lengths for different content types"""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Clean hidden characters
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace('\u00a0', ' ')  # Non-breaking space
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Split into lines if already formatted
    lines = text.split('<br>')
    wrapped_lines = []
    
    for line in lines:
        if line.startswith("Top Emotions:"):
            # Keep emotions on one line up to 250 chars
            if len(line) <= 300:
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(wrap_emotions_line(line))
        elif line.startswith("Post/Comment:"):
            # Wrap content at 200 characters
            wrapped_lines.extend(wrap_text_content(line, 200))
        else:
            wrapped_lines.extend(wrap_text_content(line, 200))
    
    return '<br>'.join(wrapped_lines)
```

### 2. Statistical Calculations

Consistent metrics calculation across visualizations:

```python
def calculate_river_flow_data(df_nodes, df_edges, selected_quarter):
    """Calculate comprehensive data for visualization"""
    nodes_q = df_nodes[df_nodes['quarter'] == selected_quarter].copy()
    edges_q = df_edges[df_edges['quarter'] == selected_quarter].copy()
    
    total_nodes = len(nodes_q)
    total_edges = len(edges_q)
    
    # Calculate co-occurrence rate
    connected_count = len(connected_nodes)
    co_occurrence_rate = (connected_count / total_nodes * 100) if total_nodes > 0 else 0
    
    # Calculate tributary statistics
    tributary_stats = {}
    for cluster in unique_clusters:
        cluster_nodes = nodes_q[nodes_q['cluster_name'] == cluster]
        avg_sentiment = cluster_nodes['sentiment'].mean()
        percentage = (len(cluster_nodes) / total_nodes) * 100
        
        tributary_stats[cluster] = {
            'count': len(cluster_nodes),
            'percentage': percentage,
            'avg_sentiment': avg_sentiment,
            'color': topic_mapping.get(cluster, {}).get('color', '#64748b')
        }
    
    return {
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'co_occurrence_rate': co_occurrence_rate,
        'tributary_stats': tributary_stats
    }
```

## Error Handling and Fallback Patterns

### 1. Graceful Data Handling

Robust error handling for missing or malformed data:

```python
try:
    main_topics = load_main_topics()
except Exception:
    main_topics = pd.DataFrame()
    # Provide fallback empty dataframe with proper structure
    main_topics = pd.DataFrame({
        'cluster_name': ['Meditation & Mindfulness'],
        'sentiment_score': [0.3],
        'pain_topic_label': ['Difficulty concentrating'],
        'quarter': ['2024Q3']
    })

# Safe type conversion
if isinstance(node['avg_score'], set):
    avg_score_display = int(next(iter(node['avg_score'])))
else:
    avg_score_display = int(float(node['avg_score']))
```

### 2. Progressive Enhancement

Features that enhance experience when available but don't break when missing:

```python
# Check for sentiment data availability
nodes_has_sentiment = 'sentiment' in df_nodes.columns
edges_has_sentiment = 'sentiment' in df_edges.columns

if not nodes_has_sentiment:
    df_nodes['sentiment'] = 0.0
    st.warning("⚠️ Sentiment data not found. Using defaults.")

if not edges_has_sentiment:
    df_edges['sentiment'] = 0.0
    st.warning("⚠️ Edge sentiment data not found. Using defaults.")
```

## UI/UX Design Patterns

### 1. Consistent Header Structure

Standardized header layout across all pages:

```python
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h1 style="font-size: 3rem; font-weight: 800;">🎯 Page Title</h1>
    <h3 style="font-size: 1.5rem; font-weight: 500;">Subtitle</h3>
    <p style="font-size: 1rem; color: #888; max-width: 800px; margin: auto;">
        Description text with context about the visualization.
    </p>
</div>
""", unsafe_allow_html=True)
```

### 2. Footer Branding

Consistent footer across all pages:

```python
st.markdown("""
<div class="footer-text">
    Powered By Terramare ᛘ𓇳     ©2025
</div>
""", unsafe_allow_html=True)
```

### 3. Information Displays

Rich information panels with statistics:

```python
st.markdown(f"""
<div style="background-color: #e3f2fd; border-left: 4px solid #2196f3; 
           padding: 0.75rem 1rem; border-radius: 0.25rem; margin: 1rem 0;">
    <div style="font-size: 0.9rem; color: #1565c0; line-height: 1.4;">
        📊 <strong>Statistics:</strong> {connected_count:,} connected themes 
        out of {total_nodes:,} total (<strong>{co_occurrence_rate:.1f}%</strong>)
    </div>
</div>
""", unsafe_allow_html=True)
```

## Performance Optimization Patterns

### 1. Lazy Loading and Caching

Strategic use of Streamlit's caching mechanisms:

```python
@st.cache_data
def process_weather_data(df, sentiment_column):
    """Cache expensive data processing operations"""
    volume_stats = df.groupby('cluster_name').size().reset_index(name='volume')
    sentiment_stats = df.groupby('cluster_name')[sentiment_column].mean().reset_index()
    return volume_stats.merge(sentiment_stats, on='cluster_name')
```

### 2. Debounced Resize Handling

Efficient window resize handling in JavaScript:

```javascript
let resizeTimeout;
function debouncedResize() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(resizePlot, 300);
}

window.addEventListener('resize', debouncedResize);
window.addEventListener('orientationchange', function() {
    setTimeout(debouncedResize, 500);
});
```

## Session State Management

Consistent state management patterns:

```python
# Initialize session state
if 'slider_index' not in st.session_state:
    st.session_state.slider_index = 0

if 'plot_interacted' not in st.session_state:
    st.session_state.plot_interacted = False

# State updates with rerun
if st.button("Next Quarter →"):
    st.session_state.slider_index = min(len(quarters) - 1, 
                                       st.session_state.slider_index + 1)
    st.rerun()
```

## Technology Stack Summary

The application demonstrates sophisticated use of:

- **Streamlit**: Web application framework with component system
- **Plotly**: Interactive visualization library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Custom HTML/CSS/JavaScript**: Enhanced interactivity
- **Parquet**: Efficient data storage format
- **Advanced NLP**: Google's GoEmotions model for sentiment analysis

## Key Innovations

1. **Hybrid Visualization**: Combines Streamlit's simplicity with custom HTML/JavaScript for enhanced interactivity
2. **Responsive Design**: Mobile-first approach with comprehensive breakpoints
3. **Smart Hover Systems**: Multi-layer interaction detection for complex visualizations
4. **Time-Aware Theming**: Dynamic color schemes based on time of day
5. **Graceful Degradation**: Robust fallbacks for missing data or failed operations
6. **Performance Optimization**: Strategic caching and debounced operations

This codebase represents a mature, production-ready data visualization platform with sophisticated UX patterns, robust error handling, and scalable architecture suitable for complex multi-dimensional data analysis.

