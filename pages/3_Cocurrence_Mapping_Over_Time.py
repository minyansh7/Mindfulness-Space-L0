import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="🎐 The Connected narratives Change Over Time", layout="wide")

def run():
    # --- Enhanced Styled CSS with Meditative Theme ---
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
        margin: 0;
        padding: 0 200px;
    }
    
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
        .description {
            font-size: 0.9rem !important;
        }
        .annotation-container {
            padding: 0 20px;
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
            font-size: 0.8rem !important;
        }
        .annotation-container {
            padding: 0 10px;
        }
    }
    
    /* Enhanced Streamlit controls styling - FIXED: No more red colors! */
    .stSelectbox > label {
        font-family: "Inter", sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #4A5568 !important;
        letter-spacing: 0.5px !important;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(5px) !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stButton > button {
        font-family: "Inter", sans-serif !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 25px !important;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
        color: #4A5568 !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(5px) !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2) !important;
        color: #667eea !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stButton > button:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header section
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 class="main-title">🎐 The Connected Narrative</h1>
        <h3 class="sub-title">Where Narratives Converge — And How They Evolve</h3>
        <p class="description">
            This map reveals a dynamic view of how narratives intersect — and how their connections evolve over time, drawn from thousands of reddit posts and comments shared between January 2024 and June 2025.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Poetic invitation section
    st.markdown("""
    <div class="annotation-container">
        <div style="display: flex; justify-content: center;">
            <div style="max-width: 500px; text-align: center;">
                <p style="font-size: 16px; color: #718096; margin: 0; line-height: 1.8; font-weight: 300; opacity: 0.9;">
                    flow over the narratives change over time<br>
                    let the connections of topics reveal themselves<br> 
                    more engaed discussions are connected by bolder lines<br>
                    watch for green tone on positive, and red for negative sentiment<br>
                       <br> 
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    @st.cache_data
    def load_edges_clusters():
        return pd.read_parquet("precomputed/timeseries/df_edges.parquet")

    @st.cache_data
    def load_nodes_clusters():
        return pd.read_parquet("precomputed/timeseries/df_nodes.parquet")

    df_edges = load_edges_clusters()
    df_nodes = load_nodes_clusters()

    quarters = sorted(df_nodes['quarter'].unique())
    quarter_labels = [f"{q[:4]}Q{q[-1]}" for q in quarters]
    reverse_label_map = {f"{q[:4]}Q{q[-1]}": q for q in quarters}

    if 'slider_index' not in st.session_state:
        st.session_state.slider_index = 0

    # ===== UPDATED MEDITATIVE TIME CONTROLS =====
    # Clean time controls with meditative styling - NO RED COLORS!
    col1, col2, col3 = st.columns([2, 4, 2])

    with col2:
        # Meditative time period display
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <div style="
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-radius: 20px;
                padding: 1rem;
                border: 1px solid rgba(102, 126, 234, 0.2);
                backdrop-filter: blur(5px);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 60px;
            ">
                <div style="
                    font-size: 20px;
                    font-weight: 600;
                    color: #4A5568;
                ">Time Travel</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        # Add spacing to center the button with the label
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        if st.button("← Previous Quarter", 
                     disabled=st.session_state.slider_index == 0,
                     key="prev_btn", 
                     use_container_width=True):
            st.session_state.slider_index = max(0, st.session_state.slider_index - 1)
            st.rerun()

    with col3:
        # Add spacing to center the button with the label
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        if st.button("Next Quarter →", 
                     disabled=st.session_state.slider_index == len(quarter_labels) - 1,
                     key="next_btn", 
                     use_container_width=True):
            st.session_state.slider_index = min(len(quarter_labels) - 1, st.session_state.slider_index + 1)
            st.rerun()


    selected_quarter = reverse_label_map[quarter_labels[st.session_state.slider_index]]

    # Prepare data for current quarter
    nodes_q = df_nodes[df_nodes['quarter'] == selected_quarter].copy()
    edges_q = df_edges[df_edges['quarter'] == selected_quarter].copy()

    # Calculate statistics for the current quarter
    total_nodes_q = len(nodes_q)
    total_edges_q = len(edges_q)
    
    all_nodes_quarter = nodes_q.copy()
    
    # Count connected themes
    node_coords_q = set()
    for _, edge in edges_q.iterrows():
        node_coords_q.add((edge['x0'], edge['y0']))
        node_coords_q.add((edge['x1'], edge['y1']))
    
    def has_edge_q(row):
        node_coord = (row['x'], row['y'])
        return node_coord in node_coords_q
    
    connected_nodes_q = all_nodes_quarter[all_nodes_quarter.apply(has_edge_q, axis=1)]
    connected_count = len(connected_nodes_q)
    connected_percentage = (connected_count / total_nodes_q * 100) if total_nodes_q > 0 else 0

    # Dynamic statistics display
    st.markdown(f"""
    <div style="margin: 0.5rem 0; padding: 0;">
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    border-left: 4px solid #667eea; 
                    padding: 0.75rem 1rem; 
                    border-radius: 0.5rem; 
                    margin: 0;
                    backdrop-filter: blur(5px);">
            <div style="font-size: 0.9rem; color: #4A5568; margin: 0; line-height: 1.4; letter-spacing: 0.3px;">
                <strong>{connected_count} meditation topics</strong> are connected to each other out of <strong>{total_nodes_q} total topics</strong> 
                (<strong>{connected_percentage:.1f}%</strong>). These come from popular Reddit posts with lots of discussion.<br>
                <span style="color: #4A5568; border-bottom: 4px solid #FFD700; padding-bottom: 1px;"><strong>Hover Over</strong></span> for more details.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Rotate coordinates 90 degrees counterclockwise
    nodes_q['x_rot'] = -nodes_q['y']
    nodes_q['y_rot'] = nodes_q['x']
    edges_q['x0_rot'] = -edges_q['y0']
    edges_q['y0_rot'] = edges_q['x0']
    edges_q['x1_rot'] = -edges_q['y1']
    edges_q['y1_rot'] = edges_q['x1']

    # Use ALL nodes in the quarter
    nodes_q_all = nodes_q.copy()
    
    # Topic mapping with consistent colors from meditation app
    topic_mapping = {
        'Self-Regulation': {'color': '#1f77b4', 'icon': '🎯'},
        'Awareness': {'color': '#84cc16', 'icon': '🌿'},
        'Buddhism & Spirituality': {'color': '#f59e0b', 'icon': '🕉️'},
        'Concentration & Flow': {'color': '#ef4444', 'icon': '🌊'},
        'Practice, Retreat, & Meta': {'color': '#a855f7', 'icon': '🏛️'},
        'Anxiety & Mental Health': {'color': '#22c55e', 'icon': '💚'},
        'Meditation & Mindfulness': {'color': '#17becf', 'icon': '🧘'}
    }

    # Color mapping for clusters using consistent colors
    unique_clusters = sorted(nodes_q_all['cluster_name'].dropna().unique())
    cluster_color_map = {}
    for cluster in unique_clusters:
        if cluster in topic_mapping:
            cluster_color_map[cluster] = topic_mapping[cluster]['color']
        else:
            # Fallback color for any clusters not in mapping
            cluster_color_map[cluster] = '#808080'  # Gray fallback

    # Keep track of which nodes are connected for edge calculations
    node_coords = set()
    for _, edge in edges_q.iterrows():
        node_coords.add((edge['x0_rot'], edge['y0_rot']))
        node_coords.add((edge['x1_rot'], edge['y1_rot']))
    
    def has_edge(row):
        node_coord = (row['x_rot'], row['y_rot'])
        return node_coord in node_coords

    # Prepare edge data for JavaScript
    edge_data = []
    for _, edge in edges_q.iterrows():
        # Get cluster names for start and end nodes
        start_coord = (edge['x0_rot'], edge['y0_rot'])
        end_coord = (edge['x1_rot'], edge['y1_rot'])
        
        # Find cluster names for the coordinates
        start_cluster = "Unknown"
        end_cluster = "Unknown"
        
        for _, node in nodes_q_all.iterrows():
            node_coord = (node['x_rot'], node['y_rot'])
            if node_coord == start_coord:
                start_cluster = node['cluster_name']
            elif node_coord == end_coord:
                end_cluster = node['cluster_name']
        
        edge_data.append({
            'x0': float(edge['x0_rot']),
            'y0': float(edge['y0_rot']),
            'x1': float(edge['x1_rot']),
            'y1': float(edge['y1_rot']),
            'weight': float(edge['weight']),
            'color': edge['color'],
            'hover_text': f"<b>Topics:</b> {start_cluster} ↔ {end_cluster}<br><b>Themes:</b> {edge['theme_1']} ↔ {edge['theme_2']}<br><b>Engagement Score:</b> {int(edge['weight'])}<br><b>Sentiment:</b> {edge['sentiment']:.2f}"
        })

    # Prepare node data for JavaScript
    node_data = []
    for cluster in unique_clusters:
        cluster_data = nodes_q_all[nodes_q_all['cluster_name'] == cluster]
        for _, node in cluster_data.iterrows():
            if isinstance(node['avg_score'], set):
                avg_score_display = int(next(iter(node['avg_score'])))
            else:
                avg_score_display = int(float(node['avg_score']))
            
            try:
                sentiment_value = float(node['sentiment']) if pd.notna(node['sentiment']) else 0.0
            except:
                sentiment_value = 0.0
            
            node_data.append({
                'x': float(node['x_rot']),
                'y': float(node['y_rot']),
                'size': float(node['scaled_size']),
                'color': cluster_color_map[cluster],
                'cluster': cluster,
                'hover_text': f"<b>Topic:</b> {node['cluster_name']}<br><b>Theme:</b> {node['theme']}<br><b>Engagement Score:</b> {avg_score_display}<br><b>Sentiment:</b> {sentiment_value:.2f}"
            })

    # Calculate centroids for labels
    centroids = nodes_q_all.groupby('cluster_name').apply(
        lambda g: pd.Series({
            'x': np.average(g['x_rot'], weights=g['scaled_size']),
            'y': np.average(g['y_rot'], weights=g['scaled_size'])
        })
    ).reset_index()

    # Add offset for labels
    angle_offset = np.linspace(0, 1.2 * np.pi, len(centroids), endpoint=False)
    angle_offset += np.pi / len(centroids)
    radius_offset = 0.27

    centroids['x'] += radius_offset * np.cos(angle_offset)
    centroids['y'] += radius_offset * np.sin(angle_offset)

    label_data = []
    for _, row in centroids.iterrows():
        label_data.append({
            'x': float(row['x']),
            'y': float(row['y']),
            'text': row['cluster_name'],
            'color': cluster_color_map[row['cluster_name']]
        })

    # Create HTML with embedded plot
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
                min-height: 723px;
            }}
            
            #plotDiv {{ 
                width: 100%; 
                height: 100%; 
            }}
            
            .quarter-overlay {{
                position: absolute;
                top: 15px;
                left: 15px;
                z-index: 1000;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-family: "Inter", sans-serif;
                font-size: 14px;
                font-weight: 500;
                letter-spacing: 0.5px;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
            }}
            
            @media (max-width: 768px) {{
                .quarter-overlay {{
                    top: 10px;
                    left: 10px;
                    padding: 6px 12px;
                    font-size: 12px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="quarter-overlay">
                🎐 {quarter_labels[st.session_state.slider_index]}
            </div>
            <div id="plotDiv"></div>
        </div>
        
        <script>
            const edgeData = {json.dumps(edge_data)};
            const nodeData = {json.dumps(node_data)};
            const labelData = {json.dumps(label_data)};
            const clusterColorMap = {json.dumps(cluster_color_map)};
            
            let plotDiv = document.getElementById('plotDiv');
            let currentLayout = null;
            let currentTraces = null;
            
            function isLightColor(color) {{
                if (color.startsWith('#')) {{
                    const hex = color.slice(1);
                    const r = parseInt(hex.slice(0, 2), 16);
                    const g = parseInt(hex.slice(2, 4), 16);
                    const b = parseInt(hex.slice(4, 6), 16);
                    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                    return luminance > 0.5;
                }}
                if (color.startsWith('rgba')) {{
                    const values = color.match(/rgba?\(([^)]+)\)/)[1].split(',');
                    const r = parseInt(values[0].trim());
                    const g = parseInt(values[1].trim());
                    const b = parseInt(values[2].trim());
                    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                    return luminance > 0.5;
                }}
                return true;
            }}
            
            function getContrastingTextColor(backgroundColor) {{
                return isLightColor(backgroundColor) ? '#000000' : '#ffffff';
            }}
            
            function getResponsiveLayout() {{
                const containerWidth = window.innerWidth;
                const containerHeight = window.innerHeight;
                
                let leftMargin, rightMargin, topMargin, bottomMargin;
                
                if (containerWidth <= 480) {{
                    leftMargin = rightMargin = 20;
                    topMargin = 0;
                    bottomMargin = 0;
                }} else if (containerWidth <= 768) {{
                    leftMargin = rightMargin = 50;
                    topMargin = 0;
                    bottomMargin = 0;
                }} else if (containerWidth <= 1024) {{
                    leftMargin = rightMargin = 100;
                    topMargin = 0;
                    bottomMargin = 0;
                }} else {{
                    leftMargin = rightMargin = 200;
                    topMargin = 0;
                    bottomMargin = 0;
                }}
                
                let plotHeight;
                if (containerWidth <= 480) {{
                    plotHeight = Math.max(380, containerHeight * 0.8664);
                }} else if (containerWidth <= 768) {{
                    plotHeight = Math.max(455, containerHeight * 0.8664);
                }} else {{
                    plotHeight = Math.max(524, containerHeight * 0.8664);
                }}
                
                const labelFontSize = containerWidth <= 768 ? 10 : 14;
                let hoverFontSize;
                if (containerWidth <= 480) {{
                    hoverFontSize = 10;
                }} else if (containerWidth <= 768) {{
                    hoverFontSize = 11;
                }} else if (containerWidth <= 1024) {{
                    hoverFontSize = 12;
                }} else {{
                    hoverFontSize = 12;
                }}
                
                return {{
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white',
                    font: {{ color: 'black' }},
                    xaxis: {{
                        showgrid: false,
                        zeroline: false,
                        showticklabels: false,
                        scaleanchor: 'y',
                        scaleratio: 1,
                        fixedrange: true
                    }},
                    yaxis: {{
                        showgrid: false,
                        zeroline: false,
                        showticklabels: false,
                        fixedrange: true
                    }},
                    hovermode: 'closest',
                    hoverdistance: 25,
                    hoverlabel: {{
                        bgcolor: "rgba(255,255,255,0.96)",
                        bordercolor: "rgba(160,160,160,0.4)",
                        font: {{
                            family: "DM Sans, sans-serif",
                            color: "#2c3e50",
                            size: hoverFontSize
                        }}
                    }},
                    autosize: true,
                    height: plotHeight,
                    margin: {{ 
                        t: topMargin, 
                        b: bottomMargin, 
                        l: leftMargin, 
                        r: rightMargin 
                    }},
                    showlegend: false,
                    labelFontSize: labelFontSize,
                    hoverFontSize: hoverFontSize,
                    dragmode: false
                }};
            }}
            
            function getResponsiveTraces(layout) {{
                const traces = [];
                
                try {{
                    // Add visual edge lines
                    edgeData.forEach((edge, index) => {{
                        traces.push({{
                            x: [edge.x0, edge.x1],
                            y: [edge.y0, edge.y1],
                            mode: 'lines',
                            line: {{
                                width: Math.min(8, edge.weight * 0.02),
                                color: edge.color
                            }},
                            opacity: 0.6,
                            hoverinfo: 'skip',
                            showlegend: false,
                            name: 'edge_lines'
                        }});
                    }});
                    
                    // Create scatter points along edges for hover
                    const edgeHoverX = [];
                    const edgeHoverY = [];
                    const edgeHoverTexts = [];
                    const edgeHoverColors = [];
                    
                    edgeData.forEach((edge, edgeIndex) => {{
                        const dx = edge.x1 - edge.x0;
                        const dy = edge.y1 - edge.y0;
                        const edgeLength = Math.sqrt(dx * dx + dy * dy);
                        const numPoints = Math.max(5, Math.floor(edgeLength * 20));
                        
                        for (let i = 0; i < numPoints; i++) {{
                            const t = i / (numPoints - 1);
                            const x = edge.x0 + t * dx;
                            const y = edge.y0 + t * dy;
                            
                            edgeHoverX.push(x);
                            edgeHoverY.push(y);
                            edgeHoverTexts.push(edge.hover_text);
                            edgeHoverColors.push(edge.color);
                        }}
                    }});
                    
                    // Add invisible hover points with white background
                    if (edgeHoverX.length > 0) {{
                        traces.push({{
                            x: edgeHoverX,
                            y: edgeHoverY,
                            mode: 'markers',
                            marker: {{
                                size: 12,
                                color: 'rgba(0,0,0,0)',
                                line: {{ width: 0 }}
                            }},
                            hoverinfo: 'text',
                            hovertext: edgeHoverTexts,
                            hoverlabel: {{
                                bgcolor: "rgba(255,255,255,0.96)",
                                bordercolor: "rgba(160,160,160,0.4)",
                                font: {{
                                    family: "DM Sans, sans-serif",
                                    color: "#2c3e50",
                                    size: layout.hoverFontSize
                                }}
                            }},
                            showlegend: false,
                            name: 'edge_hover_points'
                        }});
                    }}

                    // Add nodes by cluster with cluster-colored hover backgrounds
                    const clusters = [...new Set(nodeData.map(n => n.cluster))];
                    
                    clusters.forEach(cluster => {{
                        const clusterNodes = nodeData.filter(n => n.cluster === cluster);
                        
                        if (clusterNodes.length === 0) return;
                        
                        const sizes = clusterNodes.map(n => window.innerWidth <= 768 ? n.size * 0.8 : n.size);
                        
                        // Create cluster-colored hover backgrounds with transparency
                        const hoverBgColors = clusterNodes.map(n => {{
                            try {{
                                const hex = n.color || '#1f77b4';
                                const r = parseInt(hex.slice(1, 3), 16);
                                const g = parseInt(hex.slice(3, 5), 16);
                                const b = parseInt(hex.slice(5, 7), 16);
                                return `rgba(${{r}}, ${{g}}, ${{b}}, 0.9)`;
                            }} catch (e) {{
                                return 'rgba(31, 119, 180, 0.9)';
                            }}
                        }});
                        
                        // Create contrasting text colors based on cluster color
                        const hoverTextColors = clusterNodes.map(n => {{
                            const bgColor = n.color || '#1f77b4';
                            return getContrastingTextColor(bgColor);
                        }});
                        
                        traces.push({{
                            x: clusterNodes.map(n => n.x),
                            y: clusterNodes.map(n => n.y),
                            mode: 'markers',
                            marker: {{
                                size: sizes,
                                color: clusterNodes[0].color,
                                opacity: 0.7,
                                line: {{ width: 0.5, color: 'white' }}
                            }},
                            hoverinfo: 'text',
                            hovertext: clusterNodes.map(n => n.hover_text),
                            hoverlabel: {{
                                bgcolor: hoverBgColors,
                                bordercolor: clusterNodes.map(n => n.color || '#1f77b4'),
                                font: {{
                                    family: "DM Sans, sans-serif",
                                    color: hoverTextColors,
                                    size: layout.hoverFontSize
                                }}
                            }},
                            showlegend: false,
                            name: `cluster_${{cluster}}`
                        }});
                    }});
                
                    // Add cluster labels
                    labelData.forEach((label, index) => {{
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
                            hoverinfo: 'none',
                            name: `label_${{index}}`
                        }});
                    }});
                    
                    return traces;
                    
                }} catch (error) {{
                    console.error('❌ Error creating traces:', error);
                    return [];
                }}
            }}
            
            function createPlot() {{
                currentLayout = getResponsiveLayout();
                currentTraces = getResponsiveTraces(currentLayout);
                
                const config = {{
                    displayModeBar: true,
                    toImageButtonOptions: {{
                        format: 'png',
                        filename: 'living_narrative',
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
                    staticPlot: false,
                    responsive: true
                }};
                
                try {{
                    Plotly.newPlot('plotDiv', currentTraces, currentLayout, config)
                        .then(function() {{
                            setupEventListeners();
                        }})
                        .catch(function(error) {{
                            console.error('❌ Plot creation failed:', error);
                        }});
                }} catch (error) {{
                    console.error('❌ Plot creation error:', error);
                }}
            }}
            
            function resizePlot() {{
                if (plotDiv && plotDiv._fullLayout) {{
                    const newLayout = getResponsiveLayout();
                    const newTraces = getResponsiveTraces(newLayout);
                    Plotly.react('plotDiv', newTraces, newLayout);
                }}
            }}
            
            function setupEventListeners() {{
                plotDiv.on('plotly_hover', function(data) {{
                    const traceName = data.points[0].data.name;
                }});
                
                plotDiv.on('plotly_unhover', function(data) {{}});
                plotDiv.on('plotly_click', function(data) {{}});
            }}
            
            let resizeTimeout;
            function debouncedResize() {{
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(resizePlot, 300);
            }}
            
            window.addEventListener('resize', debouncedResize);
            window.addEventListener('orientationchange', function() {{
                setTimeout(debouncedResize, 500);
            }});
            
            createPlot();
            
            setTimeout(() => {{
                if (plotDiv) {{
                    Plotly.Plots.resize('plotDiv');
                }}
            }}, 100);
        </script>
    </body>
    </html>
    """

    # Render plot
    st.markdown('<div class="plot-container" style="margin: 0; padding: 0;">', unsafe_allow_html=True)
    
    base_height = 824  # Increased from 549 to 824 (1.5x)
    component_height = base_height
    
    components.html(html_code, height=component_height, scrolling=False)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer-text">
        Powered By Terramare ᛘ𓇳     ©2025
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'plot_resized' not in st.session_state:
    st.session_state.plot_resized = False

run()