import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="🕸️ Narratives & Co-occurrence Mapping", layout="wide")



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
        margin: 0.5rem 0 1rem 0;
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
            font-size: 0.8rem !important;
        }
        .annotation-container {
            padding: 0 10px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Header section
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 class="main-title">🕸️ The Narrative Web</h1>
        <h3 class="sub-title">Where Narratives Collide</h3>
        <p class="description">
            Explore meditation topics and their interconnections through engagement score weighted and sentiment-infused narratives, drawn from thousands of reddit posts and comments shared between January 2024 and June 2025.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Plot annotation with responsive design
    st.markdown("""
    <div class="annotation-container">
        <div style="display: flex; justify-content: center;">
            <div style="max-width: 600px; text-align: center;">
                <h4 style="font-size: 24px; color: #333; margin: 0 0 15px 0; font-weight: 600;">
                    👁️ Hover Over to Discover the Narratives
                </h4>
                <div style="text-align: left; margin: 0 auto;">
                    <p style="font-size: 18px; color: #444; margin: 5px 0; line-height: 1.5;">
                        • <strong>Dots</strong> represent Reddit posts or comments about meditation
                    </p>
                    <p style="font-size: 18px; color: #444; margin: 5px 0; line-height: 1.5;">
                        • <strong>Lines</strong> connect dots when their topics appear together
                    </p>
                    <p style="font-size: 18px; color: #444; margin: 5px 0; line-height: 1.5;">
                        • <strong>Size & thickness</strong> reflect community engagement
                    </p>
                    <p style="font-size: 18px; color: #444; margin: 5px 0; line-height: 1.5;">
                        • <strong>Line color</strong> shows sentiment: green (positive), red (negative)
                    </p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_edges_clusters():
        return pd.read_parquet("precomputed/timeseries/df_edges.parquet")

    @st.cache_data
    def load_nodes_clusters():
        return pd.read_parquet("precomputed/timeseries/df_nodes.parquet")

    edges = load_edges_clusters()
    nodes = load_nodes_clusters()

    df_edges = edges.copy()
    df_nodes = nodes.copy()

    # Check if sentiment column exists and has non-default values
    nodes_has_sentiment = 'sentiment' in df_nodes.columns
    edges_has_sentiment = 'sentiment' in df_edges.columns

    # Add default sentiment if column doesn't exist
    if not nodes_has_sentiment:
        df_nodes['sentiment'] = 0.0
        st.warning("⚠️ Sentiment data not found in nodes. Using default values. Please regenerate data with updated preprocessing script.")

    if not edges_has_sentiment:
        df_edges['sentiment'] = 0.0
        st.warning("⚠️ Sentiment data not found in edges. Using default values. Please regenerate data with updated preprocessing script.")

    # Apply coordinate rotation
    df_nodes['x_rotated'] = -df_nodes['y']
    df_nodes['y_rotated'] = df_nodes['x']
    df_edges['x0_rotated'] = -df_edges['y0']
    df_edges['y0_rotated'] = df_edges['x0']
    df_edges['x1_rotated'] = -df_edges['y1']
    df_edges['y1_rotated'] = df_edges['x1']

    # ===== IDENTIFY CONNECTED NODES ONLY =====
    node_coords = set()
    for _, edge in df_edges.iterrows():
        node_coords.add((edge['x0_rotated'], edge['y0_rotated']))
        node_coords.add((edge['x1_rotated'], edge['y1_rotated']))
    
    def has_edge(row):
        node_coord = (row['x_rotated'], row['y_rotated'])
        return node_coord in node_coords
    
    df_nodes_connected = df_nodes[df_nodes.apply(has_edge, axis=1)].copy()
    
    total_nodes = len(df_nodes)
    connected_nodes = len(df_nodes_connected)
    st.info(f"📊 Showing {connected_nodes} co-current themes out of total {total_nodes} themes({connected_nodes/total_nodes*100:.1f}%), among high-engagement, strong-sentiment posts or comments (engagement > 30, intensity > 0.3). One post may link to multiple themes.")

    # Color mapping for clusters
    custom_cmap = ['#1f77b4', '#808000', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#17becf']
    unique_clusters = sorted(df_nodes_connected['cluster_name'].unique())
    cluster_color_map = {name: custom_cmap[i % len(custom_cmap)] for i, name in enumerate(unique_clusters)}

    # Prepare edge data for JavaScript with BOLD NAMES + REGULAR CONTENT
    edge_data = []
    for _, edge in df_edges.iterrows():
        # Get cluster names for start and end nodes
        start_coord = (edge['x0_rotated'], edge['y0_rotated'])
        end_coord = (edge['x1_rotated'], edge['y1_rotated'])
        
        # Find cluster names for the coordinates
        start_cluster = "Unknown"
        end_cluster = "Unknown"
        
        for _, node in df_nodes_connected.iterrows():
            node_coord = (node['x_rotated'], node['y_rotated'])
            if node_coord == start_coord:
                start_cluster = node['cluster_name']
            elif node_coord == end_coord:
                end_cluster = node['cluster_name']
        
        edge_data.append({
            'x0': float(edge['x0_rotated']),
            'y0': float(edge['y0_rotated']),
            'x1': float(edge['x1_rotated']),
            'y1': float(edge['y1_rotated']),
            'weight': float(edge['weight']),
            'color': edge['color'],
            'hover_text': f"<b>Topics:</b> {start_cluster} ↔ {end_cluster}<br><b>Themes:</b> {edge['theme_1']} ↔ {edge['theme_2']}<br><b>Engagement Score:</b> {int(edge['weight'])}<br><b>Sentiment:</b> {edge['sentiment']:.2f}"
        })

    # Prepare node data for JavaScript with BOLD NAMES + REGULAR CONTENT
    node_data = []
    for cluster in unique_clusters:
        cluster_data = df_nodes_connected[df_nodes_connected['cluster_name'] == cluster]
        for _, node in cluster_data.iterrows():
            if isinstance(node['avg_score'], set):
                avg_score_display = int(next(iter(node['avg_score'])))
            else:
                avg_score_display = int(float(node['avg_score']))
            
            sentiment_value = float(node['sentiment'])
            
            node_data.append({
                'x': float(node['x_rotated']),
                'y': float(node['y_rotated']),
                'size': float(node['scaled_size'])/2.5,
                'color': cluster_color_map[cluster],
                'cluster': cluster,
                'hover_text': f"<b>Topic:</b> {node['cluster_name']}<br><b>Theme:</b> {node['theme']}<br><b>Engagement Score:</b> {avg_score_display}<br><b>Sentiment:</b> {sentiment_value:.2f}"
            })

    # Calculate centroids for labels
    centroids = df_nodes_connected.groupby('cluster_name').apply(
        lambda g: pd.Series({
            'x': np.average(g['x_rotated'], weights=g['scaled_size']),
            'y': np.average(g['y_rotated'], weights=g['scaled_size']),
            'size_sum': g['scaled_size'].sum()
        })
    ).reset_index()

    angle_offset = np.linspace(0, 2.2 * np.pi, len(centroids), endpoint=False)
    angle_offset += np.pi / len(centroids)
    radius_offset = 0.4

    centroids['x'] += radius_offset * np.cos(angle_offset)
    centroids['y'] += radius_offset * np.sin(angle_offset)

    # Prepare label data for JavaScript
    label_data = []
    for _, row in centroids.iterrows():
        label_data.append({
            'x': float(row['x']),
            'y': float(row['y']),
            'text': row['cluster_name'],
            'color': cluster_color_map[row['cluster_name']]
        })

    # Create the HTML with WORKING EDGE HOVER using scatter points
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
                min-height: 434px;
            }}
            
            #plotDiv {{ 
                width: 100%; 
                height: 100%; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
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
            
            // Function to determine if a color is light or dark
            function isLightColor(color) {{
                // Handle hex colors
                if (color.startsWith('#')) {{
                    const hex = color.slice(1);
                    const r = parseInt(hex.slice(0, 2), 16);
                    const g = parseInt(hex.slice(2, 4), 16);
                    const b = parseInt(hex.slice(4, 6), 16);
                    
                    // Calculate relative luminance
                    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                    return luminance > 0.5;
                }}
                
                // Handle rgba colors
                if (color.startsWith('rgba')) {{
                    const values = color.match(/rgba?\(([^)]+)\)/)[1].split(',');
                    const r = parseInt(values[0].trim());
                    const g = parseInt(values[1].trim());
                    const b = parseInt(values[2].trim());
                    
                    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                    return luminance > 0.5;
                }}
                
                // Default to light if can't determine
                return true;
            }}
            
            // Function to get contrasting text color
            function getContrastingTextColor(backgroundColor) {{
                return isLightColor(backgroundColor) ? '#000000' : '#ffffff';
            }}
            
            function getResponsiveLayout() {{
                const containerWidth = window.innerWidth;
                const containerHeight = window.innerHeight;
                
                let leftMargin, rightMargin, topMargin, bottomMargin;
                
                if (containerWidth <= 480) {{
                    leftMargin = rightMargin = 10;
                    topMargin = 0;
                    bottomMargin = 0;
                }} else if (containerWidth <= 768) {{
                    leftMargin = rightMargin = 30;
                    topMargin = 0;
                    bottomMargin = 0;
                }} else if (containerWidth <= 1024) {{
                    leftMargin = rightMargin = 50;
                    topMargin = 5;
                    bottomMargin = 0;
                }} else {{
                    leftMargin = rightMargin = 50;
                    topMargin = 5;
                    bottomMargin = 0;
                }}
                
                let plotHeight;
                if (containerWidth <= 480) {{
                    plotHeight = Math.max(252, containerHeight * 0.403);
                }} else if (containerWidth <= 768) {{
                    plotHeight = Math.max(286, containerHeight * 0.432);
                }} else {{
                    plotHeight = Math.max(384, containerHeight * 0.518);
                }}
                
                const labelFontSize = containerWidth <= 768 ? 12 : 14;
                
                // Responsive hover font size
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
            
            // Function to create responsive traces - WORKING SOLUTION with error handling
            function getResponsiveTraces(layout) {{
                console.log('Creating traces...');
                const traces = [];
                
                try {{
                    // STEP 1: Add visual edge lines (NO HOVER)
                    console.log('Adding edge lines...');
                    edgeData.forEach((edge, index) => {{
                        traces.push({{
                            x: [edge.x0, edge.x1],
                            y: [edge.y0, edge.y1],
                            mode: 'lines',
                            line: {{
                                width: edge.weight * 0.005,
                                color: edge.color
                            }},
                            opacity: 0.6,
                            hoverinfo: 'skip',  // CRITICAL: No hover on lines
                            showlegend: false,
                            name: 'edge_lines'
                        }});
                    }});
                    console.log(`Added ${{edgeData.length}} edge lines`);
                    
                    // STEP 2: Create scatter points along edges for hover
                    console.log('Creating edge hover points...');
                    const edgeHoverX = [];
                    const edgeHoverY = [];
                    const edgeHoverTexts = [];
                    const edgeHoverColors = [];
                    
                    edgeData.forEach((edge, edgeIndex) => {{
                        // Calculate number of points based on edge length and weight
                        const dx = edge.x1 - edge.x0;
                        const dy = edge.y1 - edge.y0;
                        const edgeLength = Math.sqrt(dx * dx + dy * dy);
                        const numPoints = Math.max(5, Math.floor(edgeLength * 20)); // More points for longer edges
                        
                        // Create points along the edge
                        for (let i = 0; i < numPoints; i++) {{
                            const t = i / (numPoints - 1); // 0 to 1
                            const x = edge.x0 + t * dx;
                            const y = edge.y0 + t * dy;
                            
                            edgeHoverX.push(x);
                            edgeHoverY.push(y);
                            edgeHoverTexts.push(edge.hover_text);
                            edgeHoverColors.push(edge.color);
                        }}
                    }});
                    
                    // Add invisible hover points
                    if (edgeHoverX.length > 0) {{
                        traces.push({{
                            x: edgeHoverX,
                            y: edgeHoverY,
                            mode: 'markers',
                            marker: {{
                                size: 12,  // Large for easy hover
                                color: 'rgba(0,0,0,0)',  // Completely invisible
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
                        console.log(`Added ${{edgeHoverX.length}} edge hover points`);
                    }}
                
                // Add nodes by cluster with individual hover colors
                console.log('Adding node clusters...');
                const clusters = [...new Set(nodeData.map(n => n.cluster))];
                console.log(`Found ${{clusters.length}} clusters:`, clusters);
                
                clusters.forEach(cluster => {{
                    const clusterNodes = nodeData.filter(n => n.cluster === cluster);
                    console.log(`Cluster ${{cluster}}: ${{clusterNodes.length}} nodes`);
                    
                    if (clusterNodes.length === 0) return;
                    
                    const sizes = clusterNodes.map(n => window.innerWidth <= 768 ? n.size * 0.8 : n.size);
                    
                    // Create individual hover background colors with transparency and contrasting text
                    const hoverBgColors = clusterNodes.map(n => {{
                        try {{
                            // Convert hex to rgba with transparency
                            const hex = n.color || '#1f77b4';
                            const r = parseInt(hex.slice(1, 3), 16);
                            const g = parseInt(hex.slice(3, 5), 16);
                            const b = parseInt(hex.slice(5, 7), 16);
                            return `rgba(${{r}}, ${{g}}, ${{b}}, 0.9)`;  // 90% opacity
                        }} catch (e) {{
                            console.warn('Color conversion error for node:', n, e);
                            return 'rgba(31, 119, 180, 0.9)';  // Default blue
                        }}
                    }});
                    
                    // Create contrasting text colors based on background
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
                console.log(`Added ${{clusters.length}} node cluster traces`);
                
                // Add cluster labels
                console.log('Adding cluster labels...');
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
                console.log(`Added ${{labelData.length}} label traces`);
                
                console.log(`Total traces created: ${{traces.length}}`);
                return traces;
                
                }} catch (error) {{
                    console.error('❌ Error creating traces:', error);
                    return [];
                }}
            }}
            
            function createPlot() {{
                console.log('Creating plot...');
                
                currentLayout = getResponsiveLayout();
                currentTraces = getResponsiveTraces(currentLayout);
                
                console.log('Layout:', currentLayout);
                console.log('Traces count:', currentTraces.length);
                console.log('Node data count:', nodeData.length);
                console.log('Edge data count:', edgeData.length);
                
                const config = {{
                    displayModeBar: true,
                    toImageButtonOptions: {{
                        format: 'png',
                        filename: 'narrative_web',
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
                            console.log('✅ Plot created successfully');
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
            
            // Event listeners with detailed debugging
            function setupEventListeners() {{
                plotDiv.on('plotly_hover', function(data) {{
                    const traceName = data.points[0].data.name;
                    console.log('Hover detected on:', traceName);
                    
                    if (traceName === 'edge_hover_points') {{
                        console.log('✅ EDGE HOVER SUCCESS! Text:', data.points[0].text);
                        console.log('Point details:', data.points[0]);
                    }} else {{
                        console.log('Hover on other element:', traceName);
                    }}
                }});
                
                plotDiv.on('plotly_unhover', function(data) {{
                    console.log('Unhover event');
                }});
                
                // Additional debugging for clicks
                plotDiv.on('plotly_click', function(data) {{
                    console.log('Click event:', data.points[0].data.name);
                }});
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

    # Render in responsive container
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    base_height = 744
    component_height = base_height
    
    components.html(html_code, height=component_height, scrolling=False)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-text">
        Powered By Terramare ᛘ𓇳     ©2025
    </div>
    """, unsafe_allow_html=True)

if 'plot_resized' not in st.session_state:
    st.session_state.plot_resized = False

run()