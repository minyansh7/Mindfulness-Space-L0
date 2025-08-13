import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json
import datetime

st.set_page_config(page_title="🌊 Professional River Flow - Living Narrative Intelligence", layout="wide")

def get_time_colors(hour):
    """Dynamic color scheme based on time of day"""
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

def run():
    # Get current time colors
    current_hour = datetime.datetime.now().hour
    colors = get_time_colors(current_hour)
    
    # --- Load Data ---
    @st.cache_data
    def load_edges_clusters():
        return pd.read_parquet("precomputed/timeseries/df_edges.parquet")

    @st.cache_data
    def load_nodes_clusters():
        return pd.read_parquet("precomputed/timeseries/df_nodes.parquet")

    df_edges = load_edges_clusters()
    df_nodes = load_nodes_clusters()

    # Handle missing sentiment columns
    if 'sentiment' not in df_nodes.columns:
        df_nodes['sentiment'] = 0.0
    if 'sentiment' not in df_edges.columns:
        df_edges['sentiment'] = 0.0

    quarters = sorted(df_nodes['quarter'].unique())
    quarter_labels = [f"{q[:4]}Q{q[-1]}" for q in quarters]
    reverse_label_map = {f"{q[:4]}Q{q[-1]}": q for q in quarters}

    if 'slider_index' not in st.session_state:
        st.session_state.slider_index = len(quarter_labels) - 1  # Default to latest quarter

    # Color mapping from your script to interface
    script_custom_cmap = ['#1f77b4', '#808000', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#17becf']
    
    # Topic mapping with colors (matching your custom_cmap order)
    topic_mapping = {
        'Self-Regulation': {'color': '#1f77b4', 'icon': '🎯'},
        'Awareness': {'color': '#84cc16', 'icon': '🌿'},  # Updated to lime for better visibility
        'Buddhism & Spirituality': {'color': '#f59e0b', 'icon': '🕉️'},
        'Concentration & Flow': {'color': '#ef4444', 'icon': '🎯'},
        'Practice, Retreat, & Meta': {'color': '#a855f7', 'icon': '🏛️'},
        'Anxiety & Mental Health': {'color': '#22c55e', 'icon': '💚'},
        'Meditation & Mindfulness': {'color': '#17becf', 'icon': '🧘'}
    }

    def calculate_river_flow_data(df_nodes, df_edges, selected_quarter):
        """Calculate comprehensive data for Professional River Flow interface"""
        
        # Filter data for current quarter
        nodes_q = df_nodes[df_nodes['quarter'] == selected_quarter].copy()
        edges_q = df_edges[df_edges['quarter'] == selected_quarter].copy()
        
        # Calculate total statistics
        total_nodes = len(nodes_q)
        total_edges = len(edges_q)
        
        # Calculate co-occurrence rate (percentage of themes that have connections)
        node_coords_q = set()
        for _, edge in edges_q.iterrows():
            node_coords_q.add((edge['x0'], edge['y0']))
            node_coords_q.add((edge['x1'], edge['y1']))
        
        def has_edge_q(row):
            return (row['x'], row['y']) in node_coords_q
        
        connected_nodes = nodes_q[nodes_q.apply(has_edge_q, axis=1)]
        connected_count = len(connected_nodes)
        co_occurrence_rate = (connected_count / total_nodes * 100) if total_nodes > 0 else 0
        
        # Calculate tributary statistics
        tributary_stats = {}
        unique_clusters = sorted(nodes_q['cluster_name'].dropna().unique())
        
        for cluster in unique_clusters:
            cluster_nodes = nodes_q[nodes_q['cluster_name'] == cluster]
            cluster_count = len(cluster_nodes)
            percentage = (cluster_count / total_nodes) * 100
            
            # Calculate average sentiment for cluster
            avg_sentiment = cluster_nodes['sentiment'].mean()
            
            # Calculate co-occurrences with other clusters
            cluster_edges = []
            for _, edge in edges_q.iterrows():
                # Find nodes at edge endpoints
                start_nodes = nodes_q[(nodes_q['x'] == edge['x0']) & (nodes_q['y'] == edge['y0'])]
                end_nodes = nodes_q[(nodes_q['x'] == edge['x1']) & (nodes_q['y'] == edge['y1'])]
                
                if not start_nodes.empty and not end_nodes.empty:
                    start_cluster = start_nodes.iloc[0]['cluster_name']
                    end_cluster = end_nodes.iloc[0]['cluster_name']
                    
                    if start_cluster == cluster or end_cluster == cluster:
                        other_cluster = end_cluster if start_cluster == cluster else start_cluster
                        if other_cluster != cluster:
                            cluster_edges.append((other_cluster, edge['weight']))
            
            # Calculate top co-occurrences
            co_occurrence_counts = {}
            for other_cluster, weight in cluster_edges:
                co_occurrence_counts[other_cluster] = co_occurrence_counts.get(other_cluster, 0) + weight
            
            # Get top 2 co-occurrences
            top_co_occurrences = sorted(co_occurrence_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            
            tributary_stats[cluster] = {
                'count': cluster_count,
                'percentage': percentage,
                'avg_sentiment': avg_sentiment,
                'co_occurrences': top_co_occurrences,
                'color': topic_mapping.get(cluster, {}).get('color', '#64748b'),
                'icon': topic_mapping.get(cluster, {}).get('icon', '📊')
            }
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'connected_count': connected_count,
            'co_occurrence_rate': co_occurrence_rate,
            'tributary_stats': tributary_stats,
            'nodes_quarter': nodes_q,
            'edges_quarter': edges_q,
            'connected_nodes': connected_nodes
        }

    def create_dynamic_visualization_html(river_data, selected_label):
        """Create dynamic HTML visualization with smart hover system from script 3"""
        
        nodes_q = river_data['nodes_quarter']
        edges_q = river_data['edges_quarter']
        
        if len(nodes_q) == 0:
            return f"""
            <div style="display: flex; justify-content: center; align-items: center; height: 600px; background: white; border-radius: 8px;">
                <div style="text-align: center; color: #64748b;">
                    <div style="font-size: 24px; margin-bottom: 8px;">📊</div>
                    <div>No data available for {selected_label}</div>
                </div>
            </div>
            """
        
        # Apply coordinate rotation
        nodes_q = nodes_q.copy()
        edges_q = edges_q.copy()
        nodes_q['x_rot'] = -nodes_q['y']
        nodes_q['y_rot'] = nodes_q['x']
        edges_q['x0_rot'] = -edges_q['y0']
        edges_q['y0_rot'] = edges_q['x0']
        edges_q['x1_rot'] = -edges_q['y1']
        edges_q['y1_rot'] = edges_q['x1']

        # Use ALL nodes in the quarter (raw data already filtered by engagement > 30, sentiment > 0.3)
        nodes_q_all = nodes_q.copy()
        
        # Keep track of which nodes are connected for edge calculations
        node_coords = set()
        for _, edge in edges_q.iterrows():
            node_coords.add((edge['x0_rot'], edge['y0_rot']))
            node_coords.add((edge['x1_rot'], edge['y1_rot']))
        
        # Color mapping
        custom_cmap = ['#1f77b4', '#808000', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#17becf']
        unique_clusters = sorted(nodes_q_all['cluster_name'].dropna().unique())
        cluster_color_map = {cluster: custom_cmap[i % len(custom_cmap)] for i, cluster in enumerate(unique_clusters)}

        # Prepare edge data for JavaScript
        edge_data = []
        for _, edge in edges_q.iterrows():
            # Get cluster names for start and end nodes
            start_coord = (edge['x0_rot'], edge['y0_rot'])
            end_coord = (edge['x1_rot'], edge['y1_rot'])
            
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

        # Create HTML with smart hover system
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
                    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; 
                    background: white;
                    overflow-x: hidden;
                }}
                
                .container {{
                    position: relative;
                    width: 100%;
                    height: 600px;
                }}
                
                #plotDiv {{ 
                    width: 100%; 
                    height: 100%; 
                }}
                
                .quarter-overlay {{
                    position: absolute;
                    top: 15px;
                    right: 15px;
                    z-index: 1000;
                    background: rgba(255, 255, 255, 0.95);
                    padding: 8px 12px;
                    border-radius: 8px;
                    font-family: "DM Sans", sans-serif;
                    font-size: 14px;
                    font-weight: 600;
                    color: #1e293b;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    border: 1px solid rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="quarter-overlay">
                    🌊 {selected_label}
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
                    return true;
                }}
                
                function getContrastingTextColor(backgroundColor) {{
                    return isLightColor(backgroundColor) ? '#000000' : '#ffffff';
                }}
                
                function getResponsiveLayout() {{
                    const containerWidth = window.innerWidth;
                    
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
                                size: containerWidth <= 768 ? 11 : 12
                            }}
                        }},
                        autosize: true,
                        height: 600,
                        margin: {{ t: 10, b: 10, l: 10, r: 10 }},
                        showlegend: false,
                        dragmode: false
                    }};
                }}
                
                function getResponsiveTraces(layout) {{
                    const traces = [];
                    
                    try {{
                        // Add visual edge lines
                        edgeData.forEach((edge) => {{
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
                        
                        // Create edge hover points
                        const edgeHoverX = [];
                        const edgeHoverY = [];
                        const edgeHoverTexts = [];
                        
                        edgeData.forEach((edge) => {{
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
                            }}
                        }});
                        
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
                                        size: layout.hoverlabel.font.size
                                    }}
                                }},
                                showlegend: false,
                                name: 'edge_hover_points'
                            }});
                        }}

                        // Add nodes by cluster
                        const clusters = [...new Set(nodeData.map(n => n.cluster))];
                        
                        clusters.forEach(cluster => {{
                            const clusterNodes = nodeData.filter(n => n.cluster === cluster);
                            
                            if (clusterNodes.length === 0) return;
                            
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
                            
                            const hoverTextColors = clusterNodes.map(n => {{
                                const bgColor = n.color || '#1f77b4';
                                return getContrastingTextColor(bgColor);
                            }});
                            
                            traces.push({{
                                x: clusterNodes.map(n => n.x),
                                y: clusterNodes.map(n => n.y),
                                mode: 'markers',
                                marker: {{
                                    size: clusterNodes.map(n => window.innerWidth <= 768 ? n.size * 0.8 : n.size),
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
                                        size: layout.hoverlabel.font.size
                                    }}
                                }},
                                showlegend: false,
                                name: `cluster_${{cluster}}`
                            }});
                        }});
                    
                        // Add cluster labels
                        labelData.forEach((label) => {{
                            traces.push({{
                                x: [label.x],
                                y: [label.y],
                                mode: 'text',
                                text: [`<b>${{label.text}}</b>`],
                                textfont: {{ 
                                    size: window.innerWidth <= 768 ? 10 : 12, 
                                    color: label.color, 
                                    family: 'DM Sans, sans-serif' 
                                }},
                                showlegend: false,
                                hoverinfo: 'none',
                                name: `label`
                            }});
                        }});
                        
                        return traces;
                        
                    }} catch (error) {{
                        console.error('Error creating traces:', error);
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
                            height: 600,
                            width: Math.min(1200, window.innerWidth),
                            scale: 2
                        }},
                        modeBarButtonsToRemove: [
                            'pan2d', 'lasso2d', 'select2d', 'zoom2d',
                            'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'
                        ],
                        scrollZoom: false,
                        doubleClick: false,
                        staticPlot: false,
                        responsive: true
                    }};
                    
                    try {{
                        Plotly.newPlot('plotDiv', currentTraces, currentLayout, config)
                            .then(function() {{
                                console.log('Plot created successfully');
                            }})
                            .catch(function(error) {{
                                console.error('Plot creation failed:', error);
                            }});
                    }} catch (error) {{
                        console.error('Plot creation error:', error);
                    }}
                }}
                
                function resizePlot() {{
                    if (plotDiv && plotDiv._fullLayout) {{
                        const newLayout = getResponsiveLayout();
                        const newTraces = getResponsiveTraces(newLayout);
                        Plotly.react('plotDiv', newTraces, newLayout);
                    }}
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
        
        return html_code

    def create_functional_timeline(quarter_labels, current_index):
        """Create a timeline where the visual dots are actually functional Streamlit buttons"""
        
        selected_label = quarter_labels[current_index]
        first_quarter = quarter_labels[0]
        last_quarter = quarter_labels[-1]
        
        # Center the timeline and make it 2/3 of screen width
        col_left, col_center, col_right = st.columns([1, 4, 1])
        
        with col_center:
            # Create the beautiful timeline header
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.95); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 16px; padding: 30px 40px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="text-align: center; margin-bottom: 25px;">
                    <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin-bottom: 12px; display: flex; align-items: center; justify-content: center; gap: 8px;">
                        <span>⏰</span>
                        <span>TIME NAVIGATION</span>
                    </div>
                    <div style="font-size: 28px; font-weight: 700; color: #6366f1; margin-bottom: 20px;">{selected_label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create the connecting line
            st.markdown("""
            <div style="position: relative; margin: 0 30px 20px 30px; height: 20px;">
                <div style="position: absolute; top: 50%; left: 0; right: 0; height: 2px; background: #e5e7eb; transform: translateY(-50%);"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create functional timeline dots using Streamlit buttons
            timeline_cols = st.columns(len(quarter_labels))
            
            for i, (col, quarter) in enumerate(zip(timeline_cols, quarter_labels)):
                with col:
                    # Use empty string to hide button text completely
                    if st.button("", 
                               key=f"q_{i}",
                               help=f"{'Current: ' + quarter if i == current_index else 'Switch to ' + quarter}",
                               disabled=(i == current_index)):
                        st.session_state.slider_index = i
                        st.rerun()
            
            # Add the start and end labels
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-top: 15px; padding: 0 30px;">
                <span style="font-size: 14px; color: #6366f1; font-weight: 600;">{first_quarter}</span>
                <span style="font-size: 14px; color: #6366f1; font-weight: 600;">{last_quarter}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Add aggressive CSS to override Streamlit's button styling
        st.markdown("""
        <style>
        /* Target buttons by their key pattern - most specific approach */
        button[data-testid="baseButton-secondary"][key*="q_"],
        button[data-testid="baseButton-primary"][key*="q_"],
        button[kind="secondary"][key*="q_"],
        button[kind="primary"][key*="q_"],
        div[data-testid="stButton"] button[key*="q_"],
        .stButton button[key*="q_"] {
            width: 20px !important;
            height: 20px !important;
            min-width: 20px !important;
            min-height: 20px !important;
            max-width: 20px !important;
            max-height: 20px !important;
            border-radius: 50% !important;
            padding: 0 !important;
            margin: 0 !important;
            border: 2px solid white !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
            transition: all 0.2s ease !important;
            font-size: 0 !important;
            line-height: 0 !important;
            overflow: hidden !important;
            background: #d1d5db !important;
            position: relative !important;
            z-index: 10 !important;
        }
        
        /* Active/Disabled button styling */
        button[data-testid="baseButton-primary"][key*="q_"][disabled],
        button[kind="primary"][key*="q_"][disabled],
        div[data-testid="stButton"] button[key*="q_"][disabled],
        .stButton button[key*="q_"][disabled] {
            background: #6366f1 !important;
            opacity: 1 !important;
            cursor: default !important;
            box-shadow: 0 3px 10px rgba(99, 102, 241, 0.4) !important;
        }
        
        /* Hover effects */
        button[data-testid="baseButton-secondary"][key*="q_"]:not([disabled]):hover,
        button[kind="secondary"][key*="q_"]:not([disabled]):hover,
        div[data-testid="stButton"] button[key*="q_"]:not([disabled]):hover,
        .stButton button[key*="q_"]:not([disabled]):hover {
            background: #9ca3af !important;
            transform: scale(1.15) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
        }
        
        /* Hide all content inside buttons */
        button[key*="q_"] * {
            display: none !important;
        }
        
        button[key*="q_"]::before,
        button[key*="q_"]::after {
            display: none !important;
        }
        
        /* Center buttons in their containers */
        div[data-testid="column"]:has(button[key*="q_"]) {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        
        .stButton:has(button[key*="q_"]) {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 0 !important;
        }
        
        /* Alternative broader targeting */
        div[data-testid="stButton"] button {
            width: 20px !important;
            height: 20px !important;
            border-radius: 50% !important;
            background: #d1d5db !important;
            border: 2px solid white !important;
            font-size: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        div[data-testid="stButton"] button[disabled] {
            background: #6366f1 !important;
            opacity: 1 !important;
        }
        
        /* Remove any text or content */
        div[data-testid="stButton"] button > div,
        div[data-testid="stButton"] button span,
        div[data-testid="stButton"] button p {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
    def create_professional_sidebar_html(river_data, selected_label, colors, quarter_labels, current_index):
        """Create the Professional River Flow sidebar HTML - cleaned up without time navigation"""
        
        tributary_stats = river_data['tributary_stats']
        co_occurrence_rate = river_data['co_occurrence_rate']
        
        # Generate story tributaries with enhanced styling
        story_tributaries_html = ""
        sorted_tributaries = sorted(tributary_stats.items(), key=lambda x: x[1]['percentage'], reverse=True)
        
        tributary_descriptions = {
            'Meditation & Mindfulness': 'Main river',
            'Anxiety & Mental Health': 'Growing tributary', 
            'Buddhism & Spirituality': 'Mountain spring',
            'Concentration & Flow': 'Fast current',
            'Practice, Retreat, & Meta': 'Deep wellspring',
            'Awareness': 'Clear stream',
            'Self-Regulation': 'Steady flow'
        }
        
        for cluster, stats in sorted_tributaries:
            description = tributary_descriptions.get(cluster, 'Flow pattern')
            
            # Calculate co-occurrence percentages
            co_occur_text = ""
            if stats['co_occurrences']:
                co_occur_list = []
                for name, weight in stats['co_occurrences'][:2]:  # Top 2
                    # Calculate percentage roughly
                    percentage = min(99, int((weight / stats['count']) * 100)) if stats['count'] > 0 else 0
                    co_occur_list.append(f"{name.split(' ')[0]} ({percentage}%)")
                co_occur_text = f"Co-occurs with: {', '.join(co_occur_list)}"
            else:
                co_occur_text = "Independent flow"
            
            story_tributaries_html += f"""
            <div style="background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(0, 0, 0, 0.1); border-radius: 12px; padding: 15px; margin-bottom: 12px; border-left: 4px solid {stats['color']};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 16px; font-weight: 700; color: #1e293b;">{stats['icon']} {cluster}</div>
                    <div style="font-size: 14px; font-weight: 700; color: {stats['color']};">{stats['percentage']:.1f}%</div>
                </div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">{description} • {stats['count']:,} discussions</div>
                <div style="font-size: 11px; color: #64748b; display: flex; align-items: center; gap: 6px;">
                    <div style="width: 6px; height: 6px; background: {stats['color']}; border-radius: 50%;"></div>
                    <span>{co_occur_text}</span>
                </div>
            </div>
            """

        html_template = f"""
        <div style="
            background: rgba(248, 250, 252, 0.95);
            border-right: 1px solid rgba(69, 183, 209, 0.2);
            padding: 20px;
            height: 600px;
            overflow-y: auto;
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        ">
            <!-- Header -->
            <div style="margin-bottom: 25px; text-align: center;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 8px;">
                    <div style="font-size: 24px;">🧠</div>
                    <div style="font-size: 20px; font-weight: 800; color: #1e293b;">Living Narrative Intelligence</div>
                </div>
                <div style="font-size: 12px; color: #64748b; line-height: 1.4;">Where meditation stories converge and flow together</div>
            </div>
            
            <!-- Flow Filters -->
            <div style="margin-bottom: 25px;">
                <div style="font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                    <span>🎚️</span>
                    <span>FLOW FILTERS</span>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <div style="font-size: 13px; color: #374151; font-weight: 600; margin-bottom: 10px;">Engagement Level</div>
                    <div style="position: relative; height: 6px; background: #e5e7eb; border-radius: 3px; margin-bottom: 8px;">
                        <div style="width: 70%; height: 100%; background: #6366f1; border-radius: 3px; position: relative;">
                            <div style="position: absolute; right: -6px; top: 50%; width: 12px; height: 12px; background: #6366f1; border: 2px solid white; border-radius: 50%; transform: translateY(-50%); box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></div>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 10px; color: #64748b;">
                        <span>Low</span>
                        <span>High</span>
                    </div>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <div style="font-size: 13px; color: #374151; font-weight: 600; margin-bottom: 10px;">Sentiment Flow</div>
                    <div style="display: flex; gap: 12px;">
                        <div style="flex: 1; padding: 12px; border-radius: 8px; text-align: center; cursor: pointer; background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); transition: all 0.2s ease;">
                            <div style="font-size: 16px; margin-bottom: 4px;">😊</div>
                            <div style="font-size: 11px; color: #16a34a; font-weight: 600;">Positive</div>
                        </div>
                        <div style="flex: 1; padding: 12px; border-radius: 8px; text-align: center; cursor: pointer; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); transition: all 0.2s ease;">
                            <div style="font-size: 16px; margin-bottom: 4px;">😔</div>
                            <div style="font-size: 11px; color: #dc2626; font-weight: 600;">Negative</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Story Tributaries -->
            <div>
                <div style="font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                    <span>📊</span>
                    <span>STORY TRIBUTARIES</span>
                </div>
                {story_tributaries_html}
            </div>
        </div>
        """
        
        return html_template

    # Main execution starts here
    selected_quarter = reverse_label_map[quarter_labels[st.session_state.slider_index]]
    selected_label = quarter_labels[st.session_state.slider_index]
    
    # Calculate river flow data
    river_data = calculate_river_flow_data(df_nodes, df_edges, selected_quarter)
    
    # Display the main title
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 0.2rem;">🌊 Professional River Flow</h1>
        <h3 style="font-size: 1.5rem; font-weight: 500; margin-top: 0;">Living Narrative Intelligence Platform</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Functional Timeline with Beautiful Dots (now actually clickable!)
    create_functional_timeline(quarter_labels, st.session_state.slider_index)
    
    # Backup navigation controls (hidden by default but available if needed)
    with st.expander("🔧 Alternative Navigation (if needed)"):
        col1, col2, col3 = st.columns([1, 8, 1])
        
        with col1:
            if st.button("⋘ Previous", key="backup_prev"):
                st.session_state.slider_index = max(0, st.session_state.slider_index - 1)
                st.rerun()

        with col3:
            if st.button("Next ⋙", key="backup_next"):
                st.session_state.slider_index = min(len(quarter_labels) - 1, st.session_state.slider_index + 1)
                st.rerun()

        with col2:
            selected_index = st.select_slider(
                label="Quarter Slider",
                options=list(range(len(quarter_labels))),
                value=st.session_state.slider_index,
                format_func=lambda x: quarter_labels[x],
                key="backup_quarter_slider"
            )
            if selected_index != st.session_state.slider_index:
                st.session_state.slider_index = selected_index
                st.rerun()
    
    # Update selected quarter and label after potential changes
    selected_quarter = reverse_label_map[quarter_labels[st.session_state.slider_index]]
    selected_label = quarter_labels[st.session_state.slider_index]
    
    # Recalculate river flow data with updated quarter
    river_data = calculate_river_flow_data(df_nodes, df_edges, selected_quarter)
    
    # Display current statistics
    st.markdown(f"""
    <div style="background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 0.75rem 1rem; border-radius: 0.25rem; margin: 1rem 0;">
        <div style="font-size: 0.9rem; color: #1565c0; line-height: 1.4;">
            📊 <strong>Quarter {selected_label}:</strong> {river_data['connected_count']:,} connected themes out of {river_data['total_nodes']:,} total 
            (<strong>{river_data['co_occurrence_rate']:.1f}%</strong> co-occurrence rate) • {river_data['total_edges']:,} narrative connections
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns: sidebar and main plot
    col_sidebar, col_plot = st.columns([1, 2])
    
    with col_sidebar:
        # Create and display the cleaned-up sidebar (no time navigation)
        sidebar_html = create_professional_sidebar_html(river_data, selected_label, colors, quarter_labels, st.session_state.slider_index)
        components.html(sidebar_html, height=600, scrolling=False)
    
    with col_plot:
        # Create and display the dynamic HTML visualization
        viz_html = create_dynamic_visualization_html(river_data, selected_label)
        components.html(viz_html, height=900, scrolling=False)
    

    # Footer
    st.markdown("""
    <div style="text-align: center; font-size: 1rem; font-weight: 600; color: #666; margin-top: 3rem; padding: 1rem; border-top: 1px solid #eee; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        Powered by Terramare ᛘ𓇳     ©2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run()