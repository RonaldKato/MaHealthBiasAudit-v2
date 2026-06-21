#!/usr/bin/env python3
"""
Display all generated visualizations from the bias audit
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from config import FIGURES_DIR

def display_all_visualizations():
    """Display all PNG visualizations from the figures directory"""
    
    if not os.path.exists(FIGURES_DIR):
        print(f" Figures directory not found: {FIGURES_DIR}")
        print("Please run the bias audit first: python run.py")
        return
    
    # Get all PNG files sorted by number
    png_files = []
    for f in os.listdir(FIGURES_DIR):
        if f.endswith('.png') and not f.startswith('._'):
            png_files.append(f)
    
    # Sort numerically by the number prefix
    def get_number(filename):
        try:
            return int(filename.split('_')[0])
        except:
            return 999
    
    png_files.sort(key=get_number)
    
    if not png_files:
        print(f" No PNG visualization files found in {FIGURES_DIR}")
        return
    
    print("\n" + "=" * 80)
    print(" MAHEALTHBIASAUDIT - VISUALIZATION GALLERY")
    print("=" * 80)
    print(f"\nFound {len(png_files)} visualizations to display.\n")
    
    # Display each visualization
    for i, png_file in enumerate(png_files, 1):
        filepath = os.path.join(FIGURES_DIR, png_file)
        
        try:
            # Load and display image
            img = mpimg.imread(filepath)
            
            # Create figure with appropriate size
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.imshow(img)
            ax.axis('off')
            
            # Add title
            title = png_file.replace('.png', '').replace('_', ' ').title()
            ax.set_title(f"Visualization {i}: {title}", fontsize=16, pad=20, fontweight='bold')
            
            plt.tight_layout()
            
            # Display
            print(f" Displaying [{i}/{len(png_files)}]: {png_file}")
            plt.show()
            
            # Ask user to continue
            if i < len(png_files):
                input("\n  Press Enter to see next visualization...")
            plt.close()
            
        except Exception as e:
            print(f" Error displaying {png_file}: {e}")
    
    print("\n" + "=" * 80)
    print(" All visualizations displayed!")
    print(f" Files also saved in: {FIGURES_DIR}")
    print("=" * 80)


def generate_report_html():
    """Generate a single HTML report with all visualizations"""
    
    if not os.path.exists(FIGURES_DIR):
        print(f"Figures directory not found: {FIGURES_DIR}")
        return
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MaHealthBiasAudit - Complete Visualization Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #1a73e8, #0d47a1);
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
            }
            .header p {
                margin: 10px 0 0;
                opacity: 0.9;
            }
            .viz-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 25px;
                padding: 20px;
            }
            .viz-card {
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .viz-card:hover {
                transform: translateY(-5px);
            }
            .viz-card h3 {
                background: #1a73e8;
                color: white;
                margin: 0;
                padding: 15px;
                font-size: 1.2em;
            }
            .viz-card img {
                width: 100%;
                height: auto;
                display: block;
            }
            .viz-card .desc {
                padding: 10px 15px;
                color: #666;
                font-size: 0.9em;
                border-top: 1px solid #eee;
            }
            .footer {
                text-align: center;
                padding: 20px;
                color: #666;
                border-top: 1px solid #ddd;
                margin-top: 30px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1> MaHealthBiasAudit</h1>
            <p>Maternal Health Bias Detection - Complete Visualization Report</p>
            <p><small>Generated: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</small></p>
        </div>
        <div class="viz-grid">
    """
    
    # Add each PNG as a card
    png_files = sorted([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png') and not f.startswith('._')])
    
    for png_file in png_files:
        title = png_file.replace('.png', '').replace('_', ' ').title()
        html_content += f"""
        <div class="viz-card">
            <h3>{title}</h3>
            <img src="{png_file}" alt="{title}">
            <div class="desc">Visualization from MaHealthBiasAudit analysis</div>
        </div>
        """
    
    html_content += """
        </div>
        <div class="footer">
            <p>MaHealthBiasAudit - Comprehensive Bias Detection for Maternal Health Data</p>
            <p>Languages: English, Swahili, Luganda, Runyankore</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(FIGURES_DIR, 'complete_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n HTML report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    import pandas as pd
    
    print("\n" + "=" * 80)
    print(" MAHEALTHBIASAUDIT - VISUALIZATION VIEWER")
    print("=" * 80)
    
    # Option 1: Display PNGs one by one
    display_all_visualizations()
    
    # Option 2: Generate HTML report
    report_path = generate_report_html()
    
    # Option 3: Open HTML report in browser
    import webbrowser
    webbrowser.open(f'file://{report_path}')
    print(f"\n HTML report opened in your browser")