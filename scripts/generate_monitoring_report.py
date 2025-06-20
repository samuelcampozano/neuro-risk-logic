#!/usr/bin/env python3
"""
Generate HTML monitoring dashboard for model performance and data quality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from app.config import settings
from app.database import SessionLocal
from app.models.assessment import Assessment
from sqlalchemy import func
from loguru import logger


class MonitoringDashboard:
    """Generate comprehensive monitoring dashboard."""
    
    def __init__(self):
        """Initialize dashboard generator."""
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_training_history(self) -> List[Dict[str, Any]]:
        """Load model training history."""
        history_path = settings.models_dir / "training_history.json"
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                return json.load(f)
        return []
    
    def load_evaluation_reports(self) -> List[Dict[str, Any]]:
        """Load all evaluation reports."""
        reports = []
        
        for report_path in settings.models_dir.glob("evaluation_report_*.json"):
            with open(report_path, 'r') as f:
                reports.append(json.load(f))
        
        return sorted(reports, key=lambda x: x.get('evaluation_date', ''), reverse=True)
    
    def get_assessment_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get assessment statistics from database."""
        db = SessionLocal()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Total assessments
            total = db.query(Assessment).count()
            recent = db.query(Assessment).filter(
                Assessment.assessment_date >= cutoff_date
            ).count()
            
            # Risk distribution
            risk_dist = db.query(
                Assessment.risk_level, 
                func.count(Assessment.id)
            ).group_by(Assessment.risk_level).all()
            
            # Daily counts
            daily_counts = db.query(
                func.date(Assessment.assessment_date).label('date'),
                func.count(Assessment.id).label('count')
            ).filter(
                Assessment.assessment_date >= cutoff_date
            ).group_by(func.date(Assessment.assessment_date)).all()
            
            return {
                'total_assessments': total,
                'recent_assessments': recent,
                'risk_distribution': dict(risk_dist),
                'daily_counts': [(str(date), count) for date, count in daily_counts]
            }
            
        finally:
            db.close()
    
    def create_performance_plot(self, training_history: List[Dict]) -> go.Figure:
        """Create model performance over time plot."""
        if not training_history:
            return go.Figure()
        
        # Extract metrics over time
        timestamps = []
        accuracies = []
        auc_scores = []
        f1_scores = []
        
        for entry in training_history:
            if 'metrics' in entry:
                timestamps.append(entry.get('timestamp', ''))
                metrics = entry['metrics']
                accuracies.append(metrics.get('accuracy', 0))
                auc_scores.append(metrics.get('auc_roc', 0))
                f1_scores.append(metrics.get('f1_score', 0))
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=accuracies,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=auc_scores,
            mode='lines+markers',
            name='AUC-ROC',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps, y=f1_scores,
            mode='lines+markers',
            name='F1-Score',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Model Performance Over Time',
            xaxis_title='Training Date',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_data_distribution_plot(self, stats: Dict[str, Any]) -> go.Figure:
        """Create data distribution plots."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Risk Level Distribution', 'Daily Assessment Volume'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}]]
        )
        
        # Risk distribution pie chart
        if 'risk_distribution' in stats:
            labels = list(stats['risk_distribution'].keys())
            values = list(stats['risk_distribution'].values())
            
            fig.add_trace(
                go.Pie(labels=labels, values=values, hole=0.4),
                row=1, col=1
            )
        
        # Daily volume bar chart
        if 'daily_counts' in stats:
            dates = [item[0] for item in stats['daily_counts']]
            counts = [item[1] for item in stats['daily_counts']]
            
            fig.add_trace(
                go.Bar(x=dates, y=counts, name='Assessments'),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Assessment Data Distribution',
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_feature_importance_plot(self, eval_reports: List[Dict]) -> go.Figure:
        """Create feature importance comparison plot."""
        if not eval_reports or not eval_reports[0].get('feature_importance'):
            return go.Figure()
        
        # Get latest feature importance
        latest_importance = eval_reports[0]['feature_importance']
        
        # Sort by importance
        features = list(latest_importance.keys())[:10]
        importances = [latest_importance[f] for f in features]
        
        fig = go.Figure(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Top 10 Feature Importances (Latest Model)',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            template='plotly_white'
        )
        
        return fig
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report."""
        # Load data
        training_history = self.load_training_history()
        eval_reports = self.load_evaluation_reports()
        assessment_stats = self.get_assessment_statistics()
        
        # Create plots
        performance_plot = self.create_performance_plot(training_history)
        distribution_plot = self.create_data_distribution_plot(assessment_stats)
        importance_plot = self.create_feature_importance_plot(eval_reports)
        
        # Get latest metrics
        latest_metrics = {}
        if eval_reports:
            latest_metrics = eval_reports[0].get('metrics', {})
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroRiskLogic - Model Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .plot-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .alert {{
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            color: #856404;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NeuroRiskLogic Model Monitoring Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Assessments</div>
                <div class="metric-value">{assessment_stats.get('total_assessments', 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Last 30 Days</div>
                <div class="metric-value">{assessment_stats.get('recent_assessments', 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value">{latest_metrics.get('accuracy', 0):.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">AUC-ROC Score</div>
                <div class="metric-value">{latest_metrics.get('auc_roc', 0):.3f}</div>
            </div>
        </div>
        
        <div class="plot-container">
            <div id="performance-plot"></div>
        </div>
        
        <div class="plot-container">
            <div id="distribution-plot"></div>
        </div>
        
        <div class="plot-container">
            <div id="importance-plot"></div>
        </div>
        
        <div class="footer">
            <p>NeuroRiskLogic v1.0.0 | Model Version: {eval_reports[0].get('model_info', {}).get('version', 'unknown') if eval_reports else 'N/A'}</p>
        </div>
    </div>
    
    <script>
        Plotly.newPlot('performance-plot', {performance_plot.to_json()});
        Plotly.newPlot('distribution-plot', {distribution_plot.to_json()});
        Plotly.newPlot('importance-plot', {importance_plot.to_json()});
    </script>
</body>
</html>
"""
        
        return html_content
    
    def save_report(self, filename: str = None):
        """Save monitoring report to file."""
        if filename is None:
            filename = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        output_path = self.output_dir / filename
        html_content = self.generate_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Monitoring report saved to {output_path}")
        return output_path


def main():
    """Generate monitoring report."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate model monitoring dashboard"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output filename"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open report in browser"
    )
    
    args = parser.parse_args()
    
    # Generate report
    dashboard = MonitoringDashboard()
    report_path = dashboard.save_report(args.output)
    
    print(f"Monitoring report generated: {report_path}")
    
    if args.open:
        import webbrowser
        webbrowser.open(f"file://{report_path.absolute()}")


if __name__ == "__main__":
    main()