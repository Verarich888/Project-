#!/usr/bin/env python3
"""
Stock Summary Statistics Visualizer
===================================
Create comprehensive visualizations for summary_statistics.csv data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockSummaryVisualizer:
    """Create comprehensive visualizations for stock summary statistics."""
    
    def __init__(self, csv_path: str, output_dir: str = "visualizations"):
        """Initialize the visualizer with data and output directory."""
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load and validate data
        self.df = self._load_data()
        self._prepare_data()
    
    def _load_data(self) -> pd.DataFrame:
        """Load and validate the summary statistics data."""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded data: {len(df)} stocks")
            print(f"✓ Columns: {list(df.columns)}")
            return df
        except Exception as e:
            raise FileNotFoundError(f"Could not load {self.csv_path}: {e}")
    
    def _prepare_data(self):
        """Prepare data for visualization."""
        # Convert date columns if they exist
        date_cols = ['Start_Date', 'End_Date']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
        
        # Ensure numeric columns are properly typed
        numeric_cols = [
            'Total_Return_Pct', 'Annualized_Volatility_Pct', 'Sharpe_Ratio',
            'Max_Drawdown_Pct', 'Start_Price', 'End_Price', 'Current_RSI'
        ]
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"✓ Data prepared for visualization")
    
    def create_performance_dashboard(self, figsize: tuple = (20, 16)) -> None:
        """Create a comprehensive performance dashboard."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Color palette for consistent styling
        colors = sns.color_palette("husl", len(self.df))
        
        # 1. Returns vs Risk Scatter Plot
        ax1 = fig.add_subplot(gs[0, :2])
        scatter = ax1.scatter(
            self.df['Annualized_Volatility_Pct'], 
            self.df['Total_Return_Pct'],
            s=100, 
            c=colors[:len(self.df)], 
            alpha=0.7,
            edgecolors='white',
            linewidth=2
        )
        
        # Add ticker labels
        for i, ticker in enumerate(self.df['Ticker']):
            ax1.annotate(ticker, 
                        (self.df['Annualized_Volatility_Pct'].iloc[i], 
                         self.df['Total_Return_Pct'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax1.set_ylabel('Total Return (%)', fontsize=12)
        ax1.set_title('Risk vs Return Profile', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=self.df['Annualized_Volatility_Pct'].median(), 
                   color='gray', linestyle='--', alpha=0.5)
        
        # 2. Sharpe Ratio Bar Chart
        ax2 = fig.add_subplot(gs[0, 2:])
        sharpe_sorted = self.df.sort_values('Sharpe_Ratio', ascending=False)
        bars = ax2.bar(range(len(sharpe_sorted)), sharpe_sorted['Sharpe_Ratio'],
                      color=colors, alpha=0.8)
        ax2.set_xticks(range(len(sharpe_sorted)))
        ax2.set_xticklabels(sharpe_sorted['Ticker'], rotation=45)
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Total Returns Comparison
        ax3 = fig.add_subplot(gs[1, :2])
        returns_sorted = self.df.sort_values('Total_Return_Pct', ascending=True)
        bars3 = ax3.barh(range(len(returns_sorted)), returns_sorted['Total_Return_Pct'],
                        color=['red' if x < 0 else 'green' for x in returns_sorted['Total_Return_Pct']],
                        alpha=0.7)
        ax3.set_yticks(range(len(returns_sorted)))
        ax3.set_yticklabels(returns_sorted['Ticker'])
        ax3.set_xlabel('Total Return (%)')
        ax3.set_title('Total Returns Comparison', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for i, bar in enumerate(bars3):
            width = bar.get_width()
            ax3.text(width + (1 if width >= 0 else -1), bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left' if width >= 0 else 'right', 
                    va='center', fontsize=10)
        
        # 4. Max Drawdown Analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        drawdown_sorted = self.df.sort_values('Max_Drawdown_Pct', ascending=True)
        bars4 = ax4.bar(range(len(drawdown_sorted)), drawdown_sorted['Max_Drawdown_Pct'],
                       color='red', alpha=0.6)
        ax4.set_xticks(range(len(drawdown_sorted)))
        ax4.set_xticklabels(drawdown_sorted['Ticker'], rotation=45)
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.set_title('Maximum Drawdown Risk', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Volatility Distribution
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.hist(self.df['Annualized_Volatility_Pct'], bins=10, 
                color='skyblue', alpha=0.7, edgecolor='black')
        ax5.axvline(self.df['Annualized_Volatility_Pct'].mean(), 
                   color='red', linestyle='--', linewidth=2, label='Mean')
        ax5.axvline(self.df['Annualized_Volatility_Pct'].median(), 
                   color='green', linestyle='--', linewidth=2, label='Median')
        ax5.set_xlabel('Annualized Volatility (%)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Volatility Distribution', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. RSI Current Levels
        ax6 = fig.add_subplot(gs[2, 2:])
        if 'Current_RSI' in self.df.columns and not self.df['Current_RSI'].isna().all():
            rsi_data = self.df.dropna(subset=['Current_RSI'])
            bars6 = ax6.bar(range(len(rsi_data)), rsi_data['Current_RSI'],
                           color=['red' if x > 70 else 'green' if x < 30 else 'gray' 
                                 for x in rsi_data['Current_RSI']],
                           alpha=0.7)
            ax6.set_xticks(range(len(rsi_data)))
            ax6.set_xticklabels(rsi_data['Ticker'], rotation=45)
            ax6.set_ylabel('RSI')
            ax6.set_title('Current RSI Levels', fontsize=14, fontweight='bold')
            ax6.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
            ax6.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'RSI data not available', transform=ax6.transAxes,
                    ha='center', va='center', fontsize=12)
            ax6.set_title('RSI Data Not Available', fontsize=14)
        
        # 7. Price Range Analysis
        ax7 = fig.add_subplot(gs[3, :2])
        price_change = ((self.df['End_Price'] - self.df['Start_Price']) / self.df['Start_Price']) * 100
        price_sorted = self.df.loc[price_change.sort_values(ascending=True).index]
        
        bars7 = ax7.barh(range(len(price_sorted)), 
                        price_change.loc[price_sorted.index],
                        color=['red' if x < 0 else 'green' for x in price_change.loc[price_sorted.index]],
                        alpha=0.7)
        ax7.set_yticks(range(len(price_sorted)))
        ax7.set_yticklabels(price_sorted['Ticker'])
        ax7.set_xlabel('Price Change (%)')
        ax7.set_title('Price Performance (Start to End)', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # 8. Summary Statistics Table
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.axis('tight')
        ax8.axis('off')
        
        # Create summary stats
        summary_stats = {
            'Metric': ['Average Return (%)', 'Average Volatility (%)', 'Average Sharpe', 
                      'Best Performer', 'Worst Performer', 'Highest Volatility', 'Lowest Volatility'],
            'Value': [
                f"{self.df['Total_Return_Pct'].mean():.2f}",
                f"{self.df['Annualized_Volatility_Pct'].mean():.2f}",
                f"{self.df['Sharpe_Ratio'].mean():.2f}",
                f"{self.df.loc[self.df['Total_Return_Pct'].idxmax(), 'Ticker']}",
                f"{self.df.loc[self.df['Total_Return_Pct'].idxmin(), 'Ticker']}",
                f"{self.df.loc[self.df['Annualized_Volatility_Pct'].idxmax(), 'Ticker']}",
                f"{self.df.loc[self.df['Annualized_Volatility_Pct'].idxmin(), 'Ticker']}"
            ]
        }
        
        table_data = pd.DataFrame(summary_stats)
        table = ax8.table(cellText=table_data.values, colLabels=table_data.columns,
                         cellLoc='center', loc='center', colWidths=[0.4, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax8.set_title('Portfolio Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_dashboard.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✓ Performance dashboard saved to {self.output_dir / 'performance_dashboard.png'}")
    
    def create_correlation_analysis(self, figsize: tuple = (12, 10)) -> None:
        """Create correlation analysis visualizations."""
        # Select numeric columns for correlation
        numeric_cols = ['Total_Return_Pct', 'Annualized_Volatility_Pct', 
                       'Sharpe_Ratio', 'Max_Drawdown_Pct']
        
        if 'Current_RSI' in self.df.columns:
            numeric_cols.append('Current_RSI')
        
        # Filter available columns
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        corr_data = self.df[available_cols].corr()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Correlation heatmap
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax1)
        ax1.set_title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Pairwise scatter plots for key relationships
        ax2.scatter(self.df['Annualized_Volatility_Pct'], self.df['Sharpe_Ratio'],
                   s=100, alpha=0.7, color='purple')
        
        # Add trend line
        z = np.polyfit(self.df['Annualized_Volatility_Pct'], self.df['Sharpe_Ratio'], 1)
        p = np.poly1d(z)
        ax2.plot(self.df['Annualized_Volatility_Pct'], 
                p(self.df['Annualized_Volatility_Pct']), "r--", alpha=0.8)
        
        ax2.set_xlabel('Annualized Volatility (%)')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Volatility vs Risk-Adjusted Returns', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add ticker labels
        for i, ticker in enumerate(self.df['Ticker']):
            ax2.annotate(ticker, 
                        (self.df['Annualized_Volatility_Pct'].iloc[i], 
                         self.df['Sharpe_Ratio'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✓ Correlation analysis saved to {self.output_dir / 'correlation_analysis.png'}")
    
    def create_risk_analysis(self, figsize: tuple = (15, 10)) -> None:
        """Create detailed risk analysis visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Risk-Return Bubble Chart
        ax1 = axes[0, 0]
        bubble_sizes = (self.df['Annualized_Volatility_Pct'] * 10)  # Scale for visibility
        scatter = ax1.scatter(self.df['Total_Return_Pct'], self.df['Max_Drawdown_Pct'],
                             s=bubble_sizes, alpha=0.6, 
                             c=self.df['Sharpe_Ratio'], cmap='RdYlGn')
        
        for i, ticker in enumerate(self.df['Ticker']):
            ax1.annotate(ticker, 
                        (self.df['Total_Return_Pct'].iloc[i], 
                         self.df['Max_Drawdown_Pct'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('Total Return (%)')
        ax1.set_ylabel('Max Drawdown (%)')
        ax1.set_title('Risk-Return Profile\n(Bubble size = Volatility, Color = Sharpe)', 
                     fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Volatility vs Sharpe Ratio
        ax2 = axes[0, 1]
        colors = ['red' if x < 0 else 'green' for x in self.df['Sharpe_Ratio']]
        ax2.bar(self.df['Ticker'], self.df['Sharpe_Ratio'], color=colors, alpha=0.7)
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Risk Distribution
        ax3 = axes[1, 0]
        metrics = ['Total_Return_Pct', 'Annualized_Volatility_Pct', 'Max_Drawdown_Pct']
        x_pos = np.arange(len(self.df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            ax3.bar(x_pos + i*width, self.df[metric], width, 
                   label=metric.replace('_', ' ').replace('Pct', '%'), alpha=0.7)
        
        ax3.set_xlabel('Stocks')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Risk Metrics Comparison', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos + width)
        ax3.set_xticklabels(self.df['Ticker'], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Risk Ranking
        ax4 = axes[1, 1]
        # Create composite risk score (higher volatility and drawdown = higher risk)
        risk_score = (self.df['Annualized_Volatility_Pct'] + abs(self.df['Max_Drawdown_Pct'])) / 2
        risk_df = pd.DataFrame({
            'Ticker': self.df['Ticker'],
            'Risk_Score': risk_score
        }).sort_values('Risk_Score', ascending=False)
        
        bars = ax4.barh(range(len(risk_df)), risk_df['Risk_Score'], 
                       color=plt.cm.Reds(np.linspace(0.3, 0.9, len(risk_df))))
        ax4.set_yticks(range(len(risk_df)))
        ax4.set_yticklabels(risk_df['Ticker'])
        ax4.set_xlabel('Composite Risk Score')
        ax4.set_title('Risk Ranking', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✓ Risk analysis saved to {self.output_dir / 'risk_analysis.png'}")
    
    def create_performance_comparison(self, figsize: tuple = (14, 8)) -> None:
        """Create performance comparison charts."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Return vs Benchmark (assuming SPY-like returns)
        benchmark_return = 10  # Assume 10% benchmark return
        ax1.scatter(self.df['Total_Return_Pct'], [benchmark_return] * len(self.df),
                   s=100, alpha=0.7, color='gray', label='Benchmark')
        ax1.scatter(self.df['Total_Return_Pct'], self.df['Total_Return_Pct'],
                   s=100, alpha=0.7, label='Actual')
        
        # Add diagonal line for reference
        max_return = max(self.df['Total_Return_Pct'].max(), benchmark_return)
        min_return = min(self.df['Total_Return_Pct'].min(), benchmark_return)
        ax1.plot([min_return, max_return], [min_return, max_return], 'k--', alpha=0.5)
        
        ax1.set_xlabel('Stock Returns (%)')
        ax1.set_ylabel('Returns (%)')
        ax1.set_title('Returns vs Benchmark', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Winners vs Losers
        winners = self.df[self.df['Total_Return_Pct'] > 0]
        losers = self.df[self.df['Total_Return_Pct'] <= 0]
        
        ax2.bar(['Winners', 'Losers'], [len(winners), len(losers)],
               color=['green', 'red'], alpha=0.7)
        ax2.set_ylabel('Number of Stocks')
        ax2.set_title(f'Winners vs Losers\n({len(winners)} vs {len(losers)})',
                     fontsize=12, fontweight='bold')
        
        # Add percentages
        total = len(self.df)
        ax2.text(0, len(winners) + 0.1, f'{len(winners)/total:.1%}', 
                ha='center', fontweight='bold')
        ax2.text(1, len(losers) + 0.1, f'{len(losers)/total:.1%}', 
                ha='center', fontweight='bold')
        
        # Volatility Buckets
        vol_buckets = pd.cut(self.df['Annualized_Volatility_Pct'], bins=3, 
                           labels=['Low', 'Medium', 'High'])
        vol_counts = vol_buckets.value_counts()
        
        ax3.pie(vol_counts.values, labels=vol_counts.index, autopct='%1.1f%%',
               colors=['lightgreen', 'orange', 'red'], alpha=0.7)
        ax3.set_title('Volatility Distribution', fontsize=12, fontweight='bold')
        
        # Performance Score (custom metric)
        # Score = Return/Volatility - |Drawdown|/100
        perf_score = (self.df['Total_Return_Pct'] / self.df['Annualized_Volatility_Pct'] 
                     - abs(self.df['Max_Drawdown_Pct']) / 100)
        
        perf_df = pd.DataFrame({
            'Ticker': self.df['Ticker'],
            'Performance_Score': perf_score
        }).sort_values('Performance_Score', ascending=True)
        
        colors = ['red' if x < 0 else 'green' for x in perf_df['Performance_Score']]
        bars = ax4.barh(range(len(perf_df)), perf_df['Performance_Score'], 
                       color=colors, alpha=0.7)
        ax4.set_yticks(range(len(perf_df)))
        ax4.set_yticklabels(perf_df['Ticker'])
        ax4.set_xlabel('Performance Score')
        ax4.set_title('Custom Performance Ranking', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✓ Performance comparison saved to {self.output_dir / 'performance_comparison.png'}")
    
    def generate_all_visualizations(self) -> None:
        """Generate all visualization types."""
        print("🎨 Creating comprehensive visualizations...")
        print("=" * 50)
        
        try:
            self.create_performance_dashboard()
            self.create_correlation_analysis()
            self.create_risk_analysis()
            self.create_performance_comparison()
            
            print("\n" + "=" * 50)
            print("✅ All visualizations completed successfully!")
            print(f"📁 Files saved to: {self.output_dir.absolute()}")
            print("\nGenerated files:")
            for file in self.output_dir.glob("*.png"):
                print(f"  📊 {file.name}")
                
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            raise
    
    def create_interactive_report(self) -> str:
        """Create an HTML report with all visualizations and insights."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Analysis Report</title>
    <style>
        body {{ font-family: 'Arial', sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .section {{ margin-bottom: 40px; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 14px; color: #6c757d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Stock Portfolio Analysis Report</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>📈 Portfolio Summary</h2>
        <div class="metric">
            <div class="metric-value">{len(self.df)}</div>
            <div class="metric-label">Total Stocks</div>
        </div>
        <div class="metric">
            <div class="metric-value {'positive' if self.df['Total_Return_Pct'].mean() > 0 else 'negative'}">
                {self.df['Total_Return_Pct'].mean():.2f}%
            </div>
            <div class="metric-label">Avg Return</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.df['Annualized_Volatility_Pct'].mean():.2f}%</div>
            <div class="metric-label">Avg Volatility</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.df['Sharpe_Ratio'].mean():.2f}</div>
            <div class="metric-label">Avg Sharpe Ratio</div>
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 Performance Dashboard</h2>
        <div class="chart">
            <img src="performance_dashboard.png" alt="Performance Dashboard">
        </div>
    </div>
    
    <div class="section">
        <h2>🔗 Correlation Analysis</h2>
        <div class="chart">
            <img src="correlation_analysis.png" alt="Correlation Analysis">
        </div>
    </div>
    
    <div class="section">
        <h2>⚠️ Risk Analysis</h2>
        <div class="chart">
            <img src="risk_analysis.png" alt="Risk Analysis">
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Performance Comparison</h2>
        <div class="chart">
            <img src="performance_comparison.png" alt="Performance Comparison">
        </div>
    </div>
    
    <div class="section">
        <h2>📋 Detailed Data</h2>
        {self.df.to_html(classes='table table-striped', table_id='data-table')}
    </div>
    
    <div class="section">
        <h2>💡 Key Insights</h2>
        <ul>
            <li><strong>Best Performer:</strong> {self.df.loc[self.df['Total_Return_Pct'].idxmax(), 'Ticker']} 
                ({self.df['Total_Return_Pct'].max():.2f}% return)</li>
            <li><strong>Worst Performer:</strong> {self.df.loc[self.df['Total_Return_Pct'].idxmin(), 'Ticker']} 
                ({self.df['Total_Return_Pct'].min():.2f}% return)</li>
            <li><strong>Most Volatile:</strong> {self.df.loc[self.df['Annualized_Volatility_Pct'].idxmax(), 'Ticker']} 
                ({self.df['Annualized_Volatility_Pct'].max():.2f}% volatility)</li>
            <li><strong>Best Risk-Adjusted:</strong> {self.df.loc[self.df['Sharpe_Ratio'].idxmax(), 'Ticker']} 
                ({self.df['Sharpe_Ratio'].max():.2f} Sharpe ratio)</li>
        </ul>
    </div>
    
</body>
</html>"""
        
        # Save HTML report
        html_path = self.output_dir / 'stock_analysis_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        


def main():
    """Main execution function with example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create visualizations from stock summary statistics')
    parser.add_argument('csv_file', help='Path to summary_statistics.csv file')
    parser.add_argument('--output-dir', default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display plots (save only)')
    
    args = parser.parse_args()
    
    # Set matplotlib backend for non-interactive mode if needed
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    
    try:
        # Create visualizer
        viz = StockSummaryVisualizer(args.csv_file, args.output_dir)
        
        # Generate all visualizations
        viz.generate_all_visualizations()
        
        # Create interactive HTML report
        html_path = viz.create_interactive_report()
        
        print(f"\n🎉 Complete! Open {html_path} in your browser to view the full report.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


def example_usage():
    """Show example usage of the visualizer."""
    print("📖 Example Usage:")
    print("-" * 50)
    print("# Basic usage:")
    print("python summary_visualizer.py summary_statistics.csv")
    print()
    print("# Custom output directory:")
    print("python summary_visualizer.py summary_statistics.csv --output-dir my_charts")
    print()
    print("# Save only (don't display):")
    print("python summary_visualizer.py summary_statistics.csv --no-show")
    print()
    print("# Programmatic usage:")
    print("viz = StockSummaryVisualizer('summary_statistics.csv')")
    print("viz.generate_all_visualizations()")
    print("viz.create_interactive_report()")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("📊 Stock Summary Statistics Visualizer")
        print("=" * 40)
        example_usage()
        sys.exit(1)
    
    sys.exit(main())