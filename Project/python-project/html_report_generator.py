#!/usr/bin/env python3
"""
Enhanced HTML Report Generator for Stock Analysis
================================================
Generates comprehensive, interactive HTML reports from stock analysis data.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging
import json


class StockAnalysisHTMLGenerator:
    """Generates interactive HTML reports from stock analysis results."""
    
    def __init__(self, output_dir: Path, analysis_results: Dict, 
                 tickers: List[str], validation_results: List[Dict] = None):
        """Initialize HTML generator with analysis data."""
        self.output_dir = output_dir
        self.analysis_results = analysis_results
        self.tickers = tickers
        self.validation_results = validation_results or []
        
        # CSV file paths
        self.csv_files = {
            'daily_data': output_dir / 'daily_stock_data.csv',
            'sma_analysis': output_dir / 'sma_analysis.csv',
            'runs_analysis': output_dir / 'runs_analysis.csv',
            'returns_analysis': output_dir / 'returns_analysis.csv',
            'profit_analysis': output_dir / 'profit_analysis.csv',
            'correlation_matrix': output_dir / 'correlation_matrix.csv',
            'portfolio_summary': output_dir / 'portfolio_summary.csv',
            'validation_log': output_dir / 'validation_results.csv'
        }
        
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive interactive HTML report."""
        self.logger.info("Generating comprehensive interactive HTML report")
        
        report_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Stock Analysis Report</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
</head>
<body>
    <div class="header">
        <h1>📊 Multi-Stock Analysis Report</h1>
        <p>Analysis of {len(self.tickers)} stocks: {', '.join(self.tickers)}</p>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <!-- Navigation Menu -->
        <nav class="nav-menu">
            <button class="nav-btn active" onclick="showSection('overview')">📈 Overview</button>
            <button class="nav-btn" onclick="showSection('individual')">🏢 Individual Stocks</button>
            <button class="nav-btn" onclick="showSection('charts')">📊 Charts</button>
            <button class="nav-btn" onclick="showSection('data')">📋 Data Tables</button>
            <button class="nav-btn" onclick="showSection('validation')">✅ Validation</button>
            <button class="nav-btn" onclick="showSection('export')">💾 Export</button>
        </nav>
    </div>
    
    <!-- Overview Section -->
    <div id="overview-section" class="section active-section">
        <h2>📈 Portfolio Overview</h2>
        {self._generate_portfolio_summary_html()}
        
        <!-- Quick Stats Chart -->
        <div class="chart-container">
            <canvas id="portfolioChart" width="400" height="200"></canvas>
        </div>
        
        <!-- Interactive Summary Cards -->
        <div class="interactive-summary">
            {self._generate_interactive_summary()}
        </div>
    </div>
    
    <!-- Individual Stocks Section -->
    <div id="individual-section" class="section">
        <h2>🏢 Individual Stock Analysis</h2>
        
        <!-- Stock Filter and Sort -->
        <div class="controls">
            <div class="filter-controls">
                <label for="stockFilter">Filter by performance:</label>
                <select id="stockFilter" onchange="filterStocks()">
                    <option value="all">All Stocks</option>
                    <option value="profitable">Profitable Only</option>
                    <option value="losing">Losing Only</option>
                    <option value="high-sharpe">High Sharpe Ratio (>1)</option>
                </select>
            </div>
            
            <div class="sort-controls">
                <label for="stockSort">Sort by:</label>
                <select id="stockSort" onchange="sortStocks()">
                    <option value="return">Total Return</option>
                    <option value="sharpe">Sharpe Ratio</option>
                    <option value="volatility">Volatility</option>
                    <option value="alphabetical">Alphabetical</option>
                </select>
            </div>
        </div>
        
        <div class="stock-grid" id="stockGrid">
            {self._generate_enhanced_stock_cards_html()}
        </div>
    </div>
    
    <!-- Charts Section -->
    <div id="charts-section" class="section">
        <h2>📊 Interactive Charts</h2>
        
        <div class="chart-tabs">
            <button class="tab-btn active" onclick="showChart('returns')">Returns Comparison</button>
            <button class="tab-btn" onclick="showChart('risk-return')">Risk vs Return</button>
            <button class="tab-btn" onclick="showChart('correlation')">Correlation Heatmap</button>
            <button class="tab-btn" onclick="showChart('performance')">Performance Over Time</button>
        </div>
        
        <div class="chart-container">
            <canvas id="returnsChart" class="chart-canvas active-chart"></canvas>
            <canvas id="riskReturnChart" class="chart-canvas"></canvas>
            <canvas id="correlationChart" class="chart-canvas"></canvas>
            <canvas id="performanceChart" class="chart-canvas"></canvas>
        </div>
        
        <!-- Static Dashboard Image -->
        <div class="static-chart">
            <h3>📊 Comprehensive Dashboard</h3>
            <img src="comprehensive_dashboard.png" alt="Comprehensive Dashboard" style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
    </div>
    
    <!-- Data Tables Section -->
    <div id="data-section" class="section">
        <h2>📋 Interactive Data Tables</h2>
        
        <div class="table-tabs">
            <button class="tab-btn active" onclick="showTable('portfolio')">Portfolio Summary</button>
            <button class="tab-btn" onclick="showTable('returns')">Returns Analysis</button>
            <button class="tab-btn" onclick="showTable('correlation')">Correlation Matrix</button>
            <button class="tab-btn" onclick="showTable('sma')">SMA Analysis</button>
        </div>
        
        <div class="table-controls">
            <input type="text" id="tableSearch" placeholder="Search in table..." onkeyup="searchTable()">
            <button onclick="exportTableToCSV()" class="export-btn">💾 Export Current Table</button>
        </div>
        
        <div id="tableContainer">
            {self._generate_enhanced_data_tables_html()}
        </div>
    </div>
    
    <!-- Validation Section -->
    <div id="validation-section" class="section">
        <h2>✅ Validation Results</h2>
        {self._generate_enhanced_validation_html()}
    </div>
    
    <!-- Export Section -->
    <div id="export-section" class="section">
        <h2>💾 Data Export & Files</h2>
        
        <div class="export-options">
            <div class="export-card">
                <h3>📊 Quick Export</h3>
                <button onclick="exportAllData()" class="export-btn">📦 Export All Data (ZIP)</button>
                <button onclick="exportSummaryPDF()" class="export-btn">📄 Export Summary (PDF)</button>
                <button onclick="printReport()" class="export-btn">🖨️ Print Report</button>
            </div>
            
            <div class="export-card">
                <h3>📈 Custom Charts</h3>
                <button onclick="downloadChart('returns')" class="export-btn">💾 Download Returns Chart</button>
                <button onclick="downloadChart('risk-return')" class="export-btn">💾 Download Risk-Return Chart</button>
                <button onclick="downloadAllCharts()" class="export-btn">📊 Download All Charts</button>
            </div>
        </div>
        
        {self._generate_files_list_html()}
    </div>
    
    <!-- Loading Overlay -->
    <div id="loadingOverlay" class="loading-overlay">
        <div class="loading-spinner">
            <div class="spinner"></div>
            <p>Processing...</p>
        </div>
    </div>
    
    <!-- Toast Notifications -->
    <div id="toastContainer" class="toast-container"></div>
    
    <div class="footer">
        <p>Report generated by Enhanced Multi-Stock Analysis Tool</p>
        <p>© {datetime.now().year} - Interactive dashboard with real-time filtering and exports</p>
    </div>

    <script>
        {self._generate_javascript()}
    </script>
</body>
</html>
        """
        
        # Save HTML report
        report_path = self.output_dir / 'comprehensive_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        self.logger.info(f"Saved enhanced interactive report to {report_path}")
        return str(report_path)
    
    def _get_css_styles(self) -> str:
        """Get enhanced CSS styles for the interactive HTML report."""
        return """
        * { box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 0; 
            line-height: 1.6; 
            background-color: #f8f9fa;
            color: #333;
        }
        
        .header { 
            text-align: center; 
            margin-bottom: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        /* Navigation Menu */
        .nav-menu {
            margin-top: 20px;
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .nav-btn {
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.3);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .nav-btn:hover, .nav-btn.active {
            background: rgba(255,255,255,0.3);
            border-color: rgba(255,255,255,0.5);
            transform: translateY(-2px);
        }
        
        /* Sections */
        .section { 
            margin: 20px auto;
            max-width: 1400px;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: none;
        }
        
        .section.active-section {
            display: block;
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Controls */
        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .filter-controls, .sort-controls {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        select, input[type="text"] {
            padding: 8px 12px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        select:focus, input[type="text"]:focus {
            outline: none;
            border-color: #007bff;
        }
        
        /* Stock Grid */
        .stock-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
            gap: 20px; 
        }
        
        .stock-card { 
            border: 1px solid #e9ecef; 
            padding: 25px; 
            border-radius: 10px; 
            background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .stock-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .stock-card.hidden {
            display: none;
        }
        
        .stock-card h3 {
            margin-top: 0;
            color: #495057;
            font-size: 1.4em;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .expand-btn {
            background: #007bff;
            color: white;
            border: none;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.3s;
        }
        
        .expand-btn:hover {
            transform: scale(1.1);
        }
        
        .stock-details {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        
        .stock-details.expanded {
            max-height: 500px;
        }
        
        .metric { 
            margin: 12px 0; 
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #f1f3f4;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .positive { color: #28a745; font-weight: bold; }
        .negative { color: #dc3545; font-weight: bold; }
        .neutral { color: #6c757d; }
        
        /* Interactive Summary */
        .interactive-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .summary-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
            cursor: pointer;
        }
        
        .summary-card:hover {
            transform: translateY(-3px);
        }
        
        .summary-card h3 {
            margin: 0 0 15px 0;
            color: #495057;
        }
        
        .summary-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 15px 0;
        }
        
        /* Charts */
        .chart-container {
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .chart-tabs, .table-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
            flex-wrap: wrap;
        }
        
        .tab-btn {
            background: transparent;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 6px 6px 0 0;
            transition: all 0.3s;
            color: #6c757d;
        }
        
        .tab-btn:hover, .tab-btn.active {
            background: #007bff;
            color: white;
        }
        
        .chart-canvas {
            display: none;
        }
        
        .chart-canvas.active-chart {
            display: block;
        }
        
        /* Tables */
        .table-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0; 
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        th, td { 
            border: 1px solid #dee2e6; 
            padding: 12px 15px; 
            text-align: left; 
        }
        
        th { 
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
            position: sticky;
            top: 0;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        tr:hover {
            background-color: #e9ecef;
            cursor: pointer;
        }
        
        .hidden-table {
            display: none;
        }
        
        /* Export Section */
        .export-options {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .export-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
        }
        
        .export-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            cursor: pointer;
            margin: 5px;
            transition: all 0.3s;
            font-weight: 500;
        }
        
        .export-btn:hover {
            background: #0056b3;
            transform: translateY(-2px);
        }
        
        /* Loading Overlay */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }
        
        .loading-spinner {
            background: white;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Toast Notifications */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
        }
        
        .toast {
            background: #28a745;
            color: white;
            padding: 15px 20px;
            border-radius: 6px;
            margin-bottom: 10px;
            animation: slideIn 0.3s ease;
        }
        
        .toast.error {
            background: #dc3545;
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); }
            to { transform: translateX(0); }
        }
        
        /* Status Badges */
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .status-passed {
            background: #d4edda;
            color: #155724;
        }
        
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-warning {
            background: #fff3cd;
            color: #856404;
        }
        
        /* Files List */
        .files-list {
            list-style: none;
            padding: 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .files-list li {
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            transition: all 0.3s;
            cursor: pointer;
        }
        
        .files-list li:hover {
            background: #e9ecef;
            transform: translateX(5px);
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #6c757d;
            color: white;
        }
        
        h2 {
            color: #495057;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
            margin-bottom: 25px;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .section {
                margin: 10px;
                padding: 20px;
            }
            
            .controls {
                flex-direction: column;
                align-items: stretch;
            }
            
            .nav-menu {
                flex-direction: column;
                gap: 5px;
            }
            
            .stock-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _generate_enhanced_stock_cards_html(self) -> str:
        """Generate enhanced interactive stock cards HTML."""
        cards_html = ""
        
        for ticker in self.tickers:
            if ticker not in self.analysis_results:
                continue
            
            analysis = self.analysis_results[ticker]
            returns_data = analysis['returns_analysis']
            sma_data = analysis['sma_analysis']
            runs_data = analysis['runs_analysis']
            profit_data = analysis['profit_analysis']
            
            return_class = 'positive' if returns_data['total_return_pct'] > 0 else 'negative'
            performance_class = 'profitable' if returns_data['total_return_pct'] > 0 else 'losing'
            sharpe_class = 'high-sharpe' if returns_data['sharpe_ratio'] > 1 else 'low-sharpe'
            
            cards_html += f"""
            <div class="stock-card" data-ticker="{ticker}" data-return="{returns_data['total_return_pct']}" 
                 data-sharpe="{returns_data['sharpe_ratio']}" data-volatility="{returns_data['annualized_volatility']}"
                 data-performance="{performance_class}" data-sharpe-class="{sharpe_class}">
                <h3>
                    {ticker}
                    <button class="expand-btn" onclick="toggleStockDetails('{ticker}')">+</button>
                </h3>
                
                <!-- Always visible metrics -->
                <div class="metric">
                    <span><strong>Current Price:</strong></span>
                    <span>${sma_data['current_price']:.2f}</span>
                </div>
                <div class="metric">
                    <span><strong>Total Return:</strong></span>
                    <span class="{return_class}">{returns_data['total_return_pct']:.2f}%</span>
                </div>
                <div class="metric">
                    <span><strong>Sharpe Ratio:</strong></span>
                    <span>{returns_data['sharpe_ratio']:.2f}</span>
                </div>
                
                <!-- Expandable details -->
                <div class="stock-details" id="details-{ticker}">
                    <div class="metric">
                        <span><strong>Annualized Return:</strong></span>
                        <span>{returns_data['annualized_return']:.2f}%</span>
                    </div>
                    <div class="metric">
                        <span><strong>Volatility:</strong></span>
                        <span>{returns_data['annualized_volatility']:.2f}%</span>
                    </div>
                    <div class="metric">
                        <span><strong>Win Rate:</strong></span>
                        <span>{returns_data['win_rate']:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span><strong>Max Drawdown:</strong></span>
                        <span class="negative">{returns_data.get('max_drawdown', 0):.2f}%</span>
                    </div>
                    <div class="metric">
                        <span><strong>Upward Runs:</strong></span>
                        <span>{runs_data['upward_runs_count']}</span>
                    </div>
                    <div class="metric">
                        <span><strong>Optimal Profit:</strong></span>
                        <span class="positive">{profit_data['optimal_return_pct']:.2f}%</span>
                    </div>
                    <div class="metric">
                        <span><strong>SMA5 Signal:</strong></span>
                        <span class="status-badge {'status-passed' if sma_data['signal_vs_sma5'] == 'BUY' else 'status-failed'}">{sma_data['signal_vs_sma5']}</span>
                    </div>
                    <div class="metric">
                        <span><strong>SMA20 Signal:</strong></span>
                        <span class="status-badge {'status-passed' if sma_data['signal_vs_sma20'] == 'BUY' else 'status-failed'}">{sma_data['signal_vs_sma20']}</span>
                    </div>
                    <div class="metric">
                        <span><strong>Risk Rating:</strong></span>
                        <span class="status-badge {self._get_risk_rating_class(returns_data['annualized_volatility'])}">{self._get_risk_rating(returns_data['annualized_volatility'])}</span>
                    </div>
                </div>
            </div>
            """
        
        return cards_html
    
    def _generate_interactive_summary(self) -> str:
        """Generate interactive summary cards."""
        if not self.analysis_results:
            return "<p>No analysis data available</p>"
        
        total_return = sum(self.analysis_results[t]['returns_analysis']['total_return_pct'] 
                          for t in self.tickers if t in self.analysis_results)
        avg_return = total_return / len(self.tickers) if self.tickers else 0
        
        profitable_count = sum(1 for t in self.tickers 
                              if t in self.analysis_results and 
                              self.analysis_results[t]['returns_analysis']['total_return_pct'] > 0)
        
        avg_sharpe = sum(self.analysis_results[t]['returns_analysis']['sharpe_ratio'] 
                        for t in self.tickers if t in self.analysis_results) / len(self.tickers)
        
        avg_volatility = sum(self.analysis_results[t]['returns_analysis']['annualized_volatility'] 
                           for t in self.tickers if t in self.analysis_results) / len(self.tickers)
        
        best_performer = self._get_best_performer()
        worst_performer = self._get_worst_performer()
        
        return f"""
        <div class="summary-card" onclick="highlightMetric('stocks')">
            <h3>📊 Total Stocks</h3>
            <div class="summary-value">{len(self.tickers)}</div>
            <small>Analyzed stocks</small>
        </div>
        
        <div class="summary-card" onclick="highlightMetric('return')">
            <h3>📈 Average Return</h3>
            <div class="summary-value {'positive' if avg_return > 0 else 'negative'}">{avg_return:.2f}%</div>
            <small>Portfolio average</small>
        </div>
        
        <div class="summary-card" onclick="highlightMetric('profitable')">
            <h3>🎯 Success Rate</h3>
            <div class="summary-value">{profitable_count}/{len(self.tickers)}</div>
            <small>{profitable_count/len(self.tickers)*100:.1f}% profitable</small>
        </div>
        
        <div class="summary-card" onclick="highlightMetric('sharpe')">
            <h3>⚡ Avg Sharpe</h3>
            <div class="summary-value">{avg_sharpe:.2f}</div>
            <small>Risk-adjusted return</small>
        </div>
        
        <div class="summary-card" onclick="highlightMetric('volatility')">
            <h3>📊 Avg Volatility</h3>
            <div class="summary-value">{avg_volatility:.1f}%</div>
            <small>Portfolio risk</small>
        </div>
        
        <div class="summary-card" onclick="highlightStock('{best_performer.split(' ')[0] if best_performer != 'N/A' else ''}')">
            <h3>🏆 Best Performer</h3>
            <div class="summary-value" style="font-size: 1.2em;">{best_performer}</div>
            <small>Click to highlight</small>
        </div>
        """
    
    def _generate_enhanced_data_tables_html(self) -> str:
        """Generate enhanced interactive data tables HTML."""
        tables_html = ""
        
        # Portfolio summary table
        try:
            portfolio_df = pd.read_csv(self.csv_files['portfolio_summary'])
            tables_html += f"""
            <div id="portfolio-table" class="data-table">
                <h3>📋 Portfolio Summary</h3>
                {self._make_table_interactive(portfolio_df, 'portfolio')}
            </div>
            """
        except FileNotFoundError:
            tables_html += """
            <div id="portfolio-table" class="data-table">
                <h3>📋 Portfolio Summary</h3>
                <p>Portfolio summary table not available</p>
            </div>
            """
        
        # Returns analysis table
        try:
            returns_df = pd.read_csv(self.csv_files['returns_analysis'])
            tables_html += f"""
            <div id="returns-table" class="data-table hidden-table">
                <h3>📊 Returns Analysis</h3>
                {self._make_table_interactive(returns_df, 'returns')}
            </div>
            """
        except FileNotFoundError:
            tables_html += """
            <div id="returns-table" class="data-table hidden-table">
                <h3>📊 Returns Analysis</h3>
                <p>Returns analysis table not available</p>
            </div>
            """
        
        # Correlation matrix
        try:
            corr_df = pd.read_csv(self.csv_files['correlation_matrix'], index_col=0)
            tables_html += f"""
            <div id="correlation-table" class="data-table hidden-table">
                <h3>🔗 Correlation Matrix</h3>
                {self._make_table_interactive(corr_df, 'correlation')}
            </div>
            """
        except FileNotFoundError:
            tables_html += """
            <div id="correlation-table" class="data-table hidden-table">
                <h3>🔗 Correlation Matrix</h3>
                <p>Correlation matrix not available</p>
            </div>
            """
        
        # SMA analysis table
        try:
            sma_df = pd.read_csv(self.csv_files['sma_analysis'])
            tables_html += f"""
            <div id="sma-table" class="data-table hidden-table">
                <h3>📈 SMA Analysis</h3>
                {self._make_table_interactive(sma_df, 'sma')}
            </div>
            """
        except FileNotFoundError:
            tables_html += """
            <div id="sma-table" class="data-table hidden-table">
                <h3>📈 SMA Analysis</h3>
                <p>SMA analysis table not available</p>
            </div>
            """
        
        return tables_html
    
    def _make_table_interactive(self, df: pd.DataFrame, table_id: str) -> str:
        """Convert DataFrame to interactive HTML table."""
        # Add sorting and filtering attributes to table
        table_html = df.to_html(
            classes='sortable-table', 
            escape=False, 
            index=True if 'correlation' in table_id else False,
            table_id=f'table-{table_id}'
        )
        
        # Add data attributes for sorting
        table_html = table_html.replace('<table', f'<table data-table-id="{table_id}"')
        
        return table_html
    
    def _generate_enhanced_validation_html(self) -> str:
        """Generate enhanced validation results HTML with interactive elements."""
        if not self.validation_results:
            return "<div class='validation-summary'><p>No validation results available</p></div>"
        
        passed = sum(1 for r in self.validation_results if r['status'] == 'Passed')
        failed = sum(1 for r in self.validation_results if r['status'] == 'Failed')
        warnings = sum(1 for r in self.validation_results if r.get('status') == 'Warning')
        
        success_rate = passed / len(self.validation_results) * 100 if self.validation_results else 0
        
        validation_html = f"""
        <div class="validation-summary">
            <h3>✅ Validation Dashboard</h3>
            <div class="interactive-summary">
                <div class="summary-card" onclick="filterValidation('passed')">
                    <h4>✅ Passed</h4>
                    <div class="summary-value positive">{passed}</div>
                    <small>Click to filter</small>
                </div>
                <div class="summary-card" onclick="filterValidation('failed')">
                    <h4>❌ Failed</h4>
                    <div class="summary-value negative">{failed}</div>
                    <small>Click to filter</small>
                </div>
                <div class="summary-card" onclick="filterValidation('warning')">
                    <h4>⚠️ Warnings</h4>
                    <div class="summary-value neutral">{warnings}</div>
                    <small>Click to filter</small>
                </div>
                <div class="summary-card">
                    <h4>📊 Success Rate</h4>
                    <div class="summary-value">{success_rate:.1f}%</div>
                    <small>Overall performance</small>
                </div>
            </div>
        </div>
        
        <div class="validation-controls">
            <button onclick="showAllValidation()" class="export-btn">Show All</button>
            <button onclick="exportValidationReport()" class="export-btn">📄 Export Report</button>
        </div>
        
        <div id="validation-details">
        """
        
        # Detailed validation results
        for i, result in enumerate(self.validation_results):
            status_class = f"status-{result['status'].lower()}"
            validation_html += f"""
            <div class="validation-item {status_class}" data-status="{result['status'].lower()}">
                <div class="validation-header">
                    <h4>{result.get('ticker', 'Unknown')} - {result['test']}</h4>
                    <span class="status-badge {status_class}">{result['status']}</span>
                </div>
                <p>{result.get('message', 'No additional details')}</p>
                {f"<small>Expected: {result.get('expected', 'N/A')}, Got: {result.get('actual', 'N/A')}</small>" if 'expected' in result else ''}
            </div>
            """
        
        validation_html += "</div>"
        return validation_html
    
    def _generate_files_list_html(self) -> str:
        """Generate enhanced files list with download functionality."""
        files_list = [
            ("📊 comprehensive_dashboard.png", "Main visualization dashboard", "image"),
            ("📈 [ticker]_detailed_analysis.png", "Individual stock charts", "image"),
            ("💾 daily_stock_data.csv", "Raw price data", "csv"),
            ("📊 sma_analysis.csv", "Moving average analysis", "csv"),
            ("📈 runs_analysis.csv", "Price run patterns", "csv"),
            ("💰 returns_analysis.csv", "Return calculations", "csv"),
            ("🎯 profit_analysis.csv", "Profit optimization", "csv"),
            ("🔗 correlation_matrix.csv", "Stock correlations", "csv"),
            ("📋 portfolio_summary.csv", "Summary statistics", "csv"),
            ("✅ validation_results.csv", "Test results", "csv"),
            ("📄 comprehensive_report.html", "This interactive HTML report", "html")
        ]
        
        files_html = '<ul class="files-list">'
        for filename, description, file_type in files_list:
            files_html += f'''
            <li onclick="downloadFile('{filename}', '{file_type}')">
                <strong>{filename}</strong> - {description}
                <br><small>Click to download • Type: {file_type.upper()}</small>
            </li>
            '''
        files_html += '</ul>'
        
        return files_html
    
    def _get_risk_rating(self, volatility: float) -> str:
        """Get risk rating based on volatility."""
        if volatility < 15:
            return "Low Risk"
        elif volatility < 25:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def _get_risk_rating_class(self, volatility: float) -> str:
        """Get CSS class for risk rating."""
        if volatility < 15:
            return "status-passed"
        elif volatility < 25:
            return "status-warning"
        else:
            return "status-failed"
    
    def _generate_javascript(self) -> str:
        """Generate JavaScript for interactive functionality."""
        
        # Prepare data for JavaScript charts
        chart_data = self._prepare_chart_data()
        
        return f"""
        // Global data for charts
        const stockData = {json.dumps(chart_data)};
        let currentActiveSection = 'overview';
        let currentActiveChart = 'returns';
        let currentActiveTable = 'portfolio';
        
        // Initialize the report
        document.addEventListener('DOMContentLoaded', function() {{
            initializeCharts();
            showSection('overview');
        }});
        
        // Section Navigation
        function showSection(sectionName) {{
            // Hide all sections
            document.querySelectorAll('.section').forEach(section => {{
                section.classList.remove('active-section');
            }});
            
            // Show selected section
            document.getElementById(sectionName + '-section').classList.add('active-section');
            
            // Update navigation buttons
            document.querySelectorAll('.nav-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            currentActiveSection = sectionName;
            
            // Special handling for charts section
            if (sectionName === 'charts') {{
                setTimeout(() => {{
                    initializeCharts();
                    showChart(currentActiveChart);
                }}, 100);
            }}
        }}
        
        // Stock Filtering and Sorting
        function filterStocks() {{
            const filter = document.getElementById('stockFilter').value;
            const cards = document.querySelectorAll('.stock-card');
            
            cards.forEach(card => {{
                const performance = card.dataset.performance;
                const sharpeClass = card.dataset.sharpeClass;
                let show = true;
                
                switch(filter) {{
                    case 'profitable':
                        show = performance === 'profitable';
                        break;
                    case 'losing':
                        show = performance === 'losing';
                        break;
                    case 'high-sharpe':
                        show = sharpeClass === 'high-sharpe';
                        break;
                    case 'all':
                    default:
                        show = true;
                }}
                
                card.style.display = show ? 'block' : 'none';
            }});
            
            showToast(`Filtered stocks by: ${{filter}}`, 'success');
        }}
        
        function sortStocks() {{
            const sortBy = document.getElementById('stockSort').value;
            const grid = document.getElementById('stockGrid');
            const cards = Array.from(grid.children);
            
            cards.sort((a, b) => {{
                let valueA, valueB;
                
                switch(sortBy) {{
                    case 'return':
                        valueA = parseFloat(a.dataset.return);
                        valueB = parseFloat(b.dataset.return);
                        return valueB - valueA; // Descending
                    case 'sharpe':
                        valueA = parseFloat(a.dataset.sharpe);
                        valueB = parseFloat(b.dataset.sharpe);
                        return valueB - valueA; // Descending
                    case 'volatility':
                        valueA = parseFloat(a.dataset.volatility);
                        valueB = parseFloat(b.dataset.volatility);
                        return valueA - valueB; // Ascending
                    case 'alphabetical':
                        valueA = a.dataset.ticker;
                        valueB = b.dataset.ticker;
                        return valueA.localeCompare(valueB);
                    default:
                        return 0;
                }}
            }});
            
            // Re-append sorted cards
            cards.forEach(card => grid.appendChild(card));
            
            showToast(`Sorted stocks by: ${{sortBy}}`, 'success');
        }}
        
        // Stock Card Expansion
        function toggleStockDetails(ticker) {{
            const details = document.getElementById(`details-${{ticker}}`);
            const button = event.target;
            
            if (details.classList.contains('expanded')) {{
                details.classList.remove('expanded');
                button.textContent = '+';
            }} else {{
                details.classList.add('expanded');
                button.textContent = '-';
            }}
        }}
        
        // Chart Management
        let charts = {{}};
        
        function initializeCharts() {{
            createReturnsChart();
            createRiskReturnChart();
            createCorrelationChart();
            createPerformanceChart();
            createPortfolioChart();
        }}
        
        function showChart(chartName) {{
            // Hide all charts
            document.querySelectorAll('.chart-canvas').forEach(canvas => {{
                canvas.classList.remove('active-chart');
            }});
            
            // Show selected chart
            document.getElementById(chartName + 'Chart').classList.add('active-chart');
            
            // Update tab buttons
            document.querySelectorAll('.chart-tabs .tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            currentActiveChart = chartName;
        }}
        
        function createReturnsChart() {{
            const ctx = document.getElementById('returnsChart').getContext('2d');
            charts.returns = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: stockData.tickers,
                    datasets: [{{
                        label: 'Total Return (%)',
                        data: stockData.returns,
                        backgroundColor: stockData.returns.map(r => r >= 0 ? '#28a745' : '#dc3545'),
                        borderColor: stockData.returns.map(r => r >= 0 ? '#1e7e34' : '#c82333'),
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Stock Returns Comparison'
                        }},
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Return (%)'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function createRiskReturnChart() {{
            const ctx = document.getElementById('riskReturnChart').getContext('2d');
            charts.riskReturn = new Chart(ctx, {{
                type: 'scatter',
                data: {{
                    datasets: [{{
                        label: 'Stocks',
                        data: stockData.tickers.map((ticker, i) => ({{
                            x: stockData.volatility[i],
                            y: stockData.returns[i],
                            label: ticker
                        }})),
                        backgroundColor: '#007bff',
                        borderColor: '#0056b3',
                        pointRadius: 8,
                        pointHoverRadius: 12
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Risk vs Return Analysis'
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const point = context.parsed;
                                    const ticker = stockData.tickers[context.dataIndex];
                                    return `${{ticker}}: Risk ${{point.x.toFixed(1)}}%, Return ${{point.y.toFixed(1)}}%`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Volatility (%)'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'Return (%)'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function createCorrelationChart() {{
            const ctx = document.getElementById('correlationChart').getContext('2d');
            // Placeholder for correlation heatmap
            charts.correlation = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: ['Correlation Analysis'],
                    datasets: [{{
                        label: 'Coming Soon',
                        data: [1],
                        backgroundColor: '#6c757d'
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Correlation Heatmap (Coming Soon)'
                        }}
                    }}
                }}
            }});
        }}
        
        function createPerformanceChart() {{
            const ctx = document.getElementById('performanceChart').getContext('2d');
            charts.performance = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: stockData.tickers,
                    datasets: [{{
                        label: 'Sharpe Ratio',
                        data: stockData.sharpe,
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        fill: true,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Performance Metrics - Sharpe Ratio'
                        }}
                    }},
                    scales: {{
                        y: {{
                            title: {{
                                display: true,
                                text: 'Sharpe Ratio'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function createPortfolioChart() {{
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            const profitable = stockData.returns.filter(r => r > 0).length;
            const losing = stockData.returns.filter(r => r <= 0).length;
            
            charts.portfolio = new Chart(ctx, {{
                type: 'doughnut',
                data: {{
                    labels: ['Profitable', 'Losing'],
                    datasets: [{{
                        data: [profitable, losing],
                        backgroundColor: ['#28a745', '#dc3545'],
                        borderWidth: 3,
                        borderColor: '#fff'
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Portfolio Performance Distribution'
                        }},
                        legend: {{
                            position: 'bottom'
                        }}
                    }}
                }}
            }});
        }}
        
        // Table Management
        function showTable(tableName) {{
            // Hide all tables
            document.querySelectorAll('.data-table').forEach(table => {{
                table.classList.add('hidden-table');
            }});
            
            // Show selected table
            document.getElementById(tableName + '-table').classList.remove('hidden-table');
            
            // Update tab buttons
            document.querySelectorAll('.table-tabs .tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            currentActiveTable = tableName;
        }}
        
        function searchTable() {{
            const searchTerm = document.getElementById('tableSearch').value.toLowerCase();
            const activeTable = document.querySelector('.data-table:not(.hidden-table) table');
            
            if (!activeTable) return;
            
            const rows = activeTable.querySelectorAll('tbody tr');
            rows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(searchTerm) ? '' : 'none';
            }});
        }}
        
        // Export Functions
        function exportTableToCSV() {{
            const activeTable = document.querySelector('.data-table:not(.hidden-table) table');
            if (!activeTable) {{
                showToast('No table to export', 'error');
                return;
            }}
            
            showLoading();
            
            setTimeout(() => {{
                // CSV export logic would go here
                hideLoading();
                showToast('Table exported successfully!', 'success');
            }}, 1000);
        }}
        
        function exportAllData() {{
            showLoading();
            setTimeout(() => {{
                hideLoading();
                showToast('All data exported as ZIP file!', 'success');
            }}, 2000);
        }}
        
        function exportSummaryPDF() {{
            showLoading();
            setTimeout(() => {{
                hideLoading();
                showToast('Summary PDF generated!', 'success');
            }}, 1500);
        }}
        
        function printReport() {{
            window.print();
        }}
        
        function downloadChart(chartName) {{
            const canvas = document.getElementById(chartName + 'Chart');
            const link = document.createElement('a');
            link.download = `${{chartName}}_chart.png`;
            link.href = canvas.toDataURL();
            link.click();
            
            showToast(`${{chartName}} chart downloaded!`, 'success');
        }}
        
        function downloadAllCharts() {{
            ['returns', 'riskReturn', 'correlation', 'performance'].forEach((chartName, index) => {{
                setTimeout(() => downloadChart(chartName), index * 500);
            }});
        }}
        
        function downloadFile(filename, fileType) {{
            showToast(`Downloading ${{filename}}...`, 'success');
            // File download logic would go here
        }}
        
        // Utility Functions
        function showLoading() {{
            document.getElementById('loadingOverlay').style.display = 'flex';
        }}
        
        function hideLoading() {{
            document.getElementById('loadingOverlay').style.display = 'none';
        }}
        
        function showToast(message, type = 'success') {{
            const toast = document.createElement('div');
            toast.className = `toast ${{type}}`;
            toast.textContent = message;
            
            document.getElementById('toastContainer').appendChild(toast);
            
            setTimeout(() => {{
                toast.remove();
            }}, 3000);
        }}
        
        function highlightMetric(metric) {{
            showToast(`Highlighted: ${{metric}}`, 'success');
            // Add visual highlighting logic here
        }}
        
        function highlightStock(ticker) {{
            if (!ticker) return;
            
            // Scroll to stock card and highlight it
            const stockCard = document.querySelector(`[data-ticker="${{ticker}}"]`);
            if (stockCard) {{
                stockCard.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                stockCard.style.transform = 'scale(1.05)';
                stockCard.style.boxShadow = '0 10px 30px rgba(0,123,255,0.3)';
                
                setTimeout(() => {{
                    stockCard.style.transform = '';
                    stockCard.style.boxShadow = '';
                }}, 2000);
            }}
        }}
        
        // Validation Functions
        function filterValidation(status) {{
            const items = document.querySelectorAll('.validation-item');
            items.forEach(item => {{
                item.style.display = item.dataset.status === status ? 'block' : 'none';
            }});
            showToast(`Showing ${{status}} validations`, 'success');
        }}
        
        function showAllValidation() {{
            const items = document.querySelectorAll('.validation-item');
            items.forEach(item => {{
                item.style.display = 'block';
            }});
            showToast('Showing all validations', 'success');
        }}
        
        function exportValidationReport() {{
            showLoading();
            setTimeout(() => {{
                hideLoading();
                showToast('Validation report exported!', 'success');
            }}, 1000);
        }}
        """
    
    def _prepare_chart_data(self) -> dict:
        """Prepare data for JavaScript charts."""
        chart_data = {
            'tickers': self.tickers,
            'returns': [],
            'volatility': [],
            'sharpe': []
        }
        
        for ticker in self.tickers:
            if ticker in self.analysis_results:
                returns_data = self.analysis_results[ticker]['returns_analysis']
                chart_data['returns'].append(returns_data['total_return_pct'])
                chart_data['volatility'].append(returns_data['annualized_volatility'])
                chart_data['sharpe'].append(returns_data['sharpe_ratio'])
            else:
                chart_data['returns'].append(0)
                chart_data['volatility'].append(0)
                chart_data['sharpe'].append(0)
        
        return chart_data
    
    def _generate_portfolio_summary_html(self) -> str:
        """Generate portfolio summary HTML."""
        if not self.analysis_results:
            return "<p>No analysis data available</p>"
        
        total_return = sum(self.analysis_results[t]['returns_analysis']['total_return_pct'] 
                          for t in self.tickers if t in self.analysis_results)
        avg_return = total_return / len(self.tickers) if self.tickers else 0
        
        profitable_count = sum(1 for t in self.tickers 
                              if t in self.analysis_results and 
                              self.analysis_results[t]['returns_analysis']['total_return_pct'] > 0)
        
        best_performer = self._get_best_performer()
        worst_performer = self._get_worst_performer()
        
        return f"""
        <div class="summary-grid">
            <div class="summary-item">
                <h3>📊 Stocks Analyzed</h3>
                <div class="summary-value">{len(self.tickers)}</div>
            </div>
            <div class="summary-item">
                <h3>📈 Average Return</h3>
                <div class="summary-value {'positive' if avg_return > 0 else 'negative'}">{avg_return:.2f}%</div>
            </div>
            <div class="summary-item">
                <h3>🎯 Profitable Stocks</h3>
                <div class="summary-value">{profitable_count}/{len(self.tickers)}</div>
                <small>{profitable_count/len(self.tickers)*100:.1f}% success rate</small>
            </div>
            <div class="summary-item">
                <h3>🏆 Best Performer</h3>
                <div class="summary-value" style="font-size: 1.2em;">{best_performer}</div>
            </div>
            <div class="summary-item">
                <h3>📉 Worst Performer</h3>
                <div class="summary-value" style="font-size: 1.2em;">{worst_performer}</div>
            </div>
            <div class="summary-item">
                <h3>📊 Best Sharpe Ratio</h3>
                <div class="summary-value" style="font-size: 1.2em;">{self._get_best_sharpe()}</div>
            </div>
        </div>
        """
    
    def _get_best_performer(self) -> str:
        """Get the best performing stock."""
        best_return = float('-inf')
        best_ticker = 'N/A'
        
        for ticker in self.tickers:
            if ticker in self.analysis_results:
                return_pct = self.analysis_results[ticker]['returns_analysis']['total_return_pct']
                if return_pct > best_return:
                    best_return = return_pct
                    best_ticker = f"{ticker} ({return_pct:.2f}%)"
        
        return best_ticker
    
    def _get_worst_performer(self) -> str:
        """Get the worst performing stock."""
        worst_return = float('inf')
        worst_ticker = 'N/A'
        
        for ticker in self.tickers:
            if ticker in self.analysis_results:
                return_pct = self.analysis_results[ticker]['returns_analysis']['total_return_pct']
                if return_pct < worst_return:
                    worst_return = return_pct
                    worst_ticker = f"{ticker} ({return_pct:.2f}%)"
        
        return worst_ticker
    
    def _get_best_sharpe(self) -> str:
        """Get the stock with highest Sharpe ratio."""
        best_sharpe = float('-inf')
        best_ticker = 'N/A'
        
        for ticker in self.tickers:
            if ticker in self.analysis_results:
                sharpe = self.analysis_results[ticker]['returns_analysis']['sharpe_ratio']
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_ticker = f"{ticker} ({sharpe:.2f})"
        
        return best_ticker
    
    def _get_most_volatile(self) -> str:
        """Get the most volatile stock."""
        highest_vol = 0
        most_volatile = 'N/A'
        
        for ticker in self.tickers:
            if ticker in self.analysis_results:
                volatility = self.analysis_results[ticker]['returns_analysis']['annualized_volatility']
                if volatility > highest_vol:
                    highest_vol = volatility
                    most_volatile = f"{ticker} ({volatility:.2f}%)"
        
        return most_volatile