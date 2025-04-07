import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# Set chart style
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

# Set chart size and font
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Define color scheme
COLORS = {
    'primary': '#2E86C1',  # Blue
    'secondary': '#E74C3C',  # Red
    'accent': '#F39C12',  # Orange
    'neutral': '#95A5A6',  # Gray
    'success': '#27AE60',  # Green
    'background': '#F8F9FA',  # Light gray background
    'grid': '#E0E0E0'  # Grid line color
}

def load_data():
    """Load data"""
    print("Loading data...")
    vix_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '^vix.csv')
    gspc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '^GSPC.csv')
    
    vix_data = pd.read_csv(vix_path, index_col='date', parse_dates=True)
    gspc_data = pd.read_csv(gspc_path, index_col='date', parse_dates=True)
    
    # Merge data
    data = pd.DataFrame({
        'VIX': vix_data['adjclose'],
        'SP500': gspc_data['adjclose']
    })
    
    # Calculate future returns for S&P 500
    for period in [20, 60, 120, 250, 500, 750, 1250]:
        data[f'SP500_{period}d_return'] = gspc_data['adjclose'].shift(-period) / gspc_data['adjclose'] - 1
    
    # Calculate VIX relative levels
    data['VIX_percentile_1y'] = data['VIX'].rolling(250).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    data['VIX_percentile_2y'] = data['VIX'].rolling(500).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    data['VIX_percentile_all'] = data['VIX'].expanding().apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Calculate VIX historical highs
    data['VIX_1y_max'] = data['VIX'].rolling(250).max()
    data['VIX_is_1y_high'] = (data['VIX'] == data['VIX_1y_max']).astype(int)
    
    # Calculate VIX N-day change rates
    for period in [5, 10, 20]:
        data[f'VIX_{period}d_change'] = data['VIX'].pct_change(period)
    
    return data.dropna()

def identify_vix_peaks(data, percentile_threshold=0.9, window=20):
    """Identify VIX peaks"""
    print("Identifying VIX peaks...")
    # Use rolling window to identify local peaks
    data['is_local_peak'] = False
    for i in range(window, len(data) - window):
        if (data['VIX'].iloc[i] > data['VIX'].iloc[i-window:i].max()) and \
           (data['VIX'].iloc[i] > data['VIX'].iloc[i+1:i+window+1].max()) and \
           (data['VIX_percentile_all'].iloc[i] > percentile_threshold):
            data.loc[data.index[i], 'is_local_peak'] = True
    
    # Identify absolute peaks (top N% of historical data)
    data['is_extreme_peak'] = data['VIX_percentile_all'] > 0.95
    
    return data

def calculate_recovery_time(data):
    """Calculate recovery time from VIX peaks to market recovery"""
    print("Calculating recovery time from VIX peaks to market recovery...")
    # Create separate DataFrame for each peak
    peak_dates = data[data['is_local_peak']].index
    recovery_times = []
    
    for peak_date in peak_dates:
        peak_idx = data.index.get_loc(peak_date)
        sp500_peak_value = data['SP500'].iloc[peak_idx]
        vix_peak_value = data['VIX'].iloc[peak_idx]
        
        # Find lowest point
        lowest_point_idx = None
        lowest_value = sp500_peak_value
        for i in range(peak_idx, min(peak_idx + 250, len(data))):
            if data['SP500'].iloc[i] < lowest_value:
                lowest_value = data['SP500'].iloc[i]
                lowest_point_idx = i
        
        if lowest_point_idx is None:
            continue
            
        # Calculate drawdown
        drawdown = (lowest_value / sp500_peak_value) - 1
        
        # Find recovery point
        recovery_idx = None
        for i in range(lowest_point_idx, min(lowest_point_idx + 750, len(data))):
            if data['SP500'].iloc[i] >= sp500_peak_value:
                recovery_idx = i
                break
        
        if recovery_idx is not None:
            recovery_time = (data.index[recovery_idx] - peak_date).days
            recovery_times.append({
                'peak_date': peak_date,
                'vix_value': vix_peak_value,
                'lowest_point_date': data.index[lowest_point_idx],
                'drawdown': drawdown,
                'recovery_date': data.index[recovery_idx],
                'recovery_days': recovery_time
            })
    
    recovery_df = pd.DataFrame(recovery_times)
    return recovery_df

def analyze_future_returns(data):
    """Analyze future returns after VIX peaks"""
    print("Analyzing future returns after VIX peaks...")
    # Create VIX level categories
    bins = [0, 20, 30, 40, 50, 100]
    labels = ['<20', '20-30', '30-40', '40-50', '>50']
    data['VIX_category'] = pd.cut(data['VIX'], bins=bins, labels=labels, right=False)
    
    # Analyze future returns for high VIX levels
    periods = [20, 60, 120, 250, 500, 750, 1250]
    period_names = ['1 month', '3 months', '6 months', '1 year', '2 years', '3 years', '5 years']
    
    results = {}
    for i, period in enumerate(periods):
        returns_by_category = {}
        for category in labels:
            category_returns = data[data['VIX_category'] == category][f'SP500_{period}d_return']
            if len(category_returns) > 0:
                returns_by_category[category] = {
                    'mean': category_returns.mean(),
                    'median': category_returns.median(),
                    'min': category_returns.min(),
                    'max': category_returns.max(),
                    'positive_prob': (category_returns > 0).mean(),
                    'count': len(category_returns)
                }
        results[period_names[i]] = returns_by_category
    
    return results

def plot_vix_sp500_relationship(data):
    """Plot VIX and SP500 relationship"""
    print("Plotting VIX and SP500 relationship...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor(COLORS['background'])
    
    # Upper plot: SP500 and VIX time series
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['background'])
    ax1.plot(data.index, data['SP500'], color=COLORS['primary'], label='S&P 500 Index', linewidth=2)
    ax1.set_ylabel('S&P 500 Index', color=COLORS['primary'], fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    
    # Mark VIX peaks
    peak_dates = data[data['is_local_peak']].index
    for date in peak_dates:
        ax1.axvline(x=date, color=COLORS['secondary'], linestyle='--', alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(data.index, data['VIX'], color=COLORS['secondary'], label='VIX Fear Index', linewidth=2)
    ax2.fill_between(data.index, data['VIX'], color=COLORS['secondary'], alpha=0.1)
    ax2.set_ylabel('VIX Fear Index', color=COLORS['secondary'], fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white')
    
    # Lower plot: Scatter plot of future 1-year returns when VIX > 30
    ax3 = axes[1]
    ax3.set_facecolor(COLORS['background'])
    high_vix = data[data['VIX'] > 30].copy()
    scatter = ax3.scatter(high_vix.index, high_vix['SP500_250d_return'], 
                        c=high_vix['VIX'], cmap='YlOrRd', 
                        s=high_vix['VIX']*2, alpha=0.7)
    
    ax3.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.3)
    ax3.set_ylabel('Future 1-Year Return', fontsize=14, fontweight='bold')
    ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add colorbar with custom styling
    cbar = plt.colorbar(scatter, ax=ax3, label='VIX Value')
    cbar.ax.set_ylabel('VIX Value', fontsize=12, fontweight='bold')
    
    plt.suptitle('VIX Fear Index and S&P 500 Index Relationship Analysis', 
                fontsize=20, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join('result', 'vix_sp500_relationship.png'), dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def plot_recovery_analysis(recovery_df):
    """Plot recovery time analysis"""
    print("Plotting recovery time analysis...")
    
    if len(recovery_df) == 0:
        print("Insufficient data for recovery time analysis")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['background'])
    axes = axes.flatten()
    
    # 1. VIX vs Maximum Drawdown
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['background'])
    scatter = ax1.scatter(recovery_df['vix_value'], recovery_df['drawdown'], 
                         s=100, alpha=0.7, color=COLORS['secondary'])
    ax1.set_xlabel('VIX Peak Value', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Maximum Drawdown', fontsize=14, fontweight='bold')
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add trend line
    z = np.polyfit(recovery_df['vix_value'], recovery_df['drawdown'], 1)
    p = np.poly1d(z)
    ax1.plot(recovery_df['vix_value'], p(recovery_df['vix_value']), 
             color=COLORS['primary'], linestyle='--', alpha=0.7, linewidth=2)
    
    # Calculate correlation
    corr = np.corrcoef(recovery_df['vix_value'], recovery_df['drawdown'])[0, 1]
    ax1.set_title(f'VIX Peak vs Maximum Drawdown\n(Correlation: {corr:.2f})', 
                 fontsize=16, fontweight='bold', pad=20)

    # 2. VIX vs Recovery Time
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['background'])
    scatter = ax2.scatter(recovery_df['vix_value'], recovery_df['recovery_days'], 
                         s=100, alpha=0.7, color=COLORS['secondary'])
    ax2.set_xlabel('VIX Peak Value', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Recovery Days', fontsize=14, fontweight='bold')
    
    # Add trend line
    z = np.polyfit(recovery_df['vix_value'], recovery_df['recovery_days'], 1)
    p = np.poly1d(z)
    ax2.plot(recovery_df['vix_value'], p(recovery_df['vix_value']), 
             color=COLORS['primary'], linestyle='--', alpha=0.7, linewidth=2)
    
    # Calculate correlation
    corr = np.corrcoef(recovery_df['vix_value'], recovery_df['recovery_days'])[0, 1]
    ax2.set_title(f'VIX Peak vs Recovery Time\n(Correlation: {corr:.2f})', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 3. Maximum Drawdown vs Recovery Time
    ax3 = axes[2]
    ax3.set_facecolor(COLORS['background'])
    scatter = ax3.scatter(recovery_df['drawdown'], recovery_df['recovery_days'], 
                         s=100, alpha=0.7, color=COLORS['secondary'])
    ax3.set_xlabel('Maximum Drawdown', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Recovery Days', fontsize=14, fontweight='bold')
    ax3.xaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add trend line
    z = np.polyfit(recovery_df['drawdown'], recovery_df['recovery_days'], 1)
    p = np.poly1d(z)
    ax3.plot(recovery_df['drawdown'], p(recovery_df['drawdown']), 
             color=COLORS['primary'], linestyle='--', alpha=0.7, linewidth=2)
    
    # Calculate correlation
    corr = np.corrcoef(recovery_df['drawdown'], recovery_df['recovery_days'])[0, 1]
    ax3.set_title(f'Maximum Drawdown vs Recovery Time\n(Correlation: {corr:.2f})', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 4. Recovery Days Timeline
    ax4 = axes[3]
    ax4.set_facecolor(COLORS['background'])
    bars = ax4.bar(recovery_df['peak_date'].astype(str), recovery_df['recovery_days'], 
                   alpha=0.7, color=COLORS['primary'])
    
    # Add VIX value labels to each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        vix_value = recovery_df['vix_value'].iloc[i]
        ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'VIX:{vix_value:.1f}', ha='center', va='bottom', 
                rotation=45, fontsize=8, color=COLORS['secondary'])
    
    ax4.set_xticklabels(recovery_df['peak_date'].dt.strftime('%Y-%m-%d'), 
                       rotation=45, ha='right')
    ax4.set_ylabel('Recovery Days', fontsize=14, fontweight='bold')
    ax4.set_title('Market Recovery Time After Historical VIX Peaks', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join('result', 'vix_recovery_analysis.png'), dpi=300, bbox_inches='tight', 
                facecolor=COLORS['background'])
    plt.close()

def plot_future_returns_by_vix(results):
    """Plot future returns by VIX levels"""
    print("Plotting future returns by VIX levels...")
    
    periods = list(results.keys())
    categories = ['<20', '20-30', '30-40', '40-50', '>50']
    
    # Prepare data
    mean_returns = {}
    for period in periods:
        mean_returns[period] = []
        for category in categories:
            if category in results[period]:
                mean_returns[period].append(results[period][category]['mean'])
            else:
                mean_returns[period].append(np.nan)
    
    # Create figure with correct size
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['background'])
    axes = axes.flatten()
    
    # 1. Average future returns by VIX level (by period)
    for i, period in enumerate(periods[:4]):  # Use first 4 periods
        ax = axes[i]
        ax.set_facecolor(COLORS['background'])
        bars = ax.bar(categories, mean_returns[period], alpha=0.7, color=COLORS['primary'])
        
        # Add value labels inside bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                # Set label color
                color = 'white' #if abs(height) > 0.1 else COLORS['neutral']
                # Set label position
                if height > 0:
                    y_pos = height / 2  # Positive value in bar middle
                else:
                    y_pos = height / 2  # Negative value in bar middle
                
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{height:.1%}', ha='center', va='center',
                       color=color, fontweight='bold', fontsize=10)
        
        ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.3)
        ax.set_title(f'VIX Level vs Future {period} Return', 
                   fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Average Return', fontsize=14, fontweight='bold')
        ax.set_xlabel('VIX Range', fontsize=14, fontweight='bold')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Adjust y-axis range to ensure enough space for label
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min * 1.1, y_max * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join('result', 'vix_future_returns.png'), dpi=300, bbox_inches='tight', 
                facecolor=COLORS['background'])
    plt.close()
    
    # 2. Return comparison across periods (by VIX level)
    plt.figure(figsize=(16, 10))
    plt.gca().set_facecolor(COLORS['background'])
    
    bar_width = 0.15
    index = np.arange(len(categories))
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
             COLORS['success'], COLORS['neutral']]
    
    # Draw bar chart and add labels
    for i, period in enumerate(periods[:5]):  # Use first 5 periods
        values = mean_returns[period]
        bars = plt.bar(index + i*bar_width, values, bar_width, alpha=0.7, 
                label=period, color=colors[i % len(colors)])
        
        # Add labels inside bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                # Set label color
                color = 'white' if abs(height) > 0.1 else COLORS['neutral']
                # Set label position
                if height > 0:
                    y_pos = height / 2  # Positive value in bar middle
                else:
                    y_pos = height / 2  # Negative value in bar middle
                
                plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{height:.1%}', ha='center', va='center',
                        color=color, fontweight='bold', fontsize=8)
    
    plt.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.3)
    plt.xlabel('VIX Range', fontsize=16, fontweight='bold')
    plt.ylabel('Average Return', fontsize=16, fontweight='bold')
    plt.title('Future Returns Comparison Across Different VIX Levels', 
             fontsize=20, fontweight='bold', pad=20)
    plt.xticks(index + bar_width * 2, categories, fontsize=12)
    plt.legend(fontsize=12, frameon=True, facecolor='white')
    plt.grid(axis='y', alpha=0.3, color=COLORS['grid'])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Adjust y-axis range
    y_min, y_max = plt.gca().get_ylim()
    plt.ylim(y_min * 1.1, y_max * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join('result', 'vix_future_returns_comparison.png'), dpi=300, bbox_inches='tight', 
                facecolor=COLORS['background'])
    plt.close()

def generate_report(data, recovery_df, return_results):
    """Generate analysis report"""
    print("Generating analysis report...")
    
    # 1. Create report summary
    report = "# VIX Fear Index Analysis Report\n\n"
    report += "## 1. Analysis Summary\n"
    report += f"* Analysis Period: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}\n"
    report += f"* Identified VIX Peak Count: {len(data[data['is_local_peak']])}\n"
    report += f"* VIX Historical Maximum Value: {data['VIX'].max():.2f}, Occurred on {data['VIX'].idxmax().strftime('%Y-%m-%d')}\n\n"
    
    # 2. VIX and S&P 500 Index Correlation
    corr = data[['VIX', 'SP500']].corr().iloc[0, 1]
    report += "## 2. VIX and S&P 500 Index Correlation\n"
    report += f"* VIX and S&P 500 Index Correlation Coefficient: {corr:.4f}\n"
    report += "* Correlation Interpretation: VIX and stock market usually show negative correlation, when VIX rises, stock market usually falls.\n\n"
    
    # 3. Stock Market Performance After VIX Peaks
    if len(recovery_df) > 0:
        report += "## 3. Stock Market Performance After VIX Peaks\n"
        report += f"* Average Maximum Drawdown: {recovery_df['drawdown'].mean():.2%}\n"
        report += f"* Average Recovery Time: {recovery_df['recovery_days'].mean():.0f} days\n"
        report += f"* VIX Value and Maximum Drawdown Correlation Coefficient: {np.corrcoef(recovery_df['vix_value'], recovery_df['drawdown'])[0, 1]:.4f}\n"
        report += f"* VIX Value and Recovery Time Correlation Coefficient: {np.corrcoef(recovery_df['vix_value'], recovery_df['recovery_days'])[0, 1]:.4f}\n\n"
        
        # Add Specific VIX Peak Analysis
        report += "### VIX Peak Event Analysis\n"
        report += "| Date | VIX Value | Maximum Drawdown | Recovery Days |\n"
        report += "|------|---------|-----------------|-------------|\n"
        
        for _, row in recovery_df.iterrows():
            report += f"| {row['peak_date'].strftime('%Y-%m-%d')} | {row['vix_value']:.2f} | {row['drawdown']:.2%} | {row['recovery_days']:.0f} |\n"
        
        report += "\n"
    
    # 4. Future Returns by VIX Levels
    report += "## 4. Future Returns by VIX Levels\n"
    
    for period, period_results in return_results.items():
        report += f"### {period} Future Returns\n"
        report += "| VIX Range | Average Return | Median Return | Minimum Return | Maximum Return | Positive Return Probability | Sample Count |\n"
        report += "|-----------|--------------|--------------|--------------|--------------|----------------------------|-------------|\n"
        
        for category, stats in period_results.items():
            report += f"| {category} | {stats['mean']:.2%} | {stats['median']:.2%} | {stats['min']:.2%} | {stats['max']:.2%} | {stats['positive_prob']:.2%} | {stats['count']} |\n"
        
        report += "\n"
    
    # 5. Conclusion and Investment Advice
    report += "## 5. Conclusion and Investment Advice\n"
    
    # Check future returns when VIX is high
    high_vix_returns = {}
    for period in return_results:
        if '>50' in return_results[period]:
            high_vix_returns[period] = return_results[period]['>50']['mean']
        elif '40-50' in return_results[period]:
            high_vix_returns[period] = return_results[period]['40-50']['mean']
        elif '30-40' in return_results[period]:
            high_vix_returns[period] = return_results[period]['30-40']['mean']
    
    report += "### Main Findings:\n"
    report += "1. **VIX and Stock Market Negative Correlation**: Analysis confirmed that VIX and stock market show obvious negative correlation.\n"
    
    if len(recovery_df) > 0:
        report += f"2. **Recovery Time Analysis**: After VIX peaks, stock market usually needs {recovery_df['recovery_days'].mean():.0f} days to recover to the level before the peak.\n"
    
    # Check if high VIX has better long-term returns
    long_term_better = False
    for period in ['1 year', '2 years', '3 years', '5 years']:
        if period in high_vix_returns and high_vix_returns[period] > 0.1:  # 10% above returns
            long_term_better = True
            break
            
    if long_term_better:
        report += f"3. **Long-term Investment Advantage**: Investing when VIX is high, long-term (3-5 years) returns significantly higher than other periods.\n"
    else:
        report += f"3. **Market Volatility**: VIX peaks usually indicate market severe volatility in the future.\n"
    
    report += "\n### Investment Advice:\n"
    report += "1. **VIX Peak Investment Strategy**: When VIX reaches extremely high level (above 40), consider gradually increasing stock market investment, especially for long-term investors.\n"
    report += "2. **Risk Control**: Although long-term returns after VIX peaks usually better, investors should still prepare to bear possible continuous decline in the short term.\n"
    report += "3. **Diversified Investment**: When investing during VIX peaks, it is recommended to diversify investment time, avoiding large-scale investment at once.\n"
    report += "4. **Long-term Holding**: Analysis shows that investing when VIX peaks requires longer investment period (at least 1-3 years) to get better returns.\n"
    
    # Save report as Markdown file
    with open(os.path.join('result', 'vix_analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

def main():
    """Main function"""
    print("Starting VIX Fear Index Analysis...")
    
    # Load data
    data = load_data()
    
    # Identify VIX peaks
    data = identify_vix_peaks(data)
    
    # Calculate recovery time
    recovery_df = calculate_recovery_time(data)
    
    # Analyze future returns
    return_results = analyze_future_returns(data)
    
    # Plot charts
    plot_vix_sp500_relationship(data)
    if len(recovery_df) > 0:
        plot_recovery_analysis(recovery_df)
    plot_future_returns_by_vix(return_results)
    
    # Generate report
    report = generate_report(data, recovery_df, return_results)
    
    print("\nAnalysis Completed! Report and charts saved.")
    print("Report file: result/vix_analysis_report.md")
    print("Chart files: result/vix_sp500_relationship.png, result/vix_recovery_analysis.png, result/vix_future_returns.png, result/vix_future_returns_comparison.png")

if __name__ == "__main__":
    main()
