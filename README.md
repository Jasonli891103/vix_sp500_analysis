# VIX Fear Index and S&P 500 Market Analysis

This repository provides a comprehensive analysis of the VIX Fear Index and its relationship with the S&P 500 Index. It explores how spikes in market volatility, as measured by VIX, affect future returns of the stock market, particularly during times of market stress.

## 📊 Overview

- **Period Covered**: April 2007 – April 2020
- **Focus**: VIX peak detection, correlation with S&P 500, future return expectations at different VIX levels, and historical recovery patterns.
- **Goal**: Help investors understand the implications of VIX spikes and devise strategic entry points for long-term investing.

## 🗂️ Project Structure

```
├── data/                         # Historical VIX and S&P 500 data
├── main/                         # Python scripts for analysis and visualization
│   └── vix_index.py              # Main analysis script
├── result/                       # Output images and report
│   ├── vix_sp500_relationship.png
│   ├── vix_recovery_analysis.png
│   ├── vix_future_returns.png
│   ├── vix_future_returns_comparison.png
│   └── vix_analysis_report.md
└── vix_analysis_report.pdf    
```

## 📌 Key Features

### 1. VIX and S&P 500 Correlation
- Calculates the correlation coefficient between VIX and S&P 500 closing prices.
- Visualizes synchronized peaks and troughs with annotated time-series charts.

### 2. VIX Peak Detection and Recovery
- Identifies major local VIX peaks.
- Calculates subsequent market drawdown and days to recovery.
- Analyzes relationships between VIX spike intensity and recovery metrics.

### 3. Future Return Forecasting
- Evaluates future stock returns for periods ranging from 1 month to 5 years.
- Compares returns by different VIX levels (e.g., <20, 20–30, 30–40, 40–50, >50).
- Plots grouped bar charts to show risk-reward patterns.

### 4. Investment Strategy Recommendations
- Recommends gradual investment during high-VIX periods.
- Highlights long-term return advantages when buying during market fear.

## 🚀 Getting Started

### Clone the Repository
```bash
git clone https://github.com/Jasonli891103/vix_sp500_analysis.git
cd vix_sp500_analysis
```

### Set Up Environment
Install required libraries:
```bash
pip install -r requirements.txt  # (or manually install pandas, matplotlib, seaborn, etc.)
```

### Run the Analysis
```bash
cd main
python vix_index.py
```

Generated figures and reports will be saved in the `result/` directory.

## 📄 PDF Report
You can directly read the comprehensive summary and figures in:
- `result/vix_analysis_report.pdf`

## 🧠 Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn

## 📘 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
