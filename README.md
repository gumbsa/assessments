# **Stock Score Assessment Generator (v4)**  
*Automated Fundamental Analysis with HTML Output*

This project generates structured, readable HTML reports for any set of stock tickers using **Yahoo Finance** data via `yfinance`. Each ticker is scored across **seven fundamental categories**, and both per-ticker and summary reports are created using external Jinja2 templates.

The reports are designed for clarity, repeatability, and mild personality—complete with a footer badge reading **“From Gumbs Enterprise”**.

---

## **Features**

- **Seven-Factor Scoring Model**
  - Financial Strength  
  - Profitability & Efficiency  
  - Competitive Advantage  
  - Business Model Quality  
  - Risk Profile  
  - Industry Positioning  
  - Market Metrics / Shareholder Return  
- **Optional Contextual Metrics** (CAGR, CapEx/Rev, Payout Ratio, etc.)
- **Fully HTML-based Output**
  - Individual per-ticker reports  
  - Summary comparison report (optional)
- **Template-Driven Architecture** (Option A)
  - Templates stored in:  
    `templates/stock_scores_ticker.html`  
    `templates/stock_scores_summary.html`
  - Automatically created if missing
- **Terminal Status Output**
  - Prints each ticker’s score live during processing
- **Hard-coded footer** (“From Gumbs Enterprise”) with centered white text on navy background

---

## **Folder Structure**

```
project/
│
├── generate_stock_scores_v4.py
│
└── templates/
    ├── stock_scores_ticker.html
    └── stock_scores_summary.html
```

The program will create the `templates/` folder and populate both template files on first run if they don't already exist.

---

## **Installation**

```bash
pip install yfinance jinja2 numpy
```

Python 3.8+ is recommended.

---

## **Usage**

### Run with explicit tickers

```bash
python generate_stock_scores_v4.py AAPL MSFT GOOGL
```

### Run using a text file of tickers

```bash
python generate_stock_scores_v4.py --infile tickers.txt
```

### Generate a summary comparison report

```bash
python generate_stock_scores_v4.py --infile mylist.txt --summary
```

### Change output directory

```bash
python generate_stock_scores_v4.py AAPL --outdir ./output/
```

---

## **Scoring Model Overview**

Each ticker receives a 0–5 score in seven fundamental categories.  
Total maximum score: **35 points**

The scoring functions blend:
- Financial ratios (Debt/EBITDA, ROIC, margins, etc.)
- Balance sheet strength
- Market valuation efficiency
- Quality-of-earnings indicators

Missing or unavailable underlying metrics automatically skip their sub-weights (Option B1 logic), reducing the chance of “N/A” cascades.

---

## **Templates**

Both templates live in the `templates/` directory:

- **stock_scores_ticker.html** — full report for a single ticker  
- **stock_scores_summary.html** — multi-ticker comparison table

You may freely customize:
- Styles  
- Layout  
- Additional fields  

The script injects values but preserves your modifications.

---

## **Output Example**

```
Assessing NVDA ...
Saved: ...\NVDA_20251129.html (Score: 27/35)
```

A summary HTML file is produced when `--summary` is enabled.

---

## **Notes / Philosophy**

This script tries to strike a balance between:
- **Automation** (fast batch scoring)
- **Transparency** (rationales explain the score)
- **Extensibility** (template system, modular functions)
- **Graceful degradation** (missing data handled without crashing)

It’s designed to be production-lean but still human-friendly.

---

## **Future Extensions**

- CSV export for all computed metrics  
- Alternative data feeds (premium fundamentals provider)  
- Visual scoring dashboards  
- Plugin scoring modules  
- Automated scheduling  

If you'd like any of these added, just ask.

