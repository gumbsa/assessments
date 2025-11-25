#!/usr/bin/env python3
r"""
generate_stock_scores_v4.py

Updated v4: 
- 7 core factors scored out of 35 (including Industry Positioning)
- Operational Resilience removed
- Optional metrics displayed in light gray
- Uses yfinance API
- HTML report per ticker with date and ticker in filename
- Supports --infile or positional tickers
"""

import argparse, os, sys, datetime, math
import numpy as np
import yfinance as yf
from jinja2 import Template

# --- HTML Template ---
TEMPLATE_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{{ ticker }} Stock Assessment</title>
<style>
body { font-family: Arial, sans-serif; margin:20px; }
h1,h2,h3 { color:#333; }
table { border-collapse: collapse; width:100%; margin-bottom:20px;}
th, td { border:1px solid #ccc; padding:5px; text-align:left;}
th { background:#eee; }
.optional td { background:#f0f0f0; }
</style>
</head>
<body>
<h1>{{ ticker }} Stock Assessment ({{ date }})</h1>

<h2>Core Factor Scores (0 - 5)</h2>
<table>
<tr><th>Factor</th><th>Score</th><th>Rationale / Key Metrics</th></tr>
{% for f in factors %}
<tr><td>{{ f.name }}</td><td>{{ f.score }}</td><td>{{ f.rationale }}</td></tr>
{% endfor %}
<tr><th>Total (out of {{ max_total }})</th><th>{{ total_score }}</th><th></th></tr>
</table>

<h2>Optional Metrics (for context)</h2>
<table class="optional">
<tr><th>Metric</th><th>Value</th></tr>
{% for k,v in optional_metrics.items() %}
<tr><td>{{ k }}</td><td>{{ v }}</td></tr>
{% endfor %}
</table>

<h2>Underlying Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{% for k,v in metrics.items() %}
<tr><td>{{ k }}</td><td>{{ v }}</td></tr>
{% endfor %}
</table>

<p><em>Note: Core score out of 35; optional metrics displayed for transparency but not scored.</em></p>
</body>
</html>
"""

# --- Utility Functions ---
def pretty(x):
    try:
        if x is None or (isinstance(x,float) and math.isnan(x)): return "N/A"
        if isinstance(x,(int,)): x = float(x)
        absx = abs(x)
        if absx>=1e12: return f"${x/1e12:,.2f}T"
        if absx>=1e9: return f"${x/1e9:,.2f}B"
        if absx>=1e6: return f"${x/1e6:,.2f}M"
        return f"${x:,.2f}"
    except: return str(x)

def safe_get(df,candidates):
    if df is None: return None
    for c in candidates:
        if c in df.index:
            try:
                val = df.loc[c].dropna().astype(float)
                if len(val)>0: return float(val.iloc[0])
            except: continue
    return None

def compute_cagr(values):
    if values is None: return None
    vals = [v for v in values if v is not None and v>0]
    if len(vals)<2: return None
    start, end = vals[0], vals[-1]
    n = len(vals)-1
    if start<=0 or n<=0: return None
    return (end/start)**(1.0/n)-1.0

# --- Extract financials from yfinance ---
def extract_financials_yf_v3(ticker_obj):
    info = ticker_obj.info or {}
    metrics = {}
    metrics['market_cap'] = info.get('marketCap')
    metrics['trailing_pe'] = info.get('trailingPE')
    metrics['beta'] = info.get('beta')
    metrics['quote_price'] = ticker_obj.history(period="1d")['Close'].iloc[-1] if hasattr(ticker_obj,'history') else info.get('previousClose')

    income_stmt = getattr(ticker_obj,'income_stmt',None)
    cashflow_stmt = getattr(ticker_obj,'cashflow_stmt',None)
    balance_sheet = getattr(ticker_obj,'balance_sheet',None)

    # Revenue TTM
    rev_ttm = info.get('totalRevenue') or safe_get(income_stmt,['TotalRevenue','Revenue','Net Revenue'])
    metrics['revenue_ttm'] = rev_ttm

    # Free Cash Flow
    fcf = info.get('freeCashflow') or safe_get(cashflow_stmt,['FreeCashFlow','Free Cash Flow'])
    if fcf is None and cashflow_stmt is not None:
        op = safe_get(cashflow_stmt,['TotalCashFromOperatingActivities','NetCashProvidedByOperatingActivities'])
        capex = safe_get(cashflow_stmt,['CapitalExpenditures','CapEx'])
        if op is not None and capex is not None: fcf = op + capex
    metrics['free_cash_flow'] = fcf

    # Total debt
    total_debt = info.get('totalDebt')
    if total_debt is None and balance_sheet is not None:
        lt = safe_get(balance_sheet,['LongTermDebt','Long-Term Debt'])
        st = safe_get(balance_sheet,['ShortTermDebt','Short-Term Debt'])
        total_debt = (lt or 0.0)+(st or 0.0) if (lt or st) else None
    metrics['total_debt'] = total_debt

    ebit = info.get('ebit') or safe_get(income_stmt,['OperatingIncome','EBIT'])
    ebitda = info.get('ebitda')
    metrics['ebit'] = ebit
    metrics['ebitda'] = ebitda
    metrics['debt_ebitda'] = total_debt/(ebitda+1e-9) if total_debt and ebitda else None

    # Interest coverage
    interest = info.get('interestExpense') or safe_get(income_stmt,['InterestExpense'])
    metrics['interest_coverage'] = abs(ebit)/(abs(interest)+1e-9) if ebit and interest else None

    # Margins
    metrics['gross_margin'] = info.get('grossMargins')
    metrics['operating_margin'] = info.get('operatingMargins') or (metrics['ebit']/metrics['revenue_ttm'] if metrics['ebit'] and metrics['revenue_ttm'] else None)

    # ROIC
    total_equity = info.get('totalStockholderEquity') or safe_get(balance_sheet,['TotalStockholderEquity'])
    cash = info.get('totalCash') or safe_get(balance_sheet,['Cash','CashAndCashEquivalents'])
    invested_cap = (total_debt or 0.0)+(total_equity or 0.0)-(cash or 0.0) if total_debt or total_equity else None
    metrics['roic'] = (abs(ebit)*0.79)/(invested_cap+1e-9) if ebit and invested_cap else None

    # Revenue std dev 5Y
    rev_series = None
    if income_stmt is not None:
        for name in ['TotalRevenue','Revenue','Net Revenue']:
            if name in income_stmt.index:
                rev_series = income_stmt.loc[name].dropna().astype(float).values[::-1]
                break
    metrics['rev_std_5y'] = float(np.std(rev_series[:5],ddof=0)) if rev_series is not None and len(rev_series)>1 else None
    metrics['rev_cagr_5y'] = compute_cagr(rev_series[:5] if rev_series is not None else None)

    # EPS CAGR 5Y
    try:
        net_income_series = income_stmt.loc['NetIncome'].dropna().astype(float).values[::-1] if income_stmt is not None else None
    except: net_income_series = None
    shares_out = info.get('sharesOutstanding')
    eps_series = net_income_series/shares_out if net_income_series is not None and shares_out else None
    metrics['eps_cagr_5y'] = compute_cagr(eps_series[:5] if eps_series is not None else None)

    # CapEx / Revenue
    capex = safe_get(cashflow_stmt,['CapitalExpenditures','CapEx'])
    metrics['capex_rev'] = abs(capex)/(rev_ttm+1e-9) if capex and rev_ttm else None

    # ROA proxy
    total_assets = info.get('totalAssets') or safe_get(balance_sheet,['TotalAssets'])
    metrics['return_on_assets'] = (abs(ebit)/total_assets) if ebit and total_assets else None

    return metrics

# --- Scoring Functions (7 core factors) ---
def score_financial_strength(metrics):
    debt_ebitda = metrics.get('debt_ebitda')
    fcf = metrics.get('free_cash_flow')
    score = 0
    rationale = []
    if debt_ebitda is not None:
        if debt_ebitda < 1: score = 5
        elif debt_ebitda < 2: score = 4
        elif debt_ebitda < 3: score = 3
        elif debt_ebitda < 4: score = 2
        else: score = 1
        rationale.append(f"Debt/EBITDA={debt_ebitda:.2f}")
    else:
        rationale.append("Debt/EBITDA N/A")
    if fcf is not None and fcf > 0:
        rationale.append("FCF positive")
    return score, "; ".join(rationale)

def score_profitability(metrics):
    roic = metrics.get('roic')
    op_margin = metrics.get('operating_margin')
    gross_margin = metrics.get('gross_margin')
    score = 0
    rationale = []
    if roic is not None:
        if roic>0.2: score =5
        elif roic>0.15: score=4
        elif roic>0.1: score=3
        elif roic>0.05: score=2
        else: score=1
        rationale.append(f"ROIC={roic:.2%}")
    if op_margin is not None: rationale.append(f"Operating Margin={op_margin:.2%}")
    if gross_margin is not None: rationale.append(f"Gross Margin={gross_margin:.2%}")
    return score, "; ".join(rationale)

def score_competitive_advantage(metrics):
    gross_margin = metrics.get('gross_margin')
    roa = metrics.get('return_on_assets')
    score=0; rationale=[]
    if gross_margin is not None:
        if gross_margin>0.5: score=5
        elif gross_margin>0.4: score=4
        elif gross_margin>0.3: score=3
        else: score=2
        rationale.append(f"Gross Margin={gross_margin:.2%}")
    else:
        rationale.append("Gross Margin N/A")
    if roa is not None:
        rationale.append(f"ROA={roa:.2%}")
    return score,"; ".join(rationale)

def score_business_model(metrics):
    gm = metrics.get('gross_margin')
    score = 0
    rationale=[]
    if gm is not None:
        if gm>0.5: score=5
        elif gm>0.4: score=4
        elif gm>0.3: score=3
        else: score=2
        rationale.append(f"Gross Margin={gm:.2%}")
    else:
        rationale.append("Gross Margin N/A")
    return score,"; ".join(rationale)

def score_risk_profile(metrics):
    beta = metrics.get('beta')
    score=0;rationale=[]
    if beta is not None:
        if beta<0.8: score=5
        elif beta<1: score=4
        elif beta<1.2: score=3
        elif beta<1.5: score=2
        else: score=1
        rationale.append(f"Beta={beta}")
    else:
        rationale.append("Beta N/A")
    return score,"; ".join(rationale)

def score_industry_positioning(metrics):
    gm = metrics.get('gross_margin')
    score = 0
    rationale = []
    if gm is not None:
        if gm>0.6: score=5
        elif gm>0.5: score=4
        elif gm>0.4: score=3
        elif gm>0.3: score=2
        else: score=1
        rationale.append(f"Gross Margin={gm:.2%}")
    else:
        rationale.append("Gross Margin N/A")
    return score,"; ".join(rationale)

def score_market_metrics(metrics):
    fcf = metrics.get('free_cash_flow')
    market_cap = metrics.get('market_cap')
    score=0;rationale=[]
    if fcf is not None and market_cap is not None and market_cap>0:
        fcf_ratio = fcf/market_cap
        if fcf_ratio>0.1: score=5
        elif fcf_ratio>0.08: score=4
        elif fcf_ratio>0.06: score=3
        elif fcf_ratio>0.04: score=2
        else: score=1
        rationale.append(f"FCF yield={fcf_ratio:.2%}")
    else:
        rationale.append("FCF yield N/A")
    return score,"; ".join(rationale)

# --- Assess ticker ---
def assess_ticker(ticker,outdir):
    t = yf.Ticker(ticker)
    metrics = extract_financials_yf_v3(t)

    # Core scores
    fs_score, fs_r = score_financial_strength(metrics)
    pf_score, pf_r = score_profitability(metrics)
    moat_score, moat_r = score_competitive_advantage(metrics)
    bm_score, bm_r = score_business_model(metrics)
    risk_score, risk_r = score_risk_profile(metrics)
    ip_score, ip_r = score_industry_positioning(metrics)
    market_score, market_r = score_market_metrics(metrics)

    factors = [
        {"name":"Financial Strength","score":fs_score,"rationale":fs_r},
        {"name":"Profitability & Efficiency","score":pf_score,"rationale":pf_r},
        {"name":"Competitive Advantage","score":moat_score,"rationale":moat_r},
        {"name":"Business Model Quality","score":bm_score,"rationale":bm_r},
        {"name":"Risk Profile","score":risk_score,"rationale":risk_r},
        {"name":"Industry Positioning","score":ip_score,"rationale":ip_r},
        {"name":"Market Metrics / Shareholder Return","score":market_score,"rationale":market_r},
    ]

    total_score = sum([f['score'] for f in factors])
    max_total = 35

    # Optional metrics
    optional_metrics = {
        "Interest Coverage (EBIT/Interest)": metrics.get('interest_coverage'),
        "Revenue CAGR 5Y": metrics.get('rev_cagr_5y'),
        "EPS CAGR 5Y": metrics.get('eps_cagr_5y'),
        "CapEx / Revenue": metrics.get('capex_rev')
    }

    display_metrics = {
        "Market Cap": pretty(metrics.get('market_cap')),
        "Trailing PE": metrics.get('trailing_pe'),
        "Beta": metrics.get('beta'),
        "Quote Price": metrics.get('quote_price'),
        "Free Cash Flow (TTM)": pretty(metrics.get('free_cash_flow')),
        "Total Debt": pretty(metrics.get('total_debt')),
        "Debt / EBITDA": metrics.get('debt_ebitda'),
        "Interest Coverage (EBIT/Interest)": metrics.get('interest_coverage'),
        "Gross Margin": metrics.get('gross_margin'),
        "Operating Margin": metrics.get('operating_margin'),
        "Return on Assets": metrics.get('return_on_assets'),
        "ROIC (proxy)": metrics.get('roic'),
        "Revenue (TTM)": pretty(metrics.get('revenue_ttm')),
        "Revenue StdDev 5Y": metrics.get('rev_std_5y'),
        "CapEx / Revenue (proxy)": metrics.get('capex_rev')
    }

    tpl = Template(TEMPLATE_HTML)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    html = tpl.render(
        ticker=ticker.upper(),
        date=datetime.datetime.now().strftime("%Y-%m-%d"),
        factors=factors,
        total_score=total_score,
        max_total=max_total,
        optional_metrics=optional_metrics,
        metrics=display_metrics
    )
    os.makedirs(outdir,exist_ok=True)
    filename = f"{ticker.upper()}_{date_str}.html"
    outpath = os.path.join(outdir,filename)
    with open(outpath,"w",encoding="utf-8") as f:
        f.write(html)
    return outpath,total_score

# --- Generate summary HTML ---
def generate_summary_html(ticker_results, outdir):
    """ticker_results: list of dicts with keys: ticker, total_score, factors (list of dicts)"""
    template = r"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Stock Summary Assessment</title>
    <style>
    body { font-family: Arial, sans-serif; margin:20px; }
    table { border-collapse: collapse; width:100%; margin-bottom:20px;}
    th, td { border:1px solid #ccc; padding:5px; text-align:center;}
    th { background:#eee; }
    .optional { background:#f0f0f0; font-size:0.9em; }
    </style>
    </head>
    <body>
    <h1>Stock Summary Assessment ({{ date }})</h1>
    <table>
    <tr>
        <th>Ticker</th>
        {% for f in factor_names %}<th>{{ f }}</th>{% endfor %}
        <th>Total (out of {{ max_total }})</th>
    </tr>
    {% for res in results %}
    <tr>
        <td>{{ res.ticker }}</td>
        {% for f in res.factors %}<td>{{ f.score }}</td>{% endfor %}
        <td>{{ res.total_score }}</td>
    </tr>
    {% endfor %}
    </table>
    </body>
    </html>
    """
    factor_names = [f['name'] for f in ticker_results[0]['factors']]
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    tpl = Template(template)
    html = tpl.render(results=ticker_results, factor_names=factor_names, date=date_str, max_total=35)
    filename = f"StockSummary_{datetime.datetime.now().strftime('%Y%m%d')}.html"
    outpath = os.path.join(outdir, filename)
    with open(outpath,"w",encoding="utf-8") as f:
        f.write(html)
    print(f"Summary saved to {outpath}")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Generate stock quality assessments (v4) using yfinance income_stmt.")
    parser.add_argument('tickers', nargs='*', help='Ticker symbols')
    parser.add_argument('--infile', help='File with tickers (one per line)')
    parser.add_argument('--outdir', default=r"C:\Users\gumbs\OneDrive\Documents\StocksResearchData\StockResearchAssessments\Assessments")
    parser.add_argument('--summary', action='store_true', help='Generate a summary HTML report for multiple tickers')

    args = parser.parse_args()

    tickers=[]
    if args.infile:
        if not os.path.exists(args.infile):
            print("Error: infile does not exist",file=sys.stderr); sys.exit(1)
        with open(args.infile,"r",encoding="utf-8") as fh:
            tickers=[line.strip() for line in fh if line.strip()]
    if args.tickers:
        tickers.extend(args.tickers)
    tickers=list(dict.fromkeys([t.upper() for t in tickers]))
    if not tickers:
        print("No tickers provided",file=sys.stderr); sys.exit(1)

    ticker_results=[]
    for t in tickers:
        print(f"Assessing {t} ...")
        try:
            outpath,total = assess_ticker(t,args.outdir)
            print(f"Saved {t} assessment to {outpath} (Total Score={total}/{35})")
            # Capture results for summary
            t_obj = yf.Ticker(t)
            metrics = extract_financials_yf_v3(t_obj)
            fs_score, fs_r = score_financial_strength(metrics)
            pf_score, pf_r = score_profitability(metrics)
            moat_score, moat_r = score_competitive_advantage(metrics)
            bm_score, bm_r = score_business_model(metrics)
            risk_score, risk_r = score_risk_profile(metrics)
            ip_score, ip_r = score_industry_positioning(metrics)
            market_score, market_r = score_market_metrics(metrics)
            factors = [
                {"name":"Financial Strength","score":fs_score},
                {"name":"Profitability & Efficiency","score":pf_score},
                {"name":"Competitive Advantage","score":moat_score},
                {"name":"Business Model Quality","score":bm_score},
                {"name":"Risk Profile","score":risk_score},
                {"name":"Industry Positioning","score":ip_score},
                {"name":"Market Metrics / Shareholder Return","score":market_score},
            ]
            ticker_results.append({"ticker":t,"total_score":total,"factors":factors})
        except Exception as e:
            print(f"Failed for {t}: {e}")

# Generate summary if requested
    if args.summary and len(ticker_results)>1:
        generate_summary_html(ticker_results, args.outdir)

if __name__=="__main__":
    main()
