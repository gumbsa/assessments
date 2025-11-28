#!/usr/bin/env python3
r"""
generate_stock_scores_v4.py

Option A variant:
- Templates stored in ./templates/ as:
    templates/stock_scores_ticker.html
    templates/stock_scores_summary.html
  (script will create them with defaults if they don't exist)
- All scoring and extraction logic preserved from prior working v4
- Console prints a score for each ticker as it is processed
- Footer added: centered white text on navy background: "From Gumbs Enterprise"
"""

import argparse
import os
import sys
import datetime
import math
import numpy as np
import yfinance as yf
from jinja2 import Template

# ---------------- Template filenames (Option A) ----------------
TEMPLATES_DIR = "templates"
TEMPLATE_TICKER_FN = "stock_scores_ticker.html"
TEMPLATE_SUMMARY_FN = "stock_scores_summary.html"

TEMPLATE_TICKER_PATH = os.path.join(TEMPLATES_DIR, TEMPLATE_TICKER_FN)
TEMPLATE_SUMMARY_PATH = os.path.join(TEMPLATES_DIR, TEMPLATE_SUMMARY_FN)

# ---------------- Default template contents ----------------
DEFAULT_TICKER_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{{ ticker }} Stock Assessment</title>
<style>
body { font-family: Arial, sans-serif; margin:20px; }
h1,h2,h3 { color:#333; }
table { border-collapse: collapse; width:100%; margin-bottom:20px;}
th, td { border:1px solid #ccc; padding:6px; text-align:left;}
th { background:#eee; }
.optional td { background:#f7f7f7; }
.mono { font-family: monospace; }
.footer { text-align:center; background:#001f4d; color:white; padding:10px; margin-top:20px; }
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
<tr><td>{{ k }}</td><td class="mono">{{ v }}</td></tr>
{% endfor %}
</table>

<h2>Underlying Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{% for k,v in metrics.items() %}
<tr><td>{{ k }}</td><td class="mono">{{ v }}</td></tr>
{% endfor %}
</table>

<p><em>Note: Core score out of {{ max_total }}; optional metrics displayed for transparency but not scored.</em></p>

<div class="footer">From Gumbs Enterprise</div>
</body>
</html>
"""

DEFAULT_SUMMARY_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Stock Summary Assessment</title>
<style>
body { font-family: Arial, sans-serif; margin:20px; }
table { border-collapse: collapse; width:100%; margin-bottom:20px;}
th, td { border:1px solid #ccc; padding:6px; text-align:center;}
th { background:#eee; }
.footer { text-align:center; background:#001f4d; color:white; padding:10px; margin-top:20px; }
a { text-decoration:none; color: #003366; }
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
    <td><a href="{{ res.file }}">{{ res.ticker }}</a></td>
    {% for f in res.factors %}<td>{{ f.score }}</td>{% endfor %}
    <td>{{ res.total_score }}</td>
</tr>
{% endfor %}
</table>

<div class="footer">From Gumbs Enterprise</div>
</body>
</html>
"""

# ---------------- Ensure templates exist ----------------
def ensure_templates():
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    if not os.path.exists(TEMPLATE_TICKER_PATH):
        with open(TEMPLATE_TICKER_PATH, "w", encoding="utf-8") as fh:
            fh.write(DEFAULT_TICKER_TEMPLATE.strip())
    if not os.path.exists(TEMPLATE_SUMMARY_PATH):
        with open(TEMPLATE_SUMMARY_PATH, "w", encoding="utf-8") as fh:
            fh.write(DEFAULT_SUMMARY_TEMPLATE.strip())

# ---------------- Utilities (unchanged) ----------------
def pretty(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "N/A"
        if isinstance(x, (int,)):
            x = float(x)
        absx = abs(x)
        if absx >= 1e12:
            return f"${x/1e12:,.2f}T"
        if absx >= 1e9:
            return f"${x/1e9:,.2f}B"
        if absx >= 1e6:
            return f"${x/1e6:,.2f}M"
        return f"${x:,.2f}"
    except Exception:
        return str(x)

def safe_get(df, candidates):
    if df is None:
        return None
    for c in candidates:
        if c in df.index:
            try:
                val = df.loc[c].dropna().astype(float)
                if len(val) > 0:
                    return float(val.iloc[0])
            except Exception:
                try:
                    val = float(df.loc[c])
                    return val
                except Exception:
                    continue
    return None

def compute_cagr(values):
    if values is None:
        return None
    vals = [v for v in values if v is not None and v > 0]
    if len(vals) < 2:
        return None
    start, end = vals[0], vals[-1]
    n = len(vals) - 1
    if start <= 0 or n <= 0:
        return None
    return (end / start) ** (1.0 / n) - 1.0

# ---------------- Robust attribute helpers (unchanged) ----------------
def get_income_df(ticker_obj):
    for attr in ('income_stmt', 'financials', 'income_statement', 'income'):
        df = getattr(ticker_obj, attr, None)
        if df is not None:
            return df
    return None

def get_cashflow_df(ticker_obj):
    for attr in ('cashflow_stmt', 'cashflow', 'cashflowStatement', 'cash_flow'):
        df = getattr(ticker_obj, attr, None)
        if df is not None:
            return df
    return None

def get_balance_df(ticker_obj):
    for attr in ('balance_sheet', 'balancesheet', 'balanceSheet', 'balance'):
        df = getattr(ticker_obj, attr, None)
        if df is not None:
            return df
    return None

# ---------------- Metric extraction (unchanged) ----------------
def extract_financials_yf_v3(ticker_obj):
    info = ticker_obj.info or {}
    metrics = {}

    metrics['market_cap'] = info.get('marketCap')
    metrics['trailing_pe'] = info.get('trailingPE')
    metrics['beta'] = info.get('beta')
    try:
        metrics['quote_price'] = ticker_obj.history(period="1d")['Close'].iloc[-1]
    except Exception:
        metrics['quote_price'] = info.get('previousClose')

    income_stmt = get_income_df(ticker_obj)
    cashflow_stmt = get_cashflow_df(ticker_obj)
    balance_sheet = get_balance_df(ticker_obj)

    rev_ttm = info.get('totalRevenue') or safe_get(income_stmt, ['TotalRevenue', 'Revenue', 'Net Revenue', 'Total Revenue'])
    metrics['revenue_ttm'] = rev_ttm

    fcf = info.get('freeCashflow') or safe_get(cashflow_stmt, ['FreeCashFlow', 'Free Cash Flow', 'FreeCashFlowFromContinuingOperations'])
    if fcf is None and cashflow_stmt is not None:
        op = safe_get(cashflow_stmt, ['TotalCashFromOperatingActivities', 'NetCashProvidedByOperatingActivities', 'OperatingCashFlow'])
        capex = safe_get(cashflow_stmt, ['CapitalExpenditures', 'CapEx'])
        if op is not None and capex is not None:
            fcf = op + capex
    metrics['free_cash_flow'] = fcf

    total_debt = info.get('totalDebt')
    if total_debt is None and balance_sheet is not None:
        lt = safe_get(balance_sheet, ['LongTermDebt', 'Long-Term Debt'])
        st = safe_get(balance_sheet, ['ShortTermDebt', 'Short-Term Debt'])
        if lt is not None or st is not None:
            total_debt = (lt or 0.0) + (st or 0.0)
    metrics['total_debt'] = total_debt

    ebit = info.get('ebit') or safe_get(income_stmt, ['OperatingIncome', 'EBIT', 'EarningsBeforeInterestAndTaxes'])
    ebitda = info.get('ebitda') or safe_get(income_stmt, ['EBITDA'])
    metrics['ebit'] = ebit
    metrics['ebitda'] = ebitda
    metrics['debt_ebitda'] = (total_debt / (abs(ebitda) + 1e-9)) if total_debt is not None and ebitda is not None else None

    interest = info.get('interestExpense') or safe_get(income_stmt, ['InterestExpense', 'Interest Paid', 'InterestPaid'])
    metrics['interest_coverage'] = (abs(ebit) / (abs(interest) + 1e-9)) if ebit is not None and interest is not None else None

    metrics['gross_margin'] = info.get('grossMargins')
    metrics['operating_margin'] = info.get('operatingMargins') or ((metrics['ebit'] / metrics['revenue_ttm']) if metrics.get('ebit') is not None and metrics.get('revenue_ttm') else None)
    metrics['profit_margin'] = info.get('profitMargins')

    total_assets = info.get('totalAssets') or safe_get(balance_sheet, ['TotalAssets', 'Assets'])
    total_equity = info.get('totalStockholderEquity') or safe_get(balance_sheet, ['TotalStockholderEquity', 'Total shareholders equity', 'Shareholders Equity'])
    metrics['financial_leverage'] = (total_assets / (total_equity + 1e-9)) if (total_assets is not None and total_equity is not None) else None

    curr_assets = safe_get(balance_sheet, ['TotalCurrentAssets', 'CurrentAssets', 'Current Assets'])
    curr_liab = safe_get(balance_sheet, ['TotalCurrentLiabilities', 'CurrentLiabilities', 'Current Liabilities'])
    metrics['current_ratio'] = (curr_assets / (curr_liab + 1e-9)) if (curr_assets is not None and curr_liab is not None) else None

    cash = info.get('totalCash') or safe_get(balance_sheet, ['Cash', 'CashAndCashEquivalents', 'Cash & Equivalents'])
    invested_cap = None
    if (total_debt is not None) or (total_equity is not None):
        invested_cap = ((total_debt or 0.0) + (total_equity or 0.0) - (cash or 0.0))
    metrics['roic'] = ((abs(ebit) * (1 - 0.21)) / (invested_cap + 1e-9)) if (ebit is not None and invested_cap is not None) else None

    rev_series = None
    if income_stmt is not None:
        for name in ['TotalRevenue', 'Revenue', 'Net Revenue', 'totalRevenue']:
            if name in income_stmt.index:
                try:
                    rev_series = income_stmt.loc[name].dropna().astype(float).values[::-1]
                except Exception:
                    rev_series = None
                break
    metrics['rev_cagr_5y'] = compute_cagr(rev_series[:5] if rev_series is not None else None)
    metrics['rev_std_5y'] = (float(np.std(rev_series[:5], ddof=0)) if rev_series is not None and len(rev_series) > 1 else None)

    try:
        net_income_series = (income_stmt.loc['NetIncome'].dropna().astype(float).values[::-1]
                             if income_stmt is not None and ('NetIncome' in income_stmt.index or 'Net Income' in income_stmt.index) else None)
    except Exception:
        net_income_series = None
    shares_out = info.get('sharesOutstanding')
    eps_series = (net_income_series / shares_out) if (net_income_series is not None and shares_out) else None
    metrics['eps_cagr_5y'] = compute_cagr(eps_series[:5] if eps_series is not None else None)

    capex = safe_get(cashflow_stmt, ['CapitalExpenditures', 'CapEx'])
    metrics['capex_rev'] = (abs(capex) / (rev_ttm + 1e-9)) if (capex is not None and rev_ttm is not None) else None

    dividends_paid = safe_get(cashflow_stmt, ['DividendsPaid', 'Dividends Paid', 'CommonStockDividendsPaid']) if cashflow_stmt is not None else None
    if dividends_paid is None:
        dividend_rate = info.get('dividendRate')
        if dividend_rate is not None and shares_out:
            dividends_paid = dividend_rate * shares_out
    net_income_latest = (net_income_series[-1] if (net_income_series is not None and len(net_income_series) > 0) else safe_get(income_stmt, ['NetIncome', 'Net Income', 'NetIncomeLoss']))
    payout_ratio = (abs(dividends_paid) / (abs(net_income_latest) + 1e-9)) if (dividends_paid is not None and net_income_latest is not None) else None
    metrics['dividend_payout_ratio'] = payout_ratio

    metrics['return_on_assets'] = info.get('returnOnAssets') or ((abs(ebit) / total_assets) if (ebit is not None and total_assets is not None) else None)

    return metrics

# ---------------- Scoring functions (unchanged B1 & others) ----------------
def _map_debt_ebitda_to_0_5(de):
    if de is None:
        return None
    if de < 1.0:
        return 5.0
    elif de < 2.0:
        return 4.0
    elif de < 3.0:
        return 3.0
    elif de < 4.0:
        return 2.0
    elif de < 6.0:
        return 1.0
    else:
        return 0.0

def _map_leverage_to_0_5(lv):
    if lv is None:
        return None
    if lv < 1.5:
        return 5.0
    elif lv < 3.0:
        return 4.0
    elif lv < 5.0:
        return 3.0
    elif lv < 8.0:
        return 2.0
    elif lv < 12.0:
        return 1.0
    else:
        return 0.0

def _map_current_ratio_to_0_5(cr):
    if cr is None:
        return None
    if cr >= 2.0:
        return 5.0
    elif cr >= 1.5:
        return 4.0
    elif cr >= 1.2:
        return 3.0
    elif cr >= 1.0:
        return 2.0
    else:
        return 0.0

def _map_payout_to_0_5(payout):
    if payout is None:
        return None
    if payout < 0.25:
        return 5.0
    elif payout < 0.4:
        return 4.0
    elif payout < 0.6:
        return 3.0
    elif payout < 0.9:
        return 2.0
    else:
        return 0.0

def score_financial_strength(metrics):
    de = metrics.get('debt_ebitda')
    lev = metrics.get('financial_leverage')
    cr = metrics.get('current_ratio')
    payout = metrics.get('dividend_payout_ratio')
    fcf = metrics.get('free_cash_flow')

    s_de = _map_debt_ebitda_to_0_5(de)
    s_lev = _map_leverage_to_0_5(lev)
    s_cr = _map_current_ratio_to_0_5(cr)
    s_payout = _map_payout_to_0_5(payout)

    weights = {'de': 0.40, 'lev': 0.25, 'cr': 0.20, 'payout': 0.15}
    subs = {}
    if s_de is not None:
        subs['de'] = (s_de, weights['de'])
    if s_lev is not None:
        subs['lev'] = (s_lev, weights['lev'])
    if s_cr is not None:
        subs['cr'] = (s_cr, weights['cr'])
    if s_payout is not None:
        subs['payout'] = (s_payout, weights['payout'])

    rationale_parts = []
    if de is not None:
        rationale_parts.append(f"Debt/EBITDA={de:.2f}")
    else:
        rationale_parts.append("Debt/EBITDA N/A")
    if lev is not None:
        rationale_parts.append(f"FinLev(A/E)={lev:.2f}")
    else:
        rationale_parts.append("FinancialLeverage N/A")
    if cr is not None:
        rationale_parts.append(f"CurrentRatio={cr:.2f}")
    else:
        rationale_parts.append("CurrentRatio N/A")
    if payout is not None:
        try:
            rationale_parts.append(f"Payout={payout:.2%}")
        except Exception:
            rationale_parts.append(f"Payout={payout}")
    else:
        rationale_parts.append("DividendPayout N/A")
    if fcf is not None and fcf > 0:
        rationale_parts.append("FCF positive (TTM)")

    if not subs:
        return 0, "; ".join(rationale_parts)

    total_w = sum([w for (_, w) in subs.values()])
    if total_w <= 0:
        return 0, "; ".join(rationale_parts)

    weighted = sum([score * (w / total_w) for (score, w) in subs.values()])
    final_score = int(round(max(0.0, min(5.0, weighted))))
    return final_score, "; ".join(rationale_parts)

def score_profitability(metrics):
    roic = metrics.get('roic')
    op_margin = metrics.get('operating_margin')
    gross_margin = metrics.get('gross_margin')
    score = 0
    rationale = []
    if roic is not None:
        if roic > 0.2:
            score = 5
        elif roic > 0.15:
            score = 4
        elif roic > 0.1:
            score = 3
        elif roic > 0.05:
            score = 2
        else:
            score = 1
        rationale.append(f"ROIC={roic:.2%}")
    else:
        rationale.append("ROIC N/A")
    if op_margin is not None:
        rationale.append(f"OperatingMargin={op_margin:.2%}")
    if gross_margin is not None:
        rationale.append(f"GrossMargin={gross_margin:.2%}")
    return score, "; ".join(rationale)

def score_competitive_advantage(metrics):
    gross_margin = metrics.get('gross_margin')
    roa = metrics.get('return_on_assets')
    score = 0
    rationale = []
    if gross_margin is not None:
        if gross_margin > 0.5:
            score = 5
        elif gross_margin > 0.4:
            score = 4
        elif gross_margin > 0.3:
            score = 3
        else:
            score = 2
        rationale.append(f"GrossMargin={gross_margin:.2%}")
    else:
        rationale.append("GrossMargin N/A")
    if roa is not None:
        rationale.append(f"ROA={roa:.2%}")
    return score, "; ".join(rationale)

def score_business_model(metrics):
    gm = metrics.get('gross_margin')
    score = 0
    rationale = []
    if gm is not None:
        if gm > 0.5:
            score = 5
        elif gm > 0.4:
            score = 4
        elif gm > 0.3:
            score = 3
        else:
            score = 2
        rationale.append(f"GrossMargin={gm:.2%}")
    else:
        rationale.append("GrossMargin N/A")
    return score, "; ".join(rationale)

def score_risk_profile(metrics):
    beta = metrics.get('beta')
    score = 0
    rationale = []
    if beta is not None:
        if beta < 0.8:
            score = 5
        elif beta < 1.0:
            score = 4
        elif beta < 1.2:
            score = 3
        elif beta < 1.5:
            score = 2
        else:
            score = 1
        rationale.append(f"Beta={beta:.2f}")
    else:
        rationale.append("Beta N/A")
    return score, "; ".join(rationale)

def score_industry_positioning(metrics):
    gm = metrics.get('gross_margin')
    score = 0
    rationale = []
    if gm is not None:
        if gm > 0.6:
            score = 5
        elif gm > 0.5:
            score = 4
        elif gm > 0.4:
            score = 3
        elif gm > 0.3:
            score = 2
        else:
            score = 1
        rationale.append(f"GrossMargin={gm:.2%}")
    else:
        rationale.append("GrossMargin N/A")
    return score, "; ".join(rationale)

def score_market_metrics(metrics):
    fcf = metrics.get('free_cash_flow')
    market_cap = metrics.get('market_cap')
    score = 0
    rationale = []
    if fcf is not None and market_cap is not None and market_cap > 0:
        fcf_ratio = fcf / market_cap
        if fcf_ratio > 0.1:
            score = 5
        elif fcf_ratio > 0.08:
            score = 4
        elif fcf_ratio > 0.06:
            score = 3
        elif fcf_ratio > 0.04:
            score = 2
        else:
            score = 1
        rationale.append(f"FCF yield={fcf_ratio:.2%}")
    else:
        rationale.append("FCF yield N/A")
    return score, "; ".join(rationale)

# ---------------- Render / file write using templates ----------------
def load_template(path):
    with open(path, "r", encoding="utf-8") as fh:
        return Template(fh.read())

def write_ticker_html(ticker, outdir, factors, metrics, optional_metrics):
    tpl = load_template(TEMPLATE_TICKER_PATH)
    total_score = sum([f['score'] for f in factors])
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    rendered = tpl.render(
        ticker=ticker,
        date=date_str,
        factors=factors,
        total_score=total_score,
        max_total=35,
        optional_metrics={k: pretty(v) for k, v in optional_metrics.items()},
        metrics={k: pretty(v) for k, v in metrics.items()}
    )
    os.makedirs(outdir, exist_ok=True)
    file_date = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"{ticker.upper()}_{file_date}.html"
    outpath = os.path.join(outdir, filename)
    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    return outpath

def generate_summary_html(ticker_results, outdir, infile_name=None):
    tpl = load_template(TEMPLATE_SUMMARY_PATH)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    if infile_name:
        base = os.path.splitext(os.path.basename(infile_name))[0]
        filename = f"StockSummary_{base}_{date_str}.html"
    else:
        tks = [r['ticker'] for r in ticker_results][:4]
        if tks:
            filename = f"StockSummary_{'_'.join(tks)}_{date_str}.html"
        else:
            filename = f"StockSummary_{date_str}.html"

    rendered = tpl.render(
        date=datetime.datetime.now().strftime("%Y-%m-%d"),
        results=ticker_results,
        factor_names=[f["name"] for f in ticker_results[0]["factors"]] if ticker_results else [],
        max_total=35
    )
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, filename)
    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    return outpath

# ---------------- Main CLI flow ----------------
def main():
    ensure_templates()

    parser = argparse.ArgumentParser(description="Generate stock quality assessments (v4) using yfinance.")
    parser.add_argument('tickers', nargs='*', help='Ticker symbols')
    parser.add_argument('--infile', help='File with tickers (one per line)')
    parser.add_argument('--outdir', default=r"C:\Users\gumbs\OneDrive\Documents\StocksResearchData\StockResearchAssessments\Assessments\\", help='Output directory')
    parser.add_argument('--summary', action='store_true', help='Generate summary HTML for multiple tickers')
    args = parser.parse_args()

    tickers = []
    if args.infile:
        if not os.path.exists(args.infile):
            print("Error: infile does not exist", file=sys.stderr)
            sys.exit(1)
        with open(args.infile, "r", encoding="utf-8") as fh:
            tickers.extend([line.strip() for line in fh if line.strip()])
    if args.tickers:
        tickers.extend(args.tickers)

    tickers = list(dict.fromkeys([t.upper() for t in tickers]))
    if not tickers:
        print("No tickers provided", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    results = []

    for t in tickers:
        print(f"Assessing {t} ...")
        try:
            tk = yf.Ticker(t)
            metrics = extract_financials_yf_v3(tk)
        except Exception as e:
            print(f"Failed for {t}: {e}", file=sys.stderr)
            continue

        fs_score, fs_r = score_financial_strength(metrics)
        pf_score, pf_r = score_profitability(metrics)
        ca_score, ca_r = score_competitive_advantage(metrics)
        bm_score, bm_r = score_business_model(metrics)
        rp_score, rp_r = score_risk_profile(metrics)
        ip_score, ip_r = score_industry_positioning(metrics)
        mm_score, mm_r = score_market_metrics(metrics)

        factors = [
            {"name": "Financial Strength", "score": fs_score, "rationale": fs_r},
            {"name": "Profitability & Efficiency", "score": pf_score, "rationale": pf_r},
            {"name": "Competitive Advantage", "score": ca_score, "rationale": ca_r},
            {"name": "Business Model Quality", "score": bm_score, "rationale": bm_r},
            {"name": "Risk Profile", "score": rp_score, "rationale": rp_r},
            {"name": "Industry Positioning", "score": ip_score, "rationale": ip_r},
            {"name": "Market Metrics / Shareholder Return", "score": mm_score, "rationale": mm_r},
        ]

        optional_metrics = {
            "Interest Coverage (EBIT/Interest)": metrics.get('interest_coverage'),
            "Revenue CAGR 5Y": metrics.get('rev_cagr_5y'),
            "EPS CAGR 5Y": metrics.get('eps_cagr_5y'),
            "CapEx / Revenue": metrics.get('capex_rev'),
            "Dividend Payout Ratio (Option A)": metrics.get('dividend_payout_ratio')
        }

        display_metrics = {
            "Market Cap": pretty(metrics.get('market_cap')),
            "Trailing PE": metrics.get('trailing_pe'),
            "Beta": metrics.get('beta'),
            "Quote Price": metrics.get('quote_price'),
            "Free Cash Flow (TTM)": pretty(metrics.get('free_cash_flow')),
            "Total Debt": pretty(metrics.get('total_debt')),
            "Debt / EBITDA": metrics.get('debt_ebitda'),
            "Financial Leverage (Assets/Equity)": metrics.get('financial_leverage'),
            "Current Ratio": metrics.get('current_ratio'),
            "Interest Coverage (EBIT/Interest)": metrics.get('interest_coverage'),
            "Gross Margin": metrics.get('gross_margin'),
            "Operating Margin": metrics.get('operating_margin'),
            "Return on Assets": metrics.get('return_on_assets'),
            "ROIC (proxy)": metrics.get('roic'),
            "Revenue (TTM)": pretty(metrics.get('revenue_ttm')),
            "Revenue StdDev 5Y": metrics.get('rev_std_5y'),
            "CapEx / Revenue (proxy)": metrics.get('capex_rev'),
            "Dividend Payout Ratio (Option A)": metrics.get('dividend_payout_ratio')
        }

        try:
            outpath = write_ticker_html(t, args.outdir, factors, display_metrics, optional_metrics)
            total = sum([f['score'] for f in factors])
            print(f"Saved: {outpath} (Score: {total}/35)")
            results.append({'ticker': t, 'file': os.path.basename(outpath), 'total_score': total, 'factors': factors})
        except Exception as e:
            print(f"Failed to write HTML for {t}: {e}", file=sys.stderr)

    if args.summary and results:
        summary_path = generate_summary_html(results, args.outdir, infile_name=args.infile)
        print(f"Summary written to: {summary_path}")

if __name__ == "__main__":
    main()
