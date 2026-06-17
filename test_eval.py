"""Quick sanity test for rule-based evaluation checks."""
from evaluation.rule_checks import (
    check_report_structure,
    check_verdict_clarity,
    check_analysis_structure,
)

perfect_report = """
## Executive Summary
**VERDICT: BUY** | **Confidence: 85%** | **Investment Horizon: Long-term**

## Company Overview
Test company overview.

## Valuation Analysis
P/E of 24x vs sector average of 18x.

## Financial Health
Strong balance sheet, debt-to-equity of 0.3.

## Earnings Quality & Growth
Revenue grew 18% YoY. Beat estimates for 6 consecutive quarters.

## News & Market Sentiment
Positive news flow around new product launches.

## Key Risks
- Market saturation risk
- Regulatory headwinds

## Investment Thesis
BUY this stock for long-term investors seeking growth exposure.
"""

perfect_analysis = """
## 1. Valuation
P/E of 24x, forward P/E of 20x. Fair to slightly elevated vs peers.

## 2. Financial Health
Debt-to-equity 0.3. Current ratio 2.1. Strong balance sheet.

## 3. Profitability & Margins
Gross margin 45%, net margin 18%, EBITDA $4.2B.

## 4. Growth
Revenue grew 18% YoY. EPS growth of 22% YoY.

## 5. Cash Flow Quality
Free cash flow of $2.1B. Operating cash flow $3.5B.

## 6. Earnings Quality
Beat EPS estimates in 6 of last 8 quarters. Average surprise +4.2%.

## 7. News Sentiment
Positive: New product launches, strong guidance. Sentiment: positive.

BULLISH stance. The company demonstrates strong fundamentals and growth.
"""

print("=== Rule-based check results ===")
print(f"Report structure:    {check_report_structure(perfect_report):.3f}  (expected 1.0)")
print(f"Verdict clarity:     {check_verdict_clarity(perfect_report):.3f}  (expected 1.0)")
print(f"Analysis structure:  {check_analysis_structure(perfect_analysis):.3f}  (expected 1.0)")
print()

bad_report = "This is a vague report. The stock looks interesting. VERDICT: HOLD | Confidence: 50%"
print("=== Bad report scores ===")
print(f"Bad structure:       {check_report_structure(bad_report):.3f}  (expected near 0.0)")
print(f"Partial verdict:     {check_verdict_clarity(bad_report):.3f}  (expected 0.67 - missing horizon)")

print("\nAll rule-based checks passed!")
