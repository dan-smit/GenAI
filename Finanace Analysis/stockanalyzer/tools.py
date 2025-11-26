from datetime import datetime, timedelta
from enum import Enum

from edgar import Company
import yfinance as yf
from langchain_core.tools import tool

prices_response_template = """
<prices>
{prices}
</prices>
""".strip()

company_filing_response_template = """
<filing>
    <company>{company}<\company>
    <filing_date>{filing_date}</filing_date>
    <sections>{sections}<\sections>
</filing>
""".strip()

@tool
def get_historical_stock_price(ticker: str) -> str:
    """
    Fetches historical stock price data for a given company ticker for the last 90 days.
    
    Args:
        ticker: The stock ticker symbol (e.g., "AAPL", "NVDA")
    
    Returns:
        A string with lines of the date and weekly close price in the last 90 days
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        stock = yf.Ticker(ticker)
        
        daily_history = stock.history(
            start=start_date_str,
            end=end_date_str,
            interval="1d"
        )
        
        weekly_resampled = daily_history["Close"].resample("W-SUN").last()
        
        lines = []
        
        for date, close in zip(weekly_resampled.index, weekly_resampled): #zip[tuple]
            date = date.replace(tzinfo=None)
            if date > end_date:
                date = end_date
            date_text = date.strftime("%Y-%m-%d")
            lines.append(f"<{date_text}>{round(close, 2)}</{date_text}>")
        
        return prices_response_template.format(prices="\n".join(lines))
    
    except Exception as e:
        return f"An error occured while fetching historical price for '{ticker}': {e}"
    

class Section(str, Enum):
    MDA = "mda"
    RISK_FACTORS = "risk_factors"
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASHFLOW_STATEMENT = "cashflow_statement"
    
    
@tool
def fetch_sec_filing_sections(ticker: str, sections: list[Section]) -> str:
    """
    Fetches specific sections from a company's last SEC filing
    
    Args:
        ticker: Ticker symbol of the company
        sections: Sections to fetch from the SEC filing. Available sections are (mda, risk_factors,
            balance_sheet, income_statement, cashflow_statement)
            
    Returns:
        A string with the company's SEC filing sections in XML format
    """
    company = Company(ticker)
    filing = company.get_filings(form="10-Q").latest()
    filing_obj = filing.obj()
    xbrl = filing.xbrl()
    statements = xbrl.statements
    
    section_data = {}
    for section in sections:
        if section == Section.MDA:
            section_data[section] = filing_obj["Item 2"]
        elif section == Section.RISK_FACTORS:
            section_data[section] = filing_obj["Item 1A"]
        elif section == Section.BALANCE_SHEET:
            section_data[section] = statements.balance_sheet()
        elif section == Section.INCOME_STATEMENT:
            section_data[section] = statements.income_statements()
        elif section == Section.CASHFLOW_STATEMENT:
            section_data[section] = statements.cashflow_statement()
            
    return company_filing_response_template.format(
        company=company.name,
        filing_date=filing.filing_date.strftime("%Y-%m-%d"),
        sections="\n".join(
            [
                f"<{section.value}>\n{data}\n</{section.value}>"
                for section, data in section_data.items()
            ]
        )
    )
    
