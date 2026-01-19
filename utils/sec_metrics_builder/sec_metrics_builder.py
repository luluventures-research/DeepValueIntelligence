#!/usr/bin/env python3
"""
SEC Metrics Builder - Comprehensive Financial Data Extraction with Warren Buffett-Style Analysis
===============================================================================================

Fetches all 104 working SEC-available financial metrics + 23 derived Warren Buffett-style metrics
for companies from 2015-2025. Creates standardized CSV files with institutional-grade analysis.

Features:
- 102 comprehensive SEC XBRL metrics (30 empty metrics removed)
- 2 market price metrics via Financial Modeling Prep API (optional)
- 23 derived Warren Buffett-style metrics (EPS, P/E, PEG, ROE, FCF, etc.)
- 127 total metrics per company per year
- 10,069 companies from SEC database
- 11 years of historical data (2015-2025)
- Respectful rate limiting (default 0.5s between companies)
- Standardized CSV format with validation
- Balance sheet equation validation
- Comprehensive failure reporting (JSON + CSV formats)
- ETF filtering options (--skip-etf / --etf-only)

Usage:
    export FMP_API_KEY="your_fmp_api_key"
    python sec_metrics_builder.py --companies all --years 2015 2025 --skip-etf
    python sec_metrics_builder.py --ticker AAPL --years 2020 2025 --fmp-api-key YOUR_KEY
    python sec_metrics_builder.py --top 100 --years 2015 2025 --rate-limit 1.0
    python sec_metrics_builder.py --top 50 --years 2015 2025 --etf-only
"""

import argparse
import csv
import json
import logging
import os
import pandas as pd
import requests
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import zipfile
import yfinance as yf
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sec_metrics_builder.log')
    ]
)
logger = logging.getLogger(__name__)

# SEC API Configuration
SEC_BASE_URL = "https://data.sec.gov/api/xbrl"
SEC_COMPANIES_FILE = Path(__file__).parent / "sec_companies.json"

# Request session with proper headers for SEC compliance
session = requests.Session()
session.headers.update({
    'User-Agent': 'SEC Financial Metrics Builder (luluventures.ivy@gmail.com)',
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'data.sec.gov'
})

# Price data sources configuration
PRICE_CACHE_FILE = Path(__file__).parent / "stock_price_history_universal.json"  # Use universal cache with proper split adjustments
PRICE_CACHE = None  # Will be loaded on initialization

# Financial Modeling Prep API configuration (fallback)
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_API_KEY = None  # Will be set from environment variable

# Comprehensive SEC XBRL tag mappings (102 metrics total) - ORDERED (30 empty metrics removed)
SEC_METRICS = {
    # BALANCE SHEET - Assets (15 metrics)
    'Assets': ['Assets', 'AssetsTotal', 'AssetsTotalCurrentAndNoncurrent'],
    'AssetsCurrent': ['AssetsCurrent'],
    'AssetsNoncurrent': ['AssetsNoncurrent'],
    'CashAndCashEquivalents': ['CashAndCashEquivalentsAtCarryingValue', 'Cash', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', 'CashAndCashEquivalents'],
    'CashAndRestrictedCash': ['CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'],
    'MarketableSecurities': ['MarketableSecurities', 'AvailableForSaleSecurities'],
    'AccountsReceivableNet': ['AccountsReceivableNetCurrent', 'AccountsReceivableNet'],
    'Inventory': ['InventoryNet', 'Inventory'],
    'PropertyPlantAndEquipmentNet': ['PropertyPlantAndEquipmentNet'],
    'PropertyPlantAndEquipmentGross': ['PropertyPlantAndEquipmentGross'],
    'AccumulatedDepreciation': ['AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment'],
    'Goodwill': ['Goodwill'],
    'IntangibleAssetsNet': ['IntangibleAssetsNetExcludingGoodwill', 'IntangibleAssetsNet'],
    'DeferredTaxAssetsNet': ['DeferredTaxAssetsNet'],
    'OtherAssets': ['OtherAssets', 'OtherAssetsNoncurrent'],

    'Liabilities': ['Liabilities', 'LiabilitiesTotal', 'LiabilitiesAndStockholdersEquity'],
    'LiabilitiesCurrent': ['LiabilitiesCurrent'],
    'LiabilitiesNoncurrent': ['LiabilitiesNoncurrent'],
    'AccountsPayable': ['AccountsPayableCurrent', 'AccountsPayable'],
    'AccruedLiabilities': ['AccruedLiabilitiesCurrent', 'AccruedLiabilities'],
    'ShortTermBorrowings': ['ShortTermBorrowings', 'DebtCurrent'],
    'LongTermDebt': ['LongTermDebt', 'LongTermDebtNoncurrent'],
    'LongTermDebtCurrent': ['LongTermDebtCurrent'],
    'DebtCurrent': ['DebtCurrent'],
    'DebtNoncurrent': ['DebtNoncurrent', 'LongTermDebt'],
    'DeferredRevenue': ['DeferredRevenue', 'ContractWithCustomerLiabilityCurrent'],
    'DeferredRevenueNoncurrent': ['DeferredRevenueNoncurrent', 'ContractWithCustomerLiabilityNoncurrent'],
    'DeferredTaxLiabilities': ['DeferredTaxLiabilitiesNoncurrent', 'DeferredTaxLiabilities'],
    'EmployeeRelatedLiabilities': ['EmployeeRelatedLiabilitiesCurrent'],
    'OperatingLeaseLiability': ['OperatingLeaseLiability'],
    'FinanceLeaseLiability': ['FinanceLeaseLiability'],
    'OtherLiabilities': ['OtherLiabilities', 'OtherLiabilitiesNoncurrent'],

    # BALANCE SHEET - Equity (8 metrics)
    'StockholdersEquity': ['StockholdersEquity', 'StockholdersEquityTotal'],
    'StockholdersEquityIncludingNCI': ['StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'CommonStockValue': ['CommonStockValue'],
    'PreferredStockValue': ['PreferredStockValue'],
    'AdditionalPaidInCapital': ['AdditionalPaidInCapital'],
    'RetainedEarnings': ['RetainedEarningsAccumulatedDeficit', 'RetainedEarnings'],
    'TreasuryStock': ['TreasuryStockValue'],
    'NoncontrollingInterest': ['MinorityInterest', 'NoncontrollingInterest'],

    # INCOME STATEMENT - Revenue & Sales (4 metrics)
    'Revenues': ['Revenues', 'Revenue', 'TotalRevenues'],
    'RevenueFromContracts': ['RevenueFromContractWithCustomerExcludingAssessedTax'],
    'SalesRevenueNet': ['SalesRevenueNet'],
    'ProductSales': ['ProductSales', 'RevenueFromRelatedParties'],

    # INCOME STATEMENT - Cost & Expenses (13 metrics)
    'CostOfRevenue': ['CostOfRevenue', 'CostOfGoodsAndServicesSold'],
    'CostOfSales': ['CostOfSales', 'CostOfGoodsSold'],
    'ResearchAndDevelopmentExpense': ['ResearchAndDevelopmentExpense'],
    'SellingGeneralAndAdministrativeExpense': ['SellingGeneralAndAdministrativeExpense'],
    'GeneralAndAdministrativeExpense': ['GeneralAndAdministrativeExpense'],
    'SellingAndMarketingExpense': ['SellingAndMarketingExpense'],
    'DepreciationDepletionAndAmortization': ['DepreciationDepletionAndAmortization'],
    'AmortizationOfIntangibleAssets': ['AmortizationOfIntangibleAssets'],
    'RestructuringCosts': ['RestructuringCosts', 'RestructuringCharges'],
    'OperatingLeaseExpense': ['OperatingLeaseExpense'],
    'StockBasedCompensation': ['ShareBasedCompensation', 'StockBasedCompensation'],
    'OtherOperatingExpenses': ['OtherOperatingIncomeExpenseNet'],
    'TotalOperatingExpenses': ['OperatingExpenses'],

    # INCOME STATEMENT - Income & Profit (12 metrics)
    'GrossProfit': ['GrossProfit'],
    'OperatingIncomeLoss': ['OperatingIncomeLoss', 'IncomeLossFromOperations'],
    'IncomeTaxExpenseBenefit': ['IncomeTaxExpenseBenefit', 'IncomeTaxExpense'],
    'NetIncomeLoss': ['NetIncomeLoss', 'NetIncome'],
    'NetIncomeAvailableToCommonShareholders': ['NetIncomeLossAvailableToCommonStockholdersBasic'],
    'InterestExpense': ['InterestExpense', 'InterestExpenseDebt'],
    'OtherNonoperatingIncomeExpense': ['NonoperatingIncomeExpense', 'OtherNonoperatingIncomeExpense'],
    'GainLossOnSaleOfAssets': ['GainLossOnSaleOfPropertyPlantEquipment', 'GainLossOnSaleOfAssets'],
    'GainLossOnInvestments': ['GainLossOnInvestments'],
    'DiscontinuedOperations': ['IncomeLossFromDiscontinuedOperationsNetOfTax'],

    # CASH FLOW STATEMENT - Operating Activities (10 metrics)
    'NetCashFromOperatingActivities': ['NetCashProvidedByUsedInOperatingActivities'],
    'DepreciationAndAmortization': ['DepreciationDepletionAndAmortization', 'Depreciation'],
    'StockBasedCompensationExpense': ['ShareBasedCompensation', 'StockBasedCompensationExpense'],
    'DeferredIncomeTaxExpenseBenefit': ['DeferredIncomeTaxExpenseBenefit'],
    'ChangeInAccountsReceivable': ['IncreaseDecreaseInAccountsReceivable'],
    'ChangeInInventories': ['IncreaseDecreaseInInventories'],
    'ChangeInPrepaidExpenses': ['IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets'],
    'ChangeInAccountsPayable': ['IncreaseDecreaseInAccountsPayable'],
    'ChangeInAccruedLiabilities': ['IncreaseDecreaseInAccruedLiabilities'],
    'ChangeInDeferredRevenue': ['IncreaseDecreaseInDeferredRevenue'],

    # CASH FLOW STATEMENT - Investing Activities (7 metrics)
    'NetCashFromInvestingActivities': ['NetCashProvidedByUsedInInvestingActivities'],
    'CapitalExpenditures': ['PaymentsToAcquirePropertyPlantAndEquipment'],
    'BusinessAcquisitions': ['PaymentsToAcquireBusinessesNetOfCashAcquired'],
    'InvestmentPurchases': ['PaymentsToAcquireInvestments'],
    'AssetSales': ['ProceedsFromSaleOfPropertyPlantAndEquipment'],
    'IntangibleAssetPurchases': ['PaymentsToAcquireIntangibleAssets'],
    'BusinessDivestitures': ['ProceedsFromDivestitureOfBusinesses'],

    # CASH FLOW STATEMENT - Financing Activities (10 metrics)
    'NetCashFromFinancingActivities': ['NetCashProvidedByUsedInFinancingActivities'],
    'CommonStockIssuance': ['ProceedsFromIssuanceOfCommonStock'],
    'ShareRepurchases': ['PaymentsForRepurchaseOfCommonStock'],
    'CommonDividendsPaid': ['PaymentsOfDividendsCommonStock'],
    'PreferredDividendsPaid': ['PaymentsOfDividendsPreferredStockAndPreferenceStock'],
    'LongTermDebtIssuance': ['ProceedsFromIssuanceOfLongTermDebt'],
    'LongTermDebtRepayments': ['RepaymentsOfLongTermDebt'],
    'ShortTermDebtProceeds': ['ProceedsFromShortTermDebt'],
    'ShortTermDebtRepayments': ['RepaymentsOfShortTermDebt'],
    'DebtIssuanceCosts': ['PaymentsOfDebtIssuanceCosts'],

    # SHARE DATA - Share Counts (7 metrics)
    'CommonStockSharesOutstanding': ['CommonStockSharesOutstanding'],
    'CommonStockSharesIssued': ['CommonStockSharesIssued'],
    'WeightedAverageSharesBasic': ['WeightedAverageNumberOfSharesOutstandingBasic'],
    'WeightedAverageSharesDiluted': ['WeightedAverageNumberOfDilutedSharesOutstanding'],
    'DilutionAdjustment': ['WeightedAverageNumberDilutedSharesOutstandingAdjustment'],
    'PreferredStockSharesOutstanding': ['PreferredStockSharesOutstanding'],
    'TreasuryStockShares': ['TreasuryStockShares'],

    # SHARE DATA - Stock-Related (1 metric)
    'StockBasedCompensationExpenseTotal': ['ShareBasedCompensation'],

    # MARKET DATA - External APIs (2 metrics)
    'MarketPriceUSD': None,  # From Financial Modeling Prep - current split-adjusted price
    'MarketPriceNonSplitAdjustedUSD': None  # From Financial Modeling Prep - raw historical price
}

# Verify we have exactly 104 metrics (102 SEC + 2 market price metrics)
assert len(SEC_METRICS) == 104, f"Expected 104 metric mappings, got {len(SEC_METRICS)}"

CORE_SEC_METRICS = [
    'MarketPriceUSD',
    'Revenues',
    'GrossProfit',
    'OperatingIncomeLoss',
    'NetIncomeLoss',
    'NetCashFromOperatingActivities',
    'Assets',
    'CashAndCashEquivalents',
    'Liabilities',
    'StockholdersEquity',
]

# DERIVED WARREN BUFFETT-STYLE METRICS (23 additional metrics)
DERIVED_METRICS = [
    # Valuation Metrics (5)
    'EarningsPerShare',
    'PriceToEarning',
    'PEGRatio',
    'BookValuePerShare', 
    'PriceToBook',
    
    # Growth Metrics (3)
    'RevenueGrowthRate',
    'NetIncomeGrowthRate',
    'BookValueGrowthRate',
    
    # Profitability Margins (4)
    'GrossMargin',
    'OperatingMargin', 
    'NetIncomeMargin',
    'FreeCashFlowMargin',
    
    # Cash Flow Metrics (2)
    'FreeCashFlow',
    'OwnerEarnings',
    
    # Financial Health Ratios (3)
    'CurrentRatio',
    'DebtToEquityRatio',
    'InterestCoverageRatio',
    
    # Return Metrics (3)
    'ReturnOnEquity',
    'ReturnOnAssets',
    'ReturnOnInvestedCapital',
    
    # Capital Allocation Metrics (3)
    'RetainedEarningsToNetIncome',
    'DividendPayoutRatio', 
    'CapitalExpenditureToDepreciation'
]

# Verify we have exactly 23 derived metrics
assert len(DERIVED_METRICS) == 23, f"Expected 23 derived metrics, got {len(DERIVED_METRICS)}"

# ETF-SPECIFIC METRICS (18 metrics focused on fund efficiency and performance)
ETF_METRICS = [
    # Fund Efficiency Metrics (6)
    'ExpenseRatio',              # Total operating expenses / Average net assets
    'AssetTurnover',             # Net income / Average total assets
    'OperatingExpenseRatio',     # Operating expenses / Total assets
    'AssetCoverageRatio',        # Total assets / Total liabilities
    'LiquidityRatio',            # Current assets / Current liabilities
    'CashToAssetsRatio',         # Cash and equivalents / Total assets
    
    # Performance Metrics (6)  
    'TotalReturn',               # (Ending NAV - Beginning NAV + Distributions) / Beginning NAV
    'AssetGrowthRate',           # Year-over-year asset growth rate
    'NAVGrowthRate',             # Net asset value growth rate
    'IncomeYield',               # Net income / Average assets (for income-generating ETFs)
    'ExpenseAdjustedReturn',     # Total return - expense ratio
    'AssetUtilizationRatio',     # Revenue (if any) / Average assets
    
    # Financial Health Metrics (4)
    'DebtToAssetRatio',          # Total liabilities / Total assets
    'EquityToAssetRatio',        # Stockholders equity / Total assets
    'AssetStabilityRatio',       # Non-current assets / Total assets
    'ExpenseEfficiencyRatio',    # Expense ratio improvement year-over-year
    
    # Scale & Market Metrics (2)
    'AUMGrowthRate',            # Assets under management growth
    'SharePriceVolatility'      # Standard deviation of price returns (requires market data)
]

# Verify we have exactly 18 ETF-specific metrics
assert len(ETF_METRICS) == 18, f"Expected 18 ETF metrics, got {len(ETF_METRICS)}"

YFINANCE_QUARTERLY_METRIC_KEYS = {
    'Revenues': 'Total Revenue',
    'GrossProfit': 'Gross Profit',
    'OperatingIncomeLoss': 'Operating Income',
    'NetIncomeLoss': 'Net Income',
    'NetCashFromOperatingActivities': 'Total Cash From Operating Activities'
}

# ETF-relevant SEC metrics subset (only metrics that make sense for funds)
ETF_RELEVANT_SEC_METRICS = {
    # BALANCE SHEET - Assets (relevant for funds)
    'Assets': ['Assets', 'AssetsTotal', 'AssetsTotalCurrentAndNoncurrent'],
    'AssetsCurrent': ['AssetsCurrent'],
    'AssetsNoncurrent': ['AssetsNoncurrent'],
    'CashAndCashEquivalents': ['CashAndCashEquivalentsAtCarryingValue', 'Cash', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', 'CashAndCashEquivalents'],
    'CashAndRestrictedCash': ['CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'],
    'MarketableSecurities': ['MarketableSecurities', 'AvailableForSaleSecurities'],
    'OtherAssets': ['OtherAssets', 'OtherAssetsNoncurrent'],
    
    # BALANCE SHEET - Liabilities (relevant for funds)
    'Liabilities': ['Liabilities', 'LiabilitiesTotal'],
    'LiabilitiesCurrent': ['LiabilitiesCurrent'],
    'LiabilitiesNoncurrent': ['LiabilitiesNoncurrent'],
    'AccruedLiabilities': ['AccruedLiabilitiesCurrent', 'AccruedLiabilities'],
    'ShortTermBorrowings': ['ShortTermBorrowings', 'DebtCurrent'],
    'LongTermDebt': ['LongTermDebt', 'LongTermDebtNoncurrent'],
    'OtherLiabilities': ['OtherLiabilities', 'OtherLiabilitiesNoncurrent'],
    
    # BALANCE SHEET - Equity (relevant for funds)
    'StockholdersEquity': ['StockholdersEquity', 'StockholdersEquityTotal'],
    'StockholdersEquityIncludingNCI': ['StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'RetainedEarnings': ['RetainedEarningsAccumulatedDeficit', 'RetainedEarnings'],
    
    # INCOME STATEMENT - Fund Income (relevant)
    'NetIncomeLoss': ['NetIncomeLoss', 'ProfitLoss'],
    'InterestAndDividendIncome': ['InvestmentIncomeInterest', 'InterestAndDividendIncomeOperating', 'DividendIncomeOperating', 'InterestIncomeOperating'],
    'IncomeTaxExpenseBenefit': ['IncomeTaxExpenseBenefit'],
    'OtherNonoperatingIncomeExpense': ['NonoperatingIncomeExpense'],
    'GainLossOnInvestments': ['GainLossOnInvestments', 'UnrealizedGainLossOnInvestments'],
    
    # Fund-specific expenses
    'TotalOperatingExpenses': ['CostsAndExpenses', 'OperatingExpenses'],
    
    # SHARE DATA (relevant for ETF units/shares)
    'CommonStockSharesOutstanding': ['CommonStockSharesOutstanding', 'SharesOutstanding'],
    'WeightedAverageSharesBasic': ['WeightedAverageNumberOfSharesOutstandingBasic'],
    'WeightedAverageSharesDiluted': ['WeightedAverageNumberOfDilutedSharesOutstanding'],
}

# ETF-relevant SEC metrics: 25 (vs 104 full company metrics)

class SECMetricsBuilder:
    """Main class for building comprehensive SEC financial metrics"""
    
    def __init__(self, rate_limit_delay: float = 0.5, fmp_api_key: Optional[str] = None):
        self.companies = {}
        self.load_companies_database()
        self.api_call_count = 0
        self.failed_companies = []
        self.failure_details = {}  # Track detailed failure reasons
        self.rate_limit_delay = rate_limit_delay  # Default 0.5 seconds between companies
        self.request_delay = 0.1  # 100ms between individual API requests
        
        # Load price cache
        self.load_price_cache()
        
        # Initialize FMP API keys (multiple for redundancy)
        self.fmp_api_keys = []
        for key_name in ['FMP_API_KEY', 'FMP_API_KEY2', 'FMP_API_KEY3']:
            key = fmp_api_key if key_name == 'FMP_API_KEY' and fmp_api_key else os.getenv(key_name)
            if key:
                self.fmp_api_keys.append(key)
        
        global FMP_API_KEY
        FMP_API_KEY = self.fmp_api_keys[0] if self.fmp_api_keys else None
        
        if not self.fmp_api_keys:
            logger.warning("⚠️  No FMP API keys found. Using cached prices only.")
        else:
            logger.info(f"📊 Loaded {len(self.fmp_api_keys)} FMP API keys for redundancy")
        
    def load_price_cache(self):
        """Load cached price database from JSON file"""
        global PRICE_CACHE
        try:
            if os.path.exists(PRICE_CACHE_FILE):
                with open(PRICE_CACHE_FILE, 'r') as f:
                    PRICE_CACHE = json.load(f)
                    logger.info(f"📦 Loaded price cache with {len(PRICE_CACHE)} companies")
            else:
                logger.warning(f"⚠️  Price cache not found: {PRICE_CACHE_FILE}")
                PRICE_CACHE = {}
        except Exception as e:
            logger.error(f"Failed to load price cache: {e}")
            PRICE_CACHE = {}
    
    def get_cached_price(self, ticker: str, year: int, price_type: str) -> Optional[float]:
        """Get cached price for ticker and year"""
        if not PRICE_CACHE:
            return None
        
        ticker_data = PRICE_CACHE.get(ticker)
        if not ticker_data:
            return None
            
        year_data = ticker_data.get('prices', {}).get(str(year))
        if not year_data:
            return None
            
        return year_data.get(price_type)

    def load_companies_database(self):
        """Load companies from sec_companies.json"""
        try:
            with open(SEC_COMPANIES_FILE, 'r') as f:
                data = json.load(f)
                self.companies = data['companies']
                logger.info(f"✅ Loaded {len(self.companies)} companies from {SEC_COMPANIES_FILE}")
        except Exception as e:
            logger.error(f"❌ Failed to load {SEC_COMPANIES_FILE}: {e}")
            raise
        
        # Load official SEC mutual fund/ETF database for enhanced detection
        self.sec_etf_symbols = set()
        self.load_sec_etf_database()
    
    def load_sec_etf_database(self):
        """Load official SEC mutual fund/ETF database for enhanced ETF detection"""
        try:
            # Try to fetch from SEC if not cached
            sec_mf_file = "sec_mf_database.json"
            if not os.path.exists(sec_mf_file):
                logger.info("📡 Fetching official SEC mutual fund/ETF database...")
                response = session.get("https://www.sec.gov/files/company_tickers_mf.json")
                response.raise_for_status()
                
                with open(sec_mf_file, 'w') as f:
                    json.dump(response.json(), f)
                logger.info(f"💾 Saved SEC MF/ETF database to {sec_mf_file}")
            
            # Load the database
            with open(sec_mf_file, 'r') as f:
                data = json.load(f)
                
            # Extract all symbols from the database
            for entry in data['data']:
                # entry format: [cik, seriesId, classId, symbol]
                symbol = entry[3]
                if symbol:  # Skip empty symbols
                    self.sec_etf_symbols.add(symbol.upper())
            
            logger.info(f"📊 Loaded {len(self.sec_etf_symbols)} symbols from official SEC MF/ETF database")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load SEC MF/ETF database: {e}")
            logger.info("🔄 Falling back to pattern-based ETF detection only")
    
    def is_etf(self, ticker: str) -> bool:
        """Check if a ticker represents an ETF/Fund using official SEC database + pattern matching"""
        if ticker not in self.companies:
            return False
        
        # Primary detection: Check official SEC mutual fund/ETF database
        if hasattr(self, 'sec_etf_symbols') and ticker.upper() in self.sec_etf_symbols:
            return True
            
        # Secondary detection: Pattern matching for cases not in SEC database
        company_title = self.companies[ticker].get('title', '').upper()
        
        # Enhanced ETF detection patterns - use word boundaries to avoid false positives  
        import re
        
        # Check for ETF as whole word (not part of other words like "NETFLIX")
        if re.search(r'\bETF\b', company_title) or 'EXCHANGE TRADED FUND' in company_title or 'EXCHANGE-TRADED FUND' in company_title:
            return True
        
        trust_indicators = [
            'SPDR', 'ISHARES', 'VANGUARD', 'INVESCO', 'PROSHARES',
            'FIRST TRUST', 'ABRDN', 'GRAYSCALE', 'DIREXION',
            'QQQ', 'SPY', 'VTI', 'IWM', 'GLD', 'SLV'
        ]
        
        fund_indicators = [
            'FUND' if not any(exclude in company_title for exclude in ['FUNDAMENTAL', 'FUNDING']) else None,
            'INDEX',
            'MUNICIPAL',
            'SECTOR'
        ]
        fund_indicators = [x for x in fund_indicators if x]  # Remove None values
            
        # Check for trust + specific fund companies
        if 'TRUST' in company_title and any(indicator in company_title for indicator in trust_indicators):
            return True
            
        # Check for fund indicators (with exclusions)
        if any(indicator in company_title for indicator in fund_indicators):
            return True
            
        return False
    
    def filter_companies_by_etf_status(self, tickers: List[str], skip_etf: bool = False, etf_only: bool = False) -> List[str]:
        """Filter companies based on ETF status"""
        if not skip_etf and not etf_only:
            return tickers  # No filtering
        
        filtered_tickers = []
        etf_count = 0
        non_etf_count = 0
        
        for ticker in tickers:
            if ticker in self.companies:
                is_etf_result = self.is_etf(ticker)
                
                if is_etf_result:
                    etf_count += 1
                    if etf_only:
                        filtered_tickers.append(ticker)
                else:
                    non_etf_count += 1
                    if skip_etf:
                        filtered_tickers.append(ticker)
            else:
                # If ticker not found in database, include it (assume non-ETF)
                if skip_etf or not etf_only:
                    filtered_tickers.append(ticker)
                    non_etf_count += 1
        
        # Log filtering results
        if skip_etf:
            logger.info(f"🚫 ETF filtering: Skipped {etf_count} ETFs, processing {len(filtered_tickers)} non-ETF companies")
        elif etf_only:
            logger.info(f"📈 ETF filtering: Found {etf_count} ETFs, skipping {non_etf_count} non-ETF companies")
        
        return filtered_tickers
    
    def get_company_facts(self, cik: str) -> Optional[Dict]:
        """Get company facts from SEC API with respectful rate limiting"""
        try:
            url = f"{SEC_BASE_URL}/companyfacts/CIK{cik}.json"
            
            # Rate limiting before each request
            time.sleep(self.request_delay)
            
            response = session.get(url, timeout=30)
            
            if response.status_code == 429:  # Rate limited
                logger.warning(f"⚠️  Rate limited for CIK {cik}, waiting 3 seconds...")
                time.sleep(3)
                response = session.get(url, timeout=30)
            
            response.raise_for_status()
            self.api_call_count += 1
            
            return response.json(), None
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                logger.warning(f"⚠️  No SEC data found for CIK {cik}")
                return None, "404_NOT_FOUND"
            else:
                logger.error(f"❌ HTTP error for CIK {cik}: {e}")
                return None, f"HTTP_ERROR_{response.status_code}"
        except Exception as e:
            logger.error(f"❌ API error for CIK {cik}: {e}")
            return None, f"API_ERROR_{str(e)[:50]}"
    
    def get_etf_yfinance_data(self, ticker: str, years: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
        """Get ETF data using Yahoo Finance across multiple years"""
        logger.info(f"📊 Fetching Yahoo Finance data for ETF {ticker}")
        
        etf_results = {}
        
        try:
            # Create yfinance ticker object
            etf = yf.Ticker(ticker)
            
            # Get basic info
            info = etf.info
            if not info:
                logger.error(f"❌ No Yahoo Finance data available for {ticker}")
                return {year: {} for year in years}
            
            # Get historical data for volatility calculations
            hist = etf.history(period="3y")  # Get 3 years for better calculations
            
            # Get dividend data
            dividends = etf.dividends
            
            # Extract base metrics from Yahoo Finance
            base_metrics = self.extract_yfinance_etf_metrics(ticker, info, hist, dividends)
            
            # Apply base metrics to all requested years
            # Note: Yahoo Finance gives current data, so we apply it to all years
            # This is a limitation but better than no data
            for year in years:
                year_metrics = base_metrics.copy()
                
                # Adjust historical metrics if we have sufficient data
                if not hist.empty and len(hist) > 252:  # At least 1 year of data
                    year_metrics = self.adjust_etf_metrics_for_year(year_metrics, hist, year)
                
                etf_results[year] = year_metrics
                
            logger.info(f"✅ Extracted {len(base_metrics)} Yahoo Finance metrics for {ticker}")
            return etf_results
            
        except Exception as e:
            logger.error(f"❌ Error fetching Yahoo Finance data for {ticker}: {e}")
            return {year: {} for year in years}
    
    def extract_yfinance_etf_metrics(self, ticker: str, info: dict, hist: pd.DataFrame, dividends: pd.Series) -> Dict[str, Optional[float]]:
        """Extract ETF metrics from Yahoo Finance data"""
        metrics = {}
        
        try:
            # Map Yahoo Finance fields to our metrics
            if 'totalAssets' in info and info['totalAssets']:
                metrics['Assets'] = float(info['totalAssets'])
            
            if 'navPrice' in info and info['navPrice']:
                metrics['NAVGrowthRate'] = float(info['navPrice'])
            
            # Income yield from multiple sources
            if 'yield' in info and info['yield']:
                metrics['IncomeYield'] = float(info['yield'])
            elif 'dividendYield' in info and info['dividendYield']:
                metrics['IncomeYield'] = float(info['dividendYield'])
            
            # Expense ratio (may not be available in basic info)
            if 'expenseRatio' in info and info['expenseRatio']:
                metrics['ExpenseRatio'] = float(info['expenseRatio'])
            
            # Calculate metrics from historical data
            if not hist.empty:
                latest_close = hist['Close'].iloc[-1]
                
                # Calculate returns and volatility
                returns = hist['Close'].pct_change().dropna()
                
                if len(returns) > 0:
                    # Volatility (annualized)
                    volatility = returns.std() * np.sqrt(252)
                    metrics['SharePriceVolatility'] = volatility
                    
                    # YTD Return (approximate using available data)
                    if len(hist) >= 252:  # At least 1 year of data
                        year_ago_price = hist['Close'].iloc[-252]
                        ytd_return = (latest_close - year_ago_price) / year_ago_price
                        metrics['TotalReturn'] = ytd_return
                    
                    # Asset growth rate (approximate as price growth for ETFs)
                    if 'TotalReturn' in metrics:
                        metrics['AssetGrowthRate'] = metrics['TotalReturn']
                        metrics['AUMGrowthRate'] = metrics['TotalReturn']
            
            # Calculate dividend-based metrics
            if not dividends.empty and not hist.empty:
                # Calculate trailing 12-month dividend yield
                recent_date = dividends.index[-1]
                one_year_ago = recent_date - pd.DateOffset(months=12)
                recent_dividends = dividends[dividends.index >= one_year_ago]
                
                if not recent_dividends.empty and latest_close:
                    annual_dividend = recent_dividends.sum()
                    calculated_yield = annual_dividend / latest_close
                    # Override income yield if calculated yield is more recent
                    if calculated_yield > 0:
                        metrics['IncomeYield'] = calculated_yield
            
            # Calculate additional ETF-specific derived metrics
            if 'Assets' in metrics and 'ExpenseRatio' in metrics:
                # Expense-adjusted return
                if 'TotalReturn' in metrics:
                    metrics['ExpenseAdjustedReturn'] = metrics['TotalReturn'] - metrics['ExpenseRatio']
                
                # ETF-specific ratios (simplified assumptions)
                metrics['AssetUtilizationRatio'] = 1.0  # ETFs typically fully invested
                metrics['EquityToAssetRatio'] = 1.0      # ETFs are typically 100% equity
                metrics['DebtToAssetRatio'] = 0.0        # ETFs typically have no debt
                metrics['LiquidityRatio'] = 1.0          # High liquidity for major ETFs
                metrics['CashToAssetsRatio'] = 0.02      # Assume 2% cash
                
                # Operating expense ratio (simplified)
                metrics['OperatingExpenseRatio'] = metrics['ExpenseRatio'] * 0.8
                
                # Asset coverage ratio (simplified for ETFs)
                metrics['AssetCoverageRatio'] = 100.0  # Very high for ETFs
            
            logger.debug(f"📊 Extracted {len(metrics)} Yahoo Finance metrics for {ticker}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error extracting Yahoo Finance metrics for {ticker}: {e}")
            return {}
    
    def adjust_etf_metrics_for_year(self, metrics: Dict[str, Optional[float]], hist: pd.DataFrame, target_year: int) -> Dict[str, Optional[float]]:
        """Adjust ETF metrics for a specific year using historical data"""
        adjusted_metrics = metrics.copy()
        
        try:
            # Filter historical data to the target year
            year_start = pd.Timestamp(f"{target_year}-01-01")
            year_end = pd.Timestamp(f"{target_year}-12-31")
            
            year_data = hist[(hist.index >= year_start) & (hist.index <= year_end)]
            
            if not year_data.empty:
                # Calculate year-specific returns
                year_returns = year_data['Close'].pct_change().dropna()
                
                if len(year_returns) > 0:
                    # Year-specific volatility
                    year_volatility = year_returns.std() * np.sqrt(252)
                    adjusted_metrics['SharePriceVolatility'] = year_volatility
                    
                    # Year-specific total return
                    if len(year_data) > 1:
                        year_return = (year_data['Close'].iloc[-1] - year_data['Close'].iloc[0]) / year_data['Close'].iloc[0]
                        adjusted_metrics['TotalReturn'] = year_return
                        
                        # Update related metrics
                        if 'ExpenseRatio' in adjusted_metrics and adjusted_metrics['ExpenseRatio']:
                            adjusted_metrics['ExpenseAdjustedReturn'] = year_return - adjusted_metrics['ExpenseRatio']
                        
                        adjusted_metrics['AssetGrowthRate'] = year_return
                        adjusted_metrics['AUMGrowthRate'] = year_return
            
        except Exception as e:
            logger.debug(f"❌ Could not adjust metrics for year {target_year}: {e}")
        
        return adjusted_metrics
    
    def add_etf_derived_metrics(self, n_cen_data: Dict[int, Dict[str, Optional[float]]], ticker: str) -> Dict[int, Dict[str, Optional[float]]]:
        """Add derived ETF metrics to N-CEN data (similar to calculate_etf_metrics but for N-CEN)"""
        enhanced_data = {}
        
        years = sorted(n_cen_data.keys())
        
        for i, year in enumerate(years):
            year_data = n_cen_data[year].copy()
            
            # Get prior year data for growth calculations
            prior_year_data = n_cen_data[years[i-1]] if i > 0 else None
            
            # Calculate growth metrics
            if prior_year_data:
                # AUM Growth Rate
                current_assets = year_data.get('Assets')
                prior_assets = prior_year_data.get('Assets')
                if current_assets and prior_assets and prior_assets != 0:
                    year_data['AUMGrowthRate'] = (current_assets - prior_assets) / prior_assets
                
                # Asset Growth Rate (same as AUM for ETFs)
                year_data['AssetGrowthRate'] = year_data.get('AUMGrowthRate')
                
                # Expense Efficiency Ratio
                current_expense = year_data.get('ExpenseRatio')
                prior_expense = prior_year_data.get('ExpenseRatio')
                if current_expense and prior_expense and current_expense != 0:
                    year_data['ExpenseEfficiencyRatio'] = prior_expense / current_expense
            
            # Calculate ratios and efficiency metrics
            assets = year_data.get('Assets')
            
            # Operating Expense Ratio (approximate from expense ratio)
            expense_ratio = year_data.get('ExpenseRatio')
            if expense_ratio:
                # Assume 80% of expense ratio is operating expenses
                year_data['OperatingExpenseRatio'] = expense_ratio * 0.8
            
            # Asset Coverage Ratio (simplified - assume minimal liabilities for ETFs)
            if assets:
                # Most ETFs have minimal liabilities, so assume 99% asset coverage
                year_data['AssetCoverageRatio'] = 100.0  # Very high for ETFs
            
            # Income Yield (if we have total return, approximate income component)
            total_return = year_data.get('TotalReturn')
            if total_return:
                # Approximate income yield as 20% of total return for broad market ETFs
                year_data['IncomeYield'] = total_return * 0.2 if total_return > 0 else 0
            
            # Asset Utilization Ratio
            investment_income = year_data.get('NetIncomeLoss')
            if assets and investment_income and assets != 0:
                year_data['AssetUtilizationRatio'] = investment_income / assets
            
            # NAV Growth Rate (approximate from total return - income yield)
            if total_return and year_data.get('IncomeYield'):
                year_data['NAVGrowthRate'] = total_return - year_data['IncomeYield']
            elif total_return:
                # If no income yield calculated, assume 80% of total return is NAV growth
                year_data['NAVGrowthRate'] = total_return * 0.8
            
            # Add market price metrics if available
            market_price = self.get_market_price_year_end(ticker, year)
            if market_price:
                year_data['MarketPriceUSD'] = market_price
                
            market_price_non_adj = self.get_market_price_non_split_adjusted(ticker, year)
            if market_price_non_adj:
                year_data['MarketPriceNonSplitAdjustedUSD'] = market_price_non_adj
            
            # Set default values for metrics we can't calculate from N-CEN
            unavailable_metrics = [
                'LiquidityRatio', 'CashToAssetsRatio', 'DebtToAssetRatio', 
                'EquityToAssetRatio', 'AssetStabilityRatio', 'SharePriceVolatility'
            ]
            for metric in unavailable_metrics:
                if metric not in year_data:
                    year_data[metric] = None
            
            enhanced_data[year] = year_data
        
        return enhanced_data
    
    def get_latest_market_price(self, ticker: str) -> Optional[float]:
        """Get the latest market price using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
            return None
        except Exception as e:
            logger.debug(f"yfinance failed to get latest price for {ticker}: {e}")
            return None

    def get_market_price_year_end(self, ticker: str, year: int) -> Optional[float]:
        """Get year-end market price (split-adjusted) - cached first, then FMP API"""
        current_year = datetime.now().year
        if year == current_year:
            return self.get_latest_market_price(ticker)

        # Try cache first
        cached_price = self.get_cached_price(ticker, year, 'MarketPriceUSD')
        if cached_price is not None:
            logger.debug(f"Using cached split-adjusted price for {ticker} {year}: ${cached_price:.2f}")
            return cached_price
        
        # Fall back to FMP API if available
        if not self.fmp_api_keys:
            return None
        
        # Try each API key until one works
        for api_key in self.fmp_api_keys:
            try:
                url = f"{FMP_BASE_URL}/historical-price-full/{ticker}"
                params = {
                    'apikey': api_key,
                    'from': f"{year}-12-01",
                    'to': f"{year}-12-31"
                }
                
                time.sleep(0.1)  # Rate limiting
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                if 'historical' in data and data['historical']:
                    # Get the last trading day of the year (closest to Dec 31)
                    last_price_entry = data['historical'][0]  # Data is sorted desc by date
                    price = float(last_price_entry['close'])
                    logger.debug(f"Fetched split-adjusted price for {ticker} {year}: ${price:.2f}")
                    return price
                
            except Exception as e:
                logger.debug(f"FMP API key failed for {ticker} {year}: {e}")
                continue
        
        logger.debug(f"All price sources failed for {ticker} {year}")
        return None
    
    def get_market_price_non_split_adjusted(self, ticker: str, year: int) -> Optional[float]:
        """Get year-end market price (non-split-adjusted/raw) - cached first, then FMP API"""
        current_year = datetime.now().year
        if year == current_year:
            return self.get_latest_market_price(ticker)

        # Try cache first
        cached_price = self.get_cached_price(ticker, year, 'MarketPriceNonSplitAdjustedUSD')
        if cached_price is not None:
            logger.debug(f"Using cached non-split-adjusted price for {ticker} {year}: ${cached_price:.2f}")
            return cached_price
        
        # Fall back to FMP API if available
        if not self.fmp_api_keys:
            return None
        
        # Try each API key until one works
        for api_key in self.fmp_api_keys:
            try:
                url = f"{FMP_BASE_URL}/historical-price-eod/non-split-adjusted"
                params = {
                    'apikey': api_key,
                    'symbol': ticker,
                    'from': f"{year}-12-01",
                    'to': f"{year}-12-31"
                }
                
                time.sleep(0.1)  # Rate limiting
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                if 'historical' in data and data['historical']:
                    # Get the last trading day of the year (closest to Dec 31)
                    last_price_entry = data['historical'][0]  # Data is sorted desc by date
                    price = float(last_price_entry['close'])
                    logger.debug(f"Fetched non-split-adjusted price for {ticker} {year}: ${price:.2f}")
                    return price
                
            except Exception as e:
                logger.debug(f"FMP API key failed for {ticker} {year}: {e}")
                continue
        
        logger.debug(f"All price sources failed for {ticker} {year}")
        return None
    
    def get_latest_ttm_eps(self, ticker: str) -> Optional[float]:
        """Get the latest TTM EPS using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if 'trailingEps' in info and info['trailingEps'] is not None:
                return float(info['trailingEps'])
            return None
        except Exception as e:
            logger.debug(f"yfinance failed to get TTM EPS for {ticker}: {e}")
            return None

    def calculate_eps(self, net_income: Optional[float], shares_basic: Optional[float], 
                     shares_diluted: Optional[float], shares_outstanding: Optional[float]) -> Optional[float]:
        """Calculate Earnings Per Share (EPS)"""
        if net_income is None:
            return None
            
        # Primary: Weighted average basic shares (GAAP standard)
        if shares_basic and shares_basic > 0:
            return net_income / shares_basic
        # Fallback hierarchy: Diluted → Outstanding shares
        elif shares_diluted and shares_diluted > 0:
            return net_income / shares_diluted
        elif shares_outstanding and shares_outstanding > 0:
            return net_income / shares_outstanding
        return None
    
    def calculate_pe_ratio(self, market_price_non_split: Optional[float], eps: Optional[float]) -> Optional[float]:
        """Calculate Price-to-Earnings (P/E) ratio"""
        if market_price_non_split is None or eps is None:
            return None
        if eps > 0:  # Positive earnings only
            pe = market_price_non_split / eps
            return pe if pe <= 10000 else 10000  # Cap extreme values
        return None  # Negative earnings = undefined P/E
    
    def calculate_peg_ratio(self, pe_ratio: Optional[float], earnings_growth_rate: Optional[float]) -> Optional[float]:
        """Calculate PEG (Price/Earnings to Growth) ratio - Peter Lynch's metric"""
        if pe_ratio is None or earnings_growth_rate is None:
            return None
        if earnings_growth_rate > 0:  # Positive growth only
            peg = pe_ratio / earnings_growth_rate  # Growth rate already in percentage form
            return peg if peg <= 100 else 100  # Cap extreme values
        return None  # Negative or zero growth = undefined PEG
    
    def calculate_book_value_per_share(self, stockholders_equity: Optional[float], 
                                     shares_outstanding: Optional[float]) -> Optional[float]:
        """Calculate Book Value Per Share"""
        if stockholders_equity is None or shares_outstanding is None:
            return None
        if shares_outstanding > 0:
            return stockholders_equity / shares_outstanding
        return None
    
    def calculate_pb_ratio(self, market_price_non_split: Optional[float], 
                          book_value_per_share: Optional[float]) -> Optional[float]:
        """Calculate Price-to-Book (P/B) ratio"""
        if market_price_non_split is None or book_value_per_share is None:
            return None
        if book_value_per_share > 0:  # Positive book value only
            pb = market_price_non_split / book_value_per_share
            return pb if pb <= 1000 else 1000  # Cap extreme values
        return None  # Negative book value = undefined P/B

    def get_best_revenue_metric(self, year_data: Dict, year: int) -> tuple[Optional[float], Optional[str]]:
        """Get best revenue metric based on accounting standards"""
        revenues = year_data.get('Revenues')
        revenue_contracts = year_data.get('RevenueFromContracts')
        sales_net = year_data.get('SalesRevenueNet')
        
        # Post-2018: Prefer ASC 606 compliant metrics
        if year >= 2018:
            if revenue_contracts is not None:
                return revenue_contracts, 'RevenueFromContracts'
            elif revenues is not None:
                return revenues, 'Revenues'
            elif sales_net is not None:
                return sales_net, 'SalesRevenueNet'
        # Pre-2018: Prefer legacy metrics
        else:
            if revenues is not None:
                return revenues, 'Revenues'
            elif revenue_contracts is not None:
                return revenue_contracts, 'RevenueFromContracts'
            elif sales_net is not None:
                return sales_net, 'SalesRevenueNet'
        return None, None
    
    def calculate_revenue_growth_rate(self, current_year_data: Dict, prior_year_data: Dict, 
                                    current_year: int, prior_year: int) -> Optional[float]:
        """Calculate Revenue Growth Rate"""
        current_revenue, current_source = self.get_best_revenue_metric(current_year_data, current_year)
        prior_revenue, prior_source = self.get_best_revenue_metric(prior_year_data, prior_year)
        
        if current_revenue is not None and prior_revenue is not None and prior_revenue != 0:
            return (current_revenue - prior_revenue) / abs(prior_revenue) * 100
        return None
    
    def calculate_net_income_growth_rate(self, current_net_income: Optional[float], 
                                       prior_net_income: Optional[float]) -> Optional[float]:
        """Calculate Net Income Growth Rate"""
        if current_net_income is not None and prior_net_income is not None and prior_net_income != 0:
            return (current_net_income - prior_net_income) / abs(prior_net_income) * 100
        return None
    
    def calculate_book_value_growth_rate(self, current_stockholders_equity: Optional[float], 
                                       prior_stockholders_equity: Optional[float]) -> Optional[float]:
        """Calculate Book Value Growth Rate"""
        if (current_stockholders_equity is not None and prior_stockholders_equity is not None 
            and prior_stockholders_equity != 0):
            return (current_stockholders_equity - prior_stockholders_equity) / abs(prior_stockholders_equity) * 100
        return None

    def calculate_gross_margin(self, year_data: Dict, year: int) -> Optional[float]:
        """Calculate Gross Margin"""
        gross_profit = year_data.get('GrossProfit')
        revenue, revenue_source = self.get_best_revenue_metric(year_data, year)
        
        if gross_profit is not None and revenue is not None and revenue > 0:
            return (gross_profit / revenue) * 100
        return None
    
    def calculate_operating_margin(self, year_data: Dict, year: int) -> Optional[float]:
        """Calculate Operating Margin"""
        operating_income = year_data.get('OperatingIncomeLoss')
        revenue, revenue_source = self.get_best_revenue_metric(year_data, year)
        
        if operating_income is not None and revenue is not None and revenue > 0:
            return (operating_income / revenue) * 100
        return None
    
    def calculate_net_income_margin(self, year_data: Dict, year: int) -> Optional[float]:
        """Calculate Net Income Margin"""
        net_income = year_data.get('NetIncomeLoss')
        revenue, revenue_source = self.get_best_revenue_metric(year_data, year)
        
        if net_income is not None and revenue is not None and revenue > 0:
            return (net_income / revenue) * 100
        return None
    
    def calculate_free_cash_flow_margin(self, year_data: Dict, year: int) -> Optional[float]:
        """Calculate Free Cash Flow Margin"""
        operating_cash_flow = year_data.get('NetCashFromOperatingActivities')
        capex = year_data.get('CapitalExpenditures')
        
        if operating_cash_flow is not None:
            free_cash_flow = operating_cash_flow - (capex or 0)
            revenue, revenue_source = self.get_best_revenue_metric(year_data, year)
            
            if revenue is not None and revenue > 0:
                return (free_cash_flow / revenue) * 100
        return None

    def calculate_free_cash_flow(self, operating_cash_flow: Optional[float], 
                               capex: Optional[float]) -> Optional[float]:
        """Calculate Free Cash Flow"""
        if operating_cash_flow is not None:
            capex_amount = capex if capex is not None else 0
            return operating_cash_flow - capex_amount
        return None
    
    def calculate_owner_earnings(self, year_data: Dict) -> Optional[float]:
        """Calculate Owner Earnings (Buffett's preferred metric)"""
        net_income = year_data.get('NetIncomeLoss')
        depreciation = year_data.get('DepreciationAndAmortization')
        capex = year_data.get('CapitalExpenditures')
        
        # Working capital changes
        change_ar = year_data.get('ChangeInAccountsReceivable', 0)
        change_inv = year_data.get('ChangeInInventories', 0)
        change_ap = year_data.get('ChangeInAccountsPayable', 0)
        
        if net_income is not None:
            owner_earnings = net_income
            if depreciation is not None:
                owner_earnings += depreciation
            if capex is not None:
                owner_earnings -= capex
            
            working_capital_change = (change_ar or 0) + (change_inv or 0) - (change_ap or 0)
            owner_earnings -= working_capital_change
            
            return owner_earnings
        return None

    def calculate_current_ratio(self, current_assets: Optional[float], 
                              current_liabilities: Optional[float]) -> Optional[float]:
        """Calculate Current Ratio"""
        if current_assets is None:
            return None
        if current_liabilities is None or current_liabilities == 0:
            return 999.99 if current_assets > 0 else None  # Infinite liquidity
        if current_liabilities > 0:
            ratio = current_assets / current_liabilities
            return min(ratio, 999.99)  # Cap at reasonable maximum
        return None
    
    def calculate_debt_to_equity_ratio(self, debt_current: Optional[float], 
                                     debt_noncurrent: Optional[float], 
                                     stockholders_equity: Optional[float]) -> Optional[float]:
        """Calculate Debt-to-Equity Ratio"""
        total_debt = (debt_current or 0) + (debt_noncurrent or 0)
        
        if stockholders_equity and stockholders_equity > 0:
            return total_debt / stockholders_equity
        return None  # Negative equity makes ratio meaningless
    
    def calculate_interest_coverage_ratio(self, operating_income: Optional[float], 
                                        interest_expense: Optional[float]) -> Optional[float]:
        """Calculate Interest Coverage Ratio"""
        if operating_income is None:
            return None
        if interest_expense and interest_expense > 0:
            return operating_income / interest_expense
        elif interest_expense == 0 and operating_income:
            return 999.99  # No interest expense = infinite coverage
        return None

    def calculate_roe(self, net_income: Optional[float], 
                     stockholders_equity: Optional[float]) -> Optional[float]:
        """Calculate Return on Equity (ROE)"""
        if net_income is None or stockholders_equity is None:
            return None
        if stockholders_equity > 0:
            return (net_income / stockholders_equity) * 100
        return None  # Negative equity makes ROE not meaningful
    
    def calculate_roa(self, net_income: Optional[float], 
                     total_assets: Optional[float]) -> Optional[float]:
        """Calculate Return on Assets (ROA)"""
        if net_income is None or total_assets is None:
            return None
        if total_assets > 0:
            return (net_income / total_assets) * 100
        return None
    
    def calculate_roic(self, net_income: Optional[float], interest_expense: Optional[float], 
                      tax_expense: Optional[float], stockholders_equity: Optional[float], 
                      debt_current: Optional[float], debt_noncurrent: Optional[float]) -> Optional[float]:
        """Calculate Return on Invested Capital (ROIC)"""
        if net_income is None:
            return None
            
        # Estimate tax rate
        if tax_expense is not None and net_income is not None:
            pre_tax_income = net_income + tax_expense
            if pre_tax_income > 0:
                tax_rate = min(tax_expense / pre_tax_income, 0.50)
            else:
                tax_rate = 0.25
        else:
            tax_rate = 0.25  # Default assumption
        
        # Calculate NOPAT (Net Operating Profit After Tax)
        interest_tax_shield = (interest_expense or 0) * (1 - tax_rate)
        nopat = (net_income or 0) + interest_tax_shield
        
        # Calculate invested capital
        total_debt = (debt_current or 0) + (debt_noncurrent or 0)
        invested_capital = (stockholders_equity or 0) + total_debt
        
        if invested_capital > 0:
            return (nopat / invested_capital) * 100
        return None

    def calculate_earnings_retention_efficiency(self, retained_earnings: Optional[float], 
                                              current_net_income: Optional[float]) -> Optional[float]:
        """Calculate Retained Earnings to Net Income ratio"""
        if retained_earnings is None or current_net_income is None:
            return None
        if current_net_income != 0:
            return (retained_earnings / abs(current_net_income)) * 100
        return None
    
    def calculate_dividend_payout_ratio(self, dividends_paid: Optional[float], 
                                      net_income: Optional[float]) -> Optional[float]:
        """Calculate Dividend Payout Ratio"""
        if net_income is None or net_income <= 0:
            return 0  # No positive earnings to pay dividends from
        if dividends_paid is None:
            return 0  # No dividends paid
        return abs(dividends_paid) / net_income * 100  # Dividends usually negative in cash flow
    
    def calculate_capex_to_depreciation_ratio(self, capex: Optional[float], 
                                            depreciation: Optional[float]) -> Optional[float]:
        """Calculate Capital Expenditure to Depreciation ratio"""
        if depreciation is None or depreciation <= 0:
            return None
        if capex is None:
            return 0  # No capex
        return abs(capex) / depreciation  # Capex usually negative in cash flow

    def debug_metric_extraction(self, ticker: str, year: int, metric_name: str):
        logger.info(f"Debugging metric extraction for {ticker}, year {year}, metric {metric_name}")
        if ticker not in self.companies:
            logger.error(f"Ticker {ticker} not found in companies database.")
            return

        company_info = self.companies[ticker]
        cik = company_info['cik']
        logger.info(f"Fetching facts for CIK {cik}...")
        facts, error_reason = self.get_company_facts(cik)
        if not facts:
            logger.error(f"Could not fetch facts for CIK {cik}: {error_reason}")
            return

        logger.info(f"Top-level fact keys (namespaces): {facts.get('facts', {}).keys()}")

        # Check us-gaap first (existing logic)
        us_gaap = facts.get('facts', {}).get('us-gaap', {})
        metrics_dict = SEC_METRICS # Assuming we are debugging a company, not an ETF

        possible_tags = metrics_dict.get(metric_name, [])
        if not possible_tags:
            logger.warning(f"No possible XBRL tags found for metric {metric_name}.")
            # Continue to search for revenue-like tags in us-gaap even if the specific metric_name tags are not found
        
        logger.info(f"Searching for tags in us-gaap: {possible_tags}")
        for tag in possible_tags:
            if tag in us_gaap:
                units = us_gaap[tag].get('units', {})
                logger.info(f"  Found tag '{tag}' in us-gaap with units: {units.keys()}")
                for unit_type, entries in units.items():
                    logger.info(f"    Unit type '{unit_type}' has {len(entries)} entries.")
                    for entry in entries:
                        entry_date = entry.get('end', '')
                        entry_form = entry.get('form', '')
                        entry_fy = entry.get('fy')
                        entry_val = entry.get('val')
                        
                        form_priority = 0
                        if entry_form in ('10-K', '20-F'):
                            form_priority = 3
                        elif entry_form == '10-Q':
                            form_priority = 2
                        else:
                            form_priority = 1

                        fy_matches = (entry_fy == year) if entry_fy else False
                        date_contains_year = str(year) in entry_date

                        logger.info(f"      Entry: date={entry_date}, form={entry_form}, fy={entry_fy}, val={entry_val}, unit={unit_type}")
                        logger.info(f"        Matches year in date: {date_contains_year}, Fiscal year matches: {fy_matches}, Form priority: {form_priority}")

                        if entry_date and str(year) in entry_date:
                            logger.info(f"        Potential match for year {year} found in entry_date.")
                            # Further logic to pick the best value would go here, but for debugging, we just log.
            else:
                logger.info(f"  Tag '{tag}' not found in us-gaap facts.")

        # Iterate through all tags within the us-gaap namespace and print any that contain "revenue" or "sales"
        logger.info("\nSearching for revenue-like tags within the us-gaap namespace:")
        for fact_tag, fact_data in us_gaap.items():
            if "revenue" in fact_tag.lower() or "sales" in fact_tag.lower():
                units = fact_data.get('units', {})
                if 'USD' in units: # Only interested in USD values for now
                    logger.info(f"  Found potential revenue tag '{fact_tag}' in us-gaap")
                    for entry in units['USD']:
                        entry_date = entry.get('end', '')
                        entry_form = entry.get('form', '')
                        entry_fy = entry.get('fy')
                        entry_val = entry.get('val')
                        logger.info(f"    Entry: date={entry_date}, form={entry_form}, fy={entry_fy}, val={entry_val}")


    def calculate_derived_metrics(self, current_year_data: Dict, prior_year_data: Optional[Dict], 
                                year: int, ticker: str) -> Dict[str, Optional[float]]:
        """Calculate all 23 derived Warren Buffett-style metrics"""
        derived = {}
        
        current_year = datetime.now().year
        if year == current_year:
            derived['EarningsPerShare'] = self.get_latest_ttm_eps(ticker)
        else:
            # 1. EarningsPerShare
            derived['EarningsPerShare'] = self.calculate_eps(
                current_year_data.get('NetIncomeLoss'),
                current_year_data.get('WeightedAverageSharesBasic'),
                current_year_data.get('WeightedAverageSharesDiluted'),
                current_year_data.get('CommonStockSharesOutstanding')
            )
        
        # 2. PriceToEarning
        derived['PriceToEarning'] = self.calculate_pe_ratio(
            current_year_data.get('MarketPriceNonSplitAdjustedUSD'),
            derived['EarningsPerShare']
        )
        
        # 3. PEGRatio (requires earnings growth rate, calculated later when prior year data available)
        derived['PEGRatio'] = None  # Will be calculated after growth metrics
        
        # 4. BookValuePerShare
        derived['BookValuePerShare'] = self.calculate_book_value_per_share(
            current_year_data.get('StockholdersEquity'),
            current_year_data.get('CommonStockSharesOutstanding')
        )
        
        # 5. PriceToBook
        derived['PriceToBook'] = self.calculate_pb_ratio(
            current_year_data.get('MarketPriceNonSplitAdjustedUSD'),
            derived['BookValuePerShare']
        )
        
        # Growth metrics (require prior year data)
        if prior_year_data:
            # 5. RevenueGrowthRate
            derived['RevenueGrowthRate'] = self.calculate_revenue_growth_rate(
                current_year_data, prior_year_data, year, year - 1
            )
            
            # 6. NetIncomeGrowthRate
            derived['NetIncomeGrowthRate'] = self.calculate_net_income_growth_rate(
                current_year_data.get('NetIncomeLoss'),
                prior_year_data.get('NetIncomeLoss')
            )
            
            # Calculate PEGRatio now that we have NetIncomeGrowthRate
            derived['PEGRatio'] = self.calculate_peg_ratio(
                derived['PriceToEarning'],
                derived['NetIncomeGrowthRate']
            )
            
            # 7. BookValueGrowthRate
            derived['BookValueGrowthRate'] = self.calculate_book_value_growth_rate(
                current_year_data.get('StockholdersEquity'),
                prior_year_data.get('StockholdersEquity')
            )
        else:
            derived['RevenueGrowthRate'] = None
            derived['NetIncomeGrowthRate'] = None
            derived['BookValueGrowthRate'] = None
        
        # 8-11. Profitability Margins
        derived['GrossMargin'] = self.calculate_gross_margin(current_year_data, year)
        derived['OperatingMargin'] = self.calculate_operating_margin(current_year_data, year)
        derived['NetIncomeMargin'] = self.calculate_net_income_margin(current_year_data, year)
        derived['FreeCashFlowMargin'] = self.calculate_free_cash_flow_margin(current_year_data, year)
        
        # 12-13. Cash Flow Metrics
        derived['FreeCashFlow'] = self.calculate_free_cash_flow(
            current_year_data.get('NetCashFromOperatingActivities'),
            current_year_data.get('CapitalExpenditures')
        )
        derived['OwnerEarnings'] = self.calculate_owner_earnings(current_year_data)
        
        # 14-16. Financial Health Ratios
        derived['CurrentRatio'] = self.calculate_current_ratio(
            current_year_data.get('AssetsCurrent'),
            current_year_data.get('LiabilitiesCurrent')
        )
        derived['DebtToEquityRatio'] = self.calculate_debt_to_equity_ratio(
            current_year_data.get('DebtCurrent'),
            current_year_data.get('DebtNoncurrent'),
            current_year_data.get('StockholdersEquity')
        )
        derived['InterestCoverageRatio'] = self.calculate_interest_coverage_ratio(
            current_year_data.get('OperatingIncomeLoss'),
            current_year_data.get('InterestExpense')
        )
        
        # 17-19. Return Metrics
        derived['ReturnOnEquity'] = self.calculate_roe(
            current_year_data.get('NetIncomeLoss'),
            current_year_data.get('StockholdersEquity')
        )
        derived['ReturnOnAssets'] = self.calculate_roa(
            current_year_data.get('NetIncomeLoss'),
            current_year_data.get('Assets')
        )
        derived['ReturnOnInvestedCapital'] = self.calculate_roic(
            current_year_data.get('NetIncomeLoss'),
            current_year_data.get('InterestExpense'),
            current_year_data.get('IncomeTaxExpenseBenefit'),
            current_year_data.get('StockholdersEquity'),
            current_year_data.get('DebtCurrent'),
            current_year_data.get('DebtNoncurrent')
        )
        
        # 20-22. Capital Allocation Metrics
        derived['RetainedEarningsToNetIncome'] = self.calculate_earnings_retention_efficiency(
            current_year_data.get('RetainedEarnings'),
            current_year_data.get('NetIncomeLoss')
        )
        derived['DividendPayoutRatio'] = self.calculate_dividend_payout_ratio(
            current_year_data.get('CommonDividendsPaid'),
            current_year_data.get('NetIncomeLoss')
        )
        derived['CapitalExpenditureToDepreciation'] = self.calculate_capex_to_depreciation_ratio(
            current_year_data.get('CapitalExpenditures'),
            current_year_data.get('DepreciationAndAmortization')
        )
        
        return derived

    def calculate_etf_metrics(self, current_year_data: Dict, prior_year_data: Optional[Dict], 
                             year: int) -> Dict[str, Optional[float]]:
        """Calculate all 18 ETF-specific metrics focused on fund efficiency and performance"""
        etf_metrics = {}
        
        # Get current year values
        assets = current_year_data.get('Assets')
        liabilities = current_year_data.get('Liabilities', 0)
        assets_current = current_year_data.get('AssetsCurrent')
        liabilities_current = current_year_data.get('LiabilitiesCurrent', 0)
        cash = current_year_data.get('CashAndCashEquivalents', 0)
        stockholders_equity = current_year_data.get('StockholdersEquity')
        net_income = current_year_data.get('NetIncomeLoss')
        revenues = current_year_data.get('Revenues')
        assets_noncurrent = current_year_data.get('AssetsNoncurrent')
        
        # Fund Efficiency Metrics (6)
        
        # 1. ExpenseRatio - approximated as operating expenses / assets
        total_expenses = (current_year_data.get('TotalOperatingExpenses') or 
                         current_year_data.get('SellingGeneralAndAdministrativeExpense', 0))
        etf_metrics['ExpenseRatio'] = (total_expenses / assets * 100) if assets and total_expenses else None
        
        # 2. AssetTurnover - net income / average total assets (simplified, no prior year avg)
        etf_metrics['AssetTurnover'] = (net_income / assets) if assets and net_income else None
        
        # 3. OperatingExpenseRatio
        etf_metrics['OperatingExpenseRatio'] = (total_expenses / assets * 100) if assets and total_expenses else None
        
        # 4. AssetCoverageRatio - total assets / total liabilities
        etf_metrics['AssetCoverageRatio'] = (assets / liabilities) if assets and liabilities and liabilities > 0 else None
        
        # 5. LiquidityRatio - current assets / current liabilities
        etf_metrics['LiquidityRatio'] = (assets_current / liabilities_current) if assets_current and liabilities_current and liabilities_current > 0 else None
        
        # 6. CashToAssetsRatio
        etf_metrics['CashToAssetsRatio'] = (cash / assets * 100) if assets and cash else None
        
        # Performance Metrics (6)
        
        # 7. TotalReturn - requires prior year data
        if prior_year_data:
            prior_assets = prior_year_data.get('Assets')
            distributions = net_income or 0  # Simplified - actual distributions would be separate
            etf_metrics['TotalReturn'] = ((assets - prior_assets + distributions) / prior_assets * 100) if prior_assets and prior_assets > 0 else None
        else:
            etf_metrics['TotalReturn'] = None
            
        # 8. AssetGrowthRate - year-over-year asset growth
        if prior_year_data:
            prior_assets = prior_year_data.get('Assets')
            etf_metrics['AssetGrowthRate'] = ((assets - prior_assets) / prior_assets * 100) if prior_assets and prior_assets > 0 else None
        else:
            etf_metrics['AssetGrowthRate'] = None
            
        # 9. NAVGrowthRate - approximated using stockholders equity growth
        if prior_year_data and stockholders_equity:
            prior_equity = prior_year_data.get('StockholdersEquity')
            etf_metrics['NAVGrowthRate'] = ((stockholders_equity - prior_equity) / prior_equity * 100) if prior_equity and prior_equity > 0 else None
        else:
            etf_metrics['NAVGrowthRate'] = None
            
        # 10. IncomeYield - net income / average assets (for income-generating ETFs)
        etf_metrics['IncomeYield'] = (net_income / assets * 100) if assets and net_income else None
        
        # 11. ExpenseAdjustedReturn - total return - expense ratio
        total_return = etf_metrics.get('TotalReturn')
        expense_ratio = etf_metrics.get('ExpenseRatio')
        etf_metrics['ExpenseAdjustedReturn'] = (total_return - expense_ratio) if total_return is not None and expense_ratio is not None else None
        
        # 12. AssetUtilizationRatio - revenue / average assets
        etf_metrics['AssetUtilizationRatio'] = (revenues / assets) if assets and revenues else None
        
        # Financial Health Metrics (4)
        
        # 13. DebtToAssetRatio
        etf_metrics['DebtToAssetRatio'] = (liabilities / assets * 100) if assets and liabilities else None
        
        # 14. EquityToAssetRatio
        etf_metrics['EquityToAssetRatio'] = (stockholders_equity / assets * 100) if assets and stockholders_equity else None
        
        # 15. AssetStabilityRatio - non-current assets / total assets
        etf_metrics['AssetStabilityRatio'] = (assets_noncurrent / assets * 100) if assets and assets_noncurrent else None
        
        # 16. ExpenseEfficiencyRatio - expense ratio improvement year-over-year
        if prior_year_data:
            prior_expenses = (prior_year_data.get('TotalOperatingExpenses') or 
                            prior_year_data.get('SellingGeneralAndAdministrativeExpense', 0))
            prior_assets = prior_year_data.get('Assets')
            prior_expense_ratio = (prior_expenses / prior_assets * 100) if prior_assets and prior_expenses else None
            current_expense_ratio = etf_metrics.get('ExpenseRatio')
            etf_metrics['ExpenseEfficiencyRatio'] = (prior_expense_ratio - current_expense_ratio) if prior_expense_ratio and current_expense_ratio else None
        else:
            etf_metrics['ExpenseEfficiencyRatio'] = None
            
        # Scale & Market Metrics (2)
        
        # 17. AUMGrowthRate - same as AssetGrowthRate for ETFs
        etf_metrics['AUMGrowthRate'] = etf_metrics.get('AssetGrowthRate')
        
        # 18. SharePriceVolatility - requires market data, set to None for now
        etf_metrics['SharePriceVolatility'] = None  # Would need historical price data
        
        return etf_metrics

    def get_latest_ttm_financial_data(self, ticker: str, metric_name: str) -> Optional[float]:
        """Get the latest TTM financial data for a given metric using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            
            # Check if the metric requires summing quarterly data
            if metric_name in YFINANCE_QUARTERLY_METRIC_KEYS:
                quarterly_financials = stock.quarterly_financials
                if not quarterly_financials.empty:
                    yf_key = YFINANCE_QUARTERLY_METRIC_KEYS[metric_name]
                    # Sum the last four quarters
                    ttm_value = quarterly_financials.loc[yf_key].iloc[:4].sum()
                    if ttm_value is not None:
                        return float(ttm_value)
            
            # Fallback to info dictionary for other TTM metrics (like EPS)
            info = stock.info
            ttm_mapping = {
                'EarningsPerShare': 'trailingEps', # Already handled in calculate_derived_metrics
                'Revenues': 'trailingAnnualRevenue', # Fallback if quarterly sum fails
                'NetIncomeLoss': 'netIncomeToCommon' # Fallback if quarterly sum fails
            }
            
            yf_key = ttm_mapping.get(metric_name)
            if yf_key and yf_key in info and info[yf_key] is not None:
                return float(info[yf_key])
            
            return None
        except Exception as e:
            logger.debug(f"yfinance failed to get TTM {metric_name} for {ticker}: {e}")
            return None

    def _get_value_for_tag(self, us_gaap: Dict, tag: str, year: int) -> Optional[float]:
        if tag in us_gaap:
            units = us_gaap[tag].get('units', {})
            
            # Try USD first, then shares, then pure numbers
            for unit_type in ['USD', 'shares', 'pure']:
                if unit_type in units:
                    entries = units[unit_type]
                    
                    # Find best entry for the target year
                    best_value = None
                    best_form_priority = 0
                    
                    for entry in entries:
                        entry_date = entry.get('end', '')
                        entry_form = entry.get('form', '')
                        entry_fy = entry.get('fy')
                        
                        # Check if this entry is for our target year
                        if entry_date and str(year) in entry_date:
                            # Prefer 10-K forms over 10-Q
                            form_priority = 3 if entry_form in ('10-K', '20-F') else (2 if entry_form == '10-Q' else 1)
                            
                            # Also check fiscal year alignment
                            fy_matches = (entry_fy == year) if entry_fy else True
                            
                            # Take best entry based on form priority and fiscal year match
                            if (form_priority > best_form_priority) or (form_priority == best_form_priority and fy_matches):
                                best_value = entry.get('val')
                                best_form_priority = form_priority
                    
                    if best_value is not None:
                        return float(best_value)
        return None

    def extract_metric_value(self, facts: Dict, metric_name: str, year: int, ticker: str = None) -> Optional[float]:
        """Extract a specific metric value for a given year"""
        current_year = datetime.now().year
        
        # For current year, try to get TTM data for specific metrics
        if year == current_year and metric_name in [
            'Revenues', 'GrossProfit', 'OperatingIncomeLoss', 'NetIncomeLoss', 'NetCashFromOperatingActivities'
        ]:
            ttm_value = self.get_latest_ttm_financial_data(ticker, metric_name)
            if ttm_value is not None:
                logger.debug(f"Fetched TTM {metric_name} for {ticker} {year}: {ttm_value}")
                return ttm_value

        try:
            # Handle market price metrics from external API
            if metric_name == 'MarketPriceUSD':
                return self.get_market_price_year_end(ticker, year) if ticker else None
            elif metric_name == 'MarketPriceNonSplitAdjustedUSD':
                return self.get_market_price_non_split_adjusted(ticker, year) if ticker else None
            
            us_gaap = facts.get('facts', {}).get('us-gaap', {})
            
            # Choose appropriate metrics dictionary based on whether this is an ETF
            is_etf = ticker and self.is_etf(ticker)
            metrics_dict = ETF_RELEVANT_SEC_METRICS if is_etf else SEC_METRICS
            
            if metric_name == 'Revenues':
                # Special handling for Revenues to get the best possible value
                revenue_tags = []
                if year >= 2018:
                    revenue_tags.extend(metrics_dict.get('RevenueFromContracts', []))
                revenue_tags.extend(metrics_dict.get('Revenues', []))
                revenue_tags.extend(metrics_dict.get('SalesRevenueNet', []))

                for tag in revenue_tags:
                    value = self._get_value_for_tag(us_gaap, tag, year)
                    if value is not None:
                        return value
                return None
            
            if metric_name == 'CostOfRevenue':
                all_cost_tags = metrics_dict.get('CostOfRevenue', [])
                all_cost_tags.extend(metrics_dict.get('CostOfSales', []))
                best_cost = None
                for tag in all_cost_tags:
                    value = self._get_value_for_tag(us_gaap, tag, year)
                    if value is not None:
                        if best_cost is None or value > best_cost:
                            best_cost = value
                return best_cost
            
            if metric_name == 'CostOfRevenue':
                all_cost_tags = metrics_dict.get('CostOfRevenue', [])
                all_cost_tags.extend(metrics_dict.get('CostOfSales', []))
                best_cost = None
                for tag in all_cost_tags:
                    value = self._get_value_for_tag(us_gaap, tag, year)
                    if value is not None:
                        if best_cost is None or value > best_cost:
                            best_cost = value
                return best_cost
            
            # Get all possible XBRL tags for this metric
            possible_tags = metrics_dict.get(metric_name, [])
            
            for tag in possible_tags:
                value = self._get_value_for_tag(us_gaap, tag, year)
                if value is not None:
                    return value
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting {metric_name} for {year}: {e}")
            return None
    
    def extract_all_metrics(self, facts: Dict, years: List[int], ticker: str = None) -> Dict[int, Dict[str, Optional[float]]]:
        """Extract all metrics for all specified years including derived metrics"""
        results = {}
        sorted_years = sorted(years)
        
        # Choose appropriate metrics set based on whether this is an ETF
        is_etf = ticker and self.is_etf(ticker)
        metrics_to_extract = ETF_RELEVANT_SEC_METRICS if is_etf else SEC_METRICS
        
        if is_etf:
            logger.info(f"🏦 Using ETF-optimized metrics ({len(metrics_to_extract)} relevant metrics)")
        
        for i, year in enumerate(sorted_years):
            year_data = {}
            
            # Extract each SEC metric using appropriate metrics keys in order
            for metric_name in metrics_to_extract.keys():
                value = self.extract_metric_value(facts, metric_name, year, ticker)
                year_data[metric_name] = value

            # Calculate GrossProfit if it's not available directly
            if year_data.get('GrossProfit') is None:
                revenues = year_data.get('Revenues')
                cost_of_revenue = year_data.get('CostOfRevenue')
                if revenues is not None and cost_of_revenue is not None:
                    gross_profit = revenues - cost_of_revenue
                    if gross_profit > 0:
                        year_data['GrossProfit'] = gross_profit
                    else:
                        logger.debug(f"Calculated negative GrossProfit for {ticker} {year}. Revenues: {revenues}, CostOfRevenue: {cost_of_revenue}. Setting to None.")
                        year_data['GrossProfit'] = None
            
            # Calculate GrossProfit if it's not available directly
            if year_data.get('GrossProfit') is None:
                revenues = year_data.get('Revenues')
                cost_of_revenue = year_data.get('CostOfRevenue')
                print(f"Revenues: {revenues}")
                print(f"Cost of Revenue: {cost_of_revenue}")
                if revenues is not None and cost_of_revenue is not None:
                    gross_profit = revenues - cost_of_revenue
                    if gross_profit > 0:
                        year_data['GrossProfit'] = gross_profit
                        print(f"Gross Profit: {gross_profit}")
                    else:
                        logger.debug(f"Calculated negative GrossProfit for {ticker} {year}. Revenues: {revenues}, CostOfRevenue: {cost_of_revenue}. Setting to None.")
                        year_data['GrossProfit'] = None
                        print(f"Gross Profit: None")
            else:
                print(f"Gross Profit: {year_data.get('GrossProfit')}")
            
            # Calculate GrossProfit if it's not available directly
            if year_data.get('GrossProfit') is None:
                revenues = year_data.get('Revenues')
                cost_of_revenue = year_data.get('CostOfRevenue')
                print(f"Revenues: {revenues}")
                print(f"Cost of Revenue: {cost_of_revenue}")
                if revenues is not None and cost_of_revenue is not None:
                    gross_profit = revenues - cost_of_revenue
                    if gross_profit > 0:
                        year_data['GrossProfit'] = gross_profit
                        print(f"Gross Profit: {gross_profit}")
                    else:
                        logger.debug(f"Calculated negative GrossProfit for {ticker} {year}. Revenues: {revenues}, CostOfRevenue: {cost_of_revenue}. Setting to None.")
                        year_data['GrossProfit'] = None
                        print(f"Gross Profit: None")
            else:
                print(f"Gross Profit: {year_data.get('GrossProfit')}")
            
            if 'CostOfRevenue' not in year_data or year_data['CostOfRevenue'] is None:
                year_data['CostOfRevenue'] = year_data.get('CostsAndExpenses')

            # Calculate GrossProfit if it's not available directly
            if year_data.get('GrossProfit') is None:
                revenues = year_data.get('Revenues')
                cost_of_revenue = year_data.get('CostOfRevenue')
                if revenues is not None and cost_of_revenue is not None:
                    gross_profit = revenues - cost_of_revenue
                    if gross_profit > 0:
                        year_data['GrossProfit'] = gross_profit
                    else:
                        logger.debug(f"Calculated negative GrossProfit for {ticker} {year}. Revenues: {revenues}, CostOfRevenue: {cost_of_revenue}. Setting to None.")
                        year_data['GrossProfit'] = None
            
            # Calculate OperatingIncomeLoss if it's not available directly
            if year_data.get('OperatingIncomeLoss') is None:
                gross_profit = year_data.get('GrossProfit')
                
                # Try to find operating expenses from components if TotalOperatingExpenses is missing
                operating_expenses = year_data.get('TotalOperatingExpenses')
                
                if operating_expenses is None:
                    sga = year_data.get('SellingGeneralAndAdministrativeExpense')
                    rnd = year_data.get('ResearchAndDevelopmentExpense')
                    other_opex = year_data.get('OtherOperatingExpenses')
                    
                    expenses_list = [e for e in [sga, rnd, other_opex] if e is not None]
                    if expenses_list:
                        operating_expenses = sum(expenses_list)
                
                if gross_profit is not None and operating_expenses is not None:
                    year_data['OperatingIncomeLoss'] = gross_profit - operating_expenses
                    logger.debug(f"Calculated OperatingIncomeLoss for {ticker} {year}: {year_data['OperatingIncomeLoss']}")
            
            # Validate balance sheet equation using total equity (including non-controlling interests)
            assets = year_data.get('Assets')
            liabilities = year_data.get('Liabilities')  
            stockholders_equity = year_data.get('StockholdersEquity')
            equity_including_nci = year_data.get('StockholdersEquityIncludingNCI')
            noncontrolling_interest = year_data.get('NoncontrollingInterest')
            
            # Use the most comprehensive equity figure available
            total_equity = None
            equity_description = ""
            
            if equity_including_nci:
                total_equity = equity_including_nci
                equity_description = "equity including NCI"
            elif stockholders_equity and noncontrolling_interest:
                total_equity = stockholders_equity + noncontrolling_interest
                equity_description = "stockholders equity + NCI"
            elif stockholders_equity:
                total_equity = stockholders_equity
                equity_description = "stockholders equity only"
            
            if assets and liabilities and total_equity:
                balance_diff = abs(assets - (liabilities + total_equity))
                tolerance = assets * 0.05  # 5% tolerance
                
                if balance_diff > tolerance:
                    logger.warning(f"⚠️  Balance sheet equation violation for {year}: "
                                 f"Assets={assets:,.0f}, L+E={liabilities + total_equity:,.0f}, "
                                 f"diff={balance_diff:,.0f} (using {equity_description})")
                    
                    # If liabilities suspiciously equal assets, recalculate
                    if abs(liabilities - assets) < (assets * 0.01):
                        logger.info(f"  🔧 Correcting liabilities: Assets - Total Equity")
                        year_data['Liabilities'] = assets - total_equity
            
            # Calculate derived metrics (company-specific) or ETF metrics
            prior_year_data = results.get(year - 1) if i > 0 else None
            if self.is_etf(ticker):
                etf_metrics = self.calculate_etf_metrics(year_data, prior_year_data, year)
                year_data.update(etf_metrics)
            else:
                derived_metrics = self.calculate_derived_metrics(year_data, prior_year_data, year, ticker)
                year_data.update(derived_metrics)
            
            results[year] = year_data
        
        return results
    
    def get_csv_header(self, ticker: str, core_metrics_only: bool = False) -> List[str]:
        """Get appropriate CSV header based on whether ticker is ETF or company"""
        if self.is_etf(ticker):
            # ETF: Use ETF-relevant SEC metrics + ETF-specific derived metrics
            return ['Year'] + list(ETF_RELEVANT_SEC_METRICS.keys()) + ETF_METRICS
        
        if core_metrics_only:
            return ['Year'] + CORE_SEC_METRICS + DERIVED_METRICS
        else:
            # Company: Use all SEC metrics + company-specific derived metrics  
            return ['Year'] + list(SEC_METRICS.keys()) + DERIVED_METRICS
    
    def create_company_csv(self, ticker: str, company_data: Dict, output_dir: str, core_metrics_only: bool = False) -> bool:
        """Create CSV file for a single company"""
        try:
            # Prepare data for CSV
            rows = []
            years = sorted(company_data.keys())
            
            # Add header row using appropriate metrics for the ticker type
            header = self.get_csv_header(ticker, core_metrics_only)
            rows.append(header)
            
            # Add data rows
            for year in years:
                year_metrics = company_data[year]
                row = [year]
                
                # Iterate through the header to get the correct metrics in order
                for metric_name in header[1:]: # Skip 'Year'
                    value = year_metrics.get(metric_name)
                    
                    # Format values appropriately
                    if value is not None:
                        if metric_name in ['EarningsPerShare', 'BookValuePerShare']:
                            row.append(f"{value:.4f}")
                        elif metric_name.endswith('Ratio') or metric_name.endswith('Rate') or metric_name.endswith('Margin'):
                            row.append(f"{value:.2f}")
                        elif abs(value) >= 1e6:
                            row.append(f"{value/1e6:.2f}")
                        elif abs(value) >= 1000:
                            row.append(f"{value:.0f}")
                        else:
                            row.append(f"{value:.4f}")
                    else:
                        row.append('')
                
                rows.append(row)
            
            # Write CSV file
            csv_path = Path(output_dir) / f"{ticker}.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
            logger.debug(f"✅ Created CSV: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create CSV for {ticker}: {e}")
            return False
    
    def validate_csv_format(self, csv_path: str, core_metrics_only: bool = False) -> bool:
        """Validate CSV file format and structure"""
        try:
            # Extract ticker from filename
            ticker = Path(csv_path).stem  # Gets filename without extension
            
            # Read the CSV
            df = pd.read_csv(csv_path)
            
            # Check if we have the expected columns based on ticker type
            if self.is_etf(ticker):
                expected_columns = ['Year'] + list(ETF_RELEVANT_SEC_METRICS.keys()) + ETF_METRICS
                metric_type_desc = f"{len(ETF_RELEVANT_SEC_METRICS)} ETF-relevant SEC + {len(ETF_METRICS)} ETF"
            elif core_metrics_only:
                expected_columns = ['Year'] + CORE_SEC_METRICS + DERIVED_METRICS
                metric_type_desc = f"{len(CORE_SEC_METRICS)} core SEC + {len(DERIVED_METRICS)} derived"
            else:
                expected_columns = ['Year'] + list(SEC_METRICS.keys()) + DERIVED_METRICS
                metric_type_desc = f"{len(SEC_METRICS)} SEC + {len(DERIVED_METRICS)} derived"
                
            if list(df.columns) != expected_columns:
                logger.error(f"❌ Column mismatch in {csv_path}")
                logger.error(f"   Expected: {len(expected_columns)} columns ({metric_type_desc})")
                logger.error(f"   Found: {len(df.columns)} columns")
                return False
            
            # Check if we have reasonable year range
            years = df['Year'].tolist()
            if not years or min(years) < 2015 or max(years) > 2025:
                logger.warning(f"⚠️  Unusual year range in {csv_path}: {min(years) if years else 'None'}-{max(years) if years else 'None'}")
            
            # Check for completely empty rows
            non_year_columns = [col for col in df.columns if col != 'Year']
            empty_rows = df[non_year_columns].isnull().all(axis=1).sum()
            
            if empty_rows == len(df):
                logger.warning(f"⚠️  All data rows empty in {csv_path}")
                return False
            
            logger.debug(f"✅ CSV validation passed: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ CSV validation failed for {csv_path}: {e}")
            return False
    
    def process_company(self, ticker: str, years: List[int], output_dir: str, core_metrics_only: bool = False) -> bool:
        """Process a single company and create its CSV with respectful delays"""
        try:
            if ticker not in self.companies:
                logger.warning(f"⚠️  Ticker {ticker} not found in companies database")
                return False
            
            company_info = self.companies[ticker]
            cik = company_info['cik']
            company_name = company_info['title']
            
            logger.info(f"📊 Processing {ticker} ({company_name}) - CIK: {cik}")
            
            # Respectful delay before processing each company
            if self.api_call_count > 0:  # Skip delay for first company
                logger.debug(f"⏱️  Waiting {self.rate_limit_delay}s before next company...")
                time.sleep(self.rate_limit_delay)
            
            # Choose data source based on ticker type: Yahoo Finance for ETFs, 10-K for companies
            if self.is_etf(ticker):
                logger.info(f"🏦 ETF detected - using Yahoo Finance instead of SEC filings")
                # Use Yahoo Finance data for ETFs
                company_data = self.get_etf_yfinance_data(ticker, years)
                if not company_data or all(not year_data for year_data in company_data.values()):
                    logger.warning(f"⚠️  No Yahoo Finance data available for {ticker}")
                    logger.info(f"💡 ETF Data Note: {ticker} may not be available on Yahoo Finance")
                    logger.info(f"🔍 Alternative Sources: Try Financial Modeling Prep, Alpha Vantage, or ETF provider APIs")
                    self.failed_companies.append(ticker)
                    self.failure_details[ticker] = {
                        'company_name': company_name,
                        'cik': cik,
                        'error_reason': 'YAHOO_FINANCE_NOT_FOUND',
                        'error_type': 'ETF_DATA_UNAVAILABLE'
                    }
                    return False
                    
                # Add additional derived ETF metrics
                company_data = self.add_etf_derived_metrics(company_data, ticker)
                
            else:
                # Use traditional 10-K/10-Q data for companies (unchanged)
                facts, error_reason = self.get_company_facts(cik)
                if not facts:
                    logger.warning(f"⚠️  No SEC data available for {ticker}")
                    self.failed_companies.append(ticker)
                    self.failure_details[ticker] = {
                        'company_name': company_name,
                        'cik': cik,
                        'error_reason': error_reason or 'UNKNOWN_ERROR',
                        'error_type': 'SEC_DATA_UNAVAILABLE'
                    }
                    return False
                
                # Extract all metrics for all years (traditional approach)
                company_data = self.extract_all_metrics(facts, years, ticker)
            
            # Check if we got any useful data
            total_values = sum(
                sum(1 for v in year_data.values() if v is not None)
                for year_data in company_data.values()
            )
            
            if total_values == 0:
                logger.warning(f"⚠️  No financial data extracted for {ticker}")
                self.failed_companies.append(ticker)
                self.failure_details[ticker] = {
                    'company_name': company_name,
                    'cik': cik,
                    'error_reason': 'NO_FINANCIAL_DATA_EXTRACTED',
                    'error_type': 'DATA_EXTRACTION_FAILED'
                }
                return False
            
            # Create CSV file
            success = self.create_company_csv(ticker, company_data, output_dir, core_metrics_only)
            if not success:
                self.failed_companies.append(ticker)
                self.failure_details[ticker] = {
                    'company_name': company_name,
                    'cik': cik,
                    'error_reason': 'CSV_CREATION_FAILED',
                    'error_type': 'FILE_OPERATION_FAILED'
                }
                return False
            
            # Validate the created CSV
            csv_path = Path(output_dir) / f"{ticker}.csv"
            if not self.validate_csv_format(str(csv_path), core_metrics_only):
                logger.warning(f"⚠️  CSV validation failed for {ticker}")
                self.failure_details[ticker] = {
                    'company_name': company_name,
                    'cik': cik,
                    'error_reason': 'CSV_VALIDATION_FAILED',
                    'error_type': 'DATA_VALIDATION_FAILED'
                }
                return False
            
            logger.info(f"✅ Successfully processed {ticker} ({total_values} data points)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process {ticker}: {e}")
            if ticker not in self.failed_companies:
                self.failed_companies.append(ticker)
            self.failure_details[ticker] = {
                'company_name': company_info.get('title', 'UNKNOWN') if 'company_info' in locals() else 'UNKNOWN',
                'cik': company_info.get('cik', 'UNKNOWN') if 'company_info' in locals() else 'UNKNOWN',
                'error_reason': f'EXCEPTION_{str(e)[:50]}',
                'error_type': 'PROCESSING_EXCEPTION'
            }
            return False
    
    def build_metrics(self, tickers: List[str], years: List[int], output_dir: str, 
                     max_companies: Optional[int] = None, core_metrics_only: bool = False) -> Dict[str, Any]:
        """Build metrics for multiple companies with respectful rate limiting"""
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Separate ETFs from companies and create ETF subfolder if needed
        etf_tickers = []
        company_tickers = []
        
        for ticker in tickers:
            if self.is_etf(ticker):
                etf_tickers.append(ticker)
            else:
                company_tickers.append(ticker)
        
        # Create ETF subfolder if we have ETFs and we're not processing ETFs exclusively
        etf_output_dir = None
        if etf_tickers and company_tickers:  # Mixed processing
            etf_output_dir = Path(output_dir) / "ETF"
            etf_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created ETF subfolder: {etf_output_dir}")
        elif etf_tickers and not company_tickers:  # ETF-only processing
            etf_output_dir = output_dir
            logger.info(f"📈 Processing ETFs only in main directory")
        
        # Limit companies if specified (after separation to maintain ratio)
        if max_companies:
            total_companies = len(company_tickers) + len(etf_tickers)
            if total_companies > max_companies:
                # Maintain proportional representation
                company_ratio = len(company_tickers) / total_companies
                max_companies_count = int(max_companies * company_ratio)
                max_etfs_count = max_companies - max_companies_count
                
                company_tickers = company_tickers[:max_companies_count]
                etf_tickers = etf_tickers[:max_etfs_count]
        
        # Recombine for processing order
        all_tickers = company_tickers + etf_tickers
        
        logger.info(f"🚀 Starting SEC metrics extraction")
        logger.info(f"   Total: {len(all_tickers)} ({len(company_tickers)} companies + {len(etf_tickers)} ETFs)")
        logger.info(f"   Years: {min(years)}-{max(years)}")
        logger.info(f"   SEC Metrics: 102 (optimized - empty metrics removed)")
        
        # Market price status
        cache_count = len(PRICE_CACHE) if PRICE_CACHE else 0
        fmp_count = len(self.fmp_api_keys) if hasattr(self, 'fmp_api_keys') else 0
        price_status = []
        if cache_count > 0:
            price_status.append(f"cached: {cache_count} companies")
        if fmp_count > 0:
            price_status.append(f"FMP API: {fmp_count} keys")
        status_str = f"({', '.join(price_status)})" if price_status else "(unavailable)"
        
        logger.info(f"   Market Metrics: 2 {status_str}")
        
        # Context-aware metric descriptions
        if etf_tickers and not company_tickers:  # ETF-only
            logger.info(f"   ETF Metrics: {len(ETF_METRICS)} (fund efficiency & performance)")
            logger.info(f"   Total Metrics: {len(SEC_METRICS) + len(ETF_METRICS)} (104 base + 18 ETF)")
        elif company_tickers and not etf_tickers:  # Company-only  
            logger.info(f"   Derived Metrics: {len(DERIVED_METRICS)} (Warren Buffett-style)")
            logger.info(f"   Total Metrics: {len(SEC_METRICS) + len(DERIVED_METRICS)} (104 base + 23 derived)")
        else: # Mixed
            logger.info(f"   Derived/ETF Metrics: {len(DERIVED_METRICS)} / {len(ETF_METRICS)}")
            logger.info(f"   Total Metrics: Up to {len(SEC_METRICS) + max(len(DERIVED_METRICS), len(ETF_METRICS))} per ticker")

        
        processed_count = 0
        
        for ticker in all_tickers:
            
            # Determine correct output directory
            current_output_dir = etf_output_dir if self.is_etf(ticker) and etf_output_dir else output_dir
            
            self.process_company(ticker, years, current_output_dir, core_metrics_only)
            processed_count += 1
            
            logger.info(f"Progress: {processed_count}/{len(all_tickers)} tickers completed")
        
        # Return summary
        summary = {
            "total_processed": processed_count,
            "total_failed": len(self.failed_companies),
            "failed_tickers": self.failed_companies,
            "failure_details": self.failure_details
        }
        
        logger.info(f"✅ Processing complete. {summary['total_processed']} successful, {summary['total_failed']} failed.")
        return summary

def main():
    """Main function to run SEC metrics builder"""
    parser = argparse.ArgumentParser(
        description="SEC Metrics Builder - Extract comprehensive financial data",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Company selection
    parser.add_argument('--ticker', type=str, help='Single company ticker to process (e.g., AAPL)')
    parser.add_argument('--companies', nargs='+', help='List of company tickers to process')
    parser.add_argument('--from-file', type=str, help='File with a list of tickers to process (one per line)')
    parser.add_argument('--top', type=int, help='Process top N companies by market cap')
    
    # Year range
    parser.add_argument('--years', nargs=2, type=int, default=[2015, 2025], 
                        help='Start and end year for data extraction (e.g., 2018 2025)')
    
    # Output and logging
    parser.add_argument('--output-dir', type=str, default='sec_metrics_output', 
                        help='Directory to save the output CSV files')
    parser.add_argument('--failure-report', type=str, default='failed_companies.json',
                        help='File to save details of failed companies')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                        help='Delay in seconds between processing each company')
    parser.add_argument('--core-metrics-only', action='store_true', help='Extract only core financial metrics (20 total)')

    # ETF filtering
    etf_group = parser.add_mutually_exclusive_group()
    etf_group.add_argument('--skip-etf', action='store_true', help='Skip processing any tickers identified as ETFs')
    etf_group.add_argument('--etf-only', action='store_true', help='Process only tickers identified as ETFs')
    
    # API and Debugging
    parser.add_argument('--fmp-api-key', type=str, help='Financial Modeling Prep API key (overrides environment variable)')
    parser.add_argument('--debug-metric', type=str, help='Debug a specific metric for a ticker and year (format: TICKER,YEAR,METRIC_NAME)')
    
    args = parser.parse_args()
    
    builder = SECMetricsBuilder(rate_limit_delay=args.rate_limit, fmp_api_key=args.fmp_api_key)

    # Debug mode for a single metric
    if args.debug_metric:
        try:
            ticker, year_str, metric_name = args.debug_metric.split(',')
            year = int(year_str)
            builder.debug_metric_extraction(ticker, year, metric_name)
            return
        except ValueError:
            logger.error("Invalid format for --debug-metric. Use TICKER,YEAR,METRIC_NAME")
            return
    
    # Determine tickers to process
    tickers_to_process = []
    if args.ticker:
        tickers_to_process = [args.ticker]
    elif args.companies:
        tickers_to_process = args.companies
    elif args.from_file:
        try:
            with open(args.from_file, 'r') as f:
                tickers_to_process = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"❌ File not found: {args.from_file}")
            return
    elif args.top:
        # Get top N companies from the database (assuming it's sorted by some metric)
        # This requires the sec_companies.json to be pre-sorted or to contain sorting info.
        # For now, we'll just take the first N companies.
        tickers_to_process = list(builder.companies.keys())[:args.top]
        logger.info(f"Processing top {args.top} companies from the database.")
    else:
        logger.error("❌ No tickers specified. Use --ticker, --companies, --from-file, or --top.")
        return

    # Filter by ETF status
    tickers_to_process = builder.filter_companies_by_etf_status(
        tickers_to_process, args.skip_etf, args.etf_only
    )
    
    if not tickers_to_process:
        logger.warning("⚠️  No companies left to process after filtering.")
        return
        
    # Define year range
    start_year, end_year = args.years
    years = list(range(start_year, end_year + 1))
    
    # Build metrics
    summary = builder.build_metrics(
        tickers_to_process, 
        years, 
        args.output_dir, 
        core_metrics_only=args.core_metrics_only
    )
    
    # Save failure report if any failures occurred
    if summary['total_failed'] > 0:
        failure_report_path = Path(args.output_dir) / args.failure_report
        with open(failure_report_path, 'w') as f:
            json.dump(summary['failure_details'], f, indent=4)
        logger.info(f"📄 Failure report saved to: {failure_report_path}")

if __name__ == '__main__':
    main()
