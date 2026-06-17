import math
import numpy as np
from functools import lru_cache
import requests
import yfinance as yf

from config.settings import get_settings

settings = get_settings()
logger = settings.get_logger()


@lru_cache(maxsize=128)
def _fetch_daily_closes_alpha_vantage(ticker: str) -> tuple[tuple[str, float], ...] | str:
    """Returns list of closing prices (newest first) or an error string."""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "compact",  # 100 trading days — sufficient for annualised vol
        "apikey": settings.vantage_alpha_api_key,
    }
    try:
        resp = requests.get(settings.vantage_alpha_base_url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return str(e)

    data = resp.json()
    logger.info(f"Fetched data for {ticker} via Alpha Vantage")

    if "Error Message" in data:
        return data["Error Message"]
    if "Information" in data:
        return data["Information"]  # rate-limit message

    time_series = data.get("Time Series (Daily)", {})
    if not time_series:
        return f"No time-series data returned for {ticker}"

    sorted_dates = sorted(time_series.keys(), reverse=True)
    closes = tuple((d, float(time_series[d]["4. close"])) for d in sorted_dates)
    logger.info(f"Extracted {len(closes)} closing prices for {ticker}")

    return closes


@lru_cache(maxsize=128)
def _fetch_daily_closes_yfinance(ticker: str) -> tuple[tuple[str, float], ...] | str:
    """Returns list of closing prices (newest first) or an error string."""
    try:
        history = yf.Ticker(ticker).history(period="6mo")  # ~125 trading days
    except Exception as e:
        return str(e)

    if history.empty:
        return f"No time-series data returned for {ticker}"

    logger.info(f"Fetched data for {ticker} via yfinance")
    dates = [ts.strftime("%Y-%m-%d") for ts in history.index]
    closes = tuple(zip(dates, history["Close"].astype(float)))[::-1]
    logger.info(f"Extracted {len(closes)} closing prices for {ticker}")

    return closes


def _fetch_daily_closes(ticker: str) -> tuple[tuple[str, float], ...] | str:
    """Dispatches to the configured market-data provider (see Settings.data_provider)."""
    if settings.data_provider == "yfinance":
        return _fetch_daily_closes_yfinance(ticker)
    return _fetch_daily_closes_alpha_vantage(ticker)


def _annualised_volatility(closes: tuple[tuple[str, float], ...]) -> float:
    """Computes annualised volatility from a series of closing prices."""
    log_returns = [math.log(closes[i][1] / closes[i + 1][1]) for i in range(len(closes) - 1)]
    n = len(log_returns)
    mean = sum(log_returns) / n
    variance = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
    ann_vol = math.sqrt(variance) * math.sqrt(252)

    logger.info(f"Computed annualised volatility: {ann_vol:.4f} from {n} log returns")
    return ann_vol


def _align_time_series(tickers: list[str]) -> dict[str, list[float]]:
    """Aligns time series data for multiple tickers by date."""
    closes_by_ticker: dict[str, dict[str, float]] = {}

    for ticker in tickers:
        data = _fetch_daily_closes(ticker)
        if isinstance(data, str):
            raise ValueError(f"{ticker}: {data}")
        closes_by_ticker[ticker] = dict(data)
    
    # set(data) -> set of keys from date to close dict, which is set of dates
    # * unpacks this for each ticker in closes_by_ticker so inside intersectio is multuple set arguments
    # intersection ensures no None or empty values
    common_dates = sorted(set.intersection(*(set(data) for data in closes_by_ticker.values())))
    if len(common_dates) < 2:
        raise ValueError(f"Need >2 common dates accros all tickers, got {len(common_dates)}")
    
    logger.info("time series data aligned successfully")
    return {
        ticker: [closes_by_ticker[ticker][date] for date in common_dates]
        for ticker in tickers
    }


def _log_returns_matrix(aligned_data: dict[str, list[float]], tickers: list[str]) -> np.ndarray:
    """Calculates the returns matrix from aligned closing prices."""
    prices = np.array([aligned_data[ticker] for ticker in tickers]).T
    # need to transpose because we want each column to represent a ticker and each row to represent a date
    returns = np.diff(np.log(prices), axis=0)  # log returns
    logger.info("returns matrix calculated successfully")
    return returns