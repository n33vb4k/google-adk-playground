import math
import numpy as np
from google.adk.agents import Agent
from google.adk.models.anthropic_llm import AnthropicLlm
from config.settings import get_settings
from portfolio_risk_agent.utils import _fetch_daily_closes, _annualised_volatility, _align_time_series, _log_returns_matrix

settings = get_settings()
logger = settings.get_logger()


def _build_model():
    """Builds the configured LLM backend (see Settings.model_provider).

    Anthropic models are instantiated explicitly via AnthropicLlm rather than
    passed as a plain "claude-..." string — ADK's model registry resolves
    bare Claude model strings to the Vertex AI-backed `Claude` class, which
    needs GOOGLE_CLOUD_PROJECT/LOCATION instead of ANTHROPIC_API_KEY.
    """
    if settings.model_provider == "anthropic":
        return AnthropicLlm(model=settings.anthropic_model)
    return settings.google_model


def get_asset_volatility(ticker: str) -> dict:
    """Returns the annualised volatility for a given asset ticker using live market data.

    Args:
        ticker: The asset ticker symbol (e.g. 'AAPL').

    Returns:
        A dict with 'status' and either 'volatility' or 'error_message'.
    """
    t = ticker.upper()
    closes = _fetch_daily_closes(t)

    if isinstance(closes, str):
        return {"status": "error", "error_message": closes}
    if len(closes) < 2:
        return {"status": "error", "error_message": f"Insufficient data for {t}"}

    return {
        "status": "success",
        "ticker": t,
        "volatility": _annualised_volatility(closes),
        "days_used": len(closes),
    }


# def compute_portfolio_volatility_zero_correlation(tickers: list[str], weights: list[float]) -> dict:
#     """Computes portfolio volatility assuming zero correlation between assets.

#     Note: assumes zero correlation — real portfolio variance is wᵀ Σ w where
#     Σ is the asset covariance matrix. This uses sqrt(Σ (w_i * σ_i)²).

#     Args:
#         tickers: List of ticker symbols.
#         weights: List of portfolio weights. Must sum to 1.0 and match tickers length.

#     Returns:
#         A dict with 'status' and either 'portfolio_volatility' or 'error_message'.
#     """
#     if len(tickers) != len(weights):
#         return {"status": "error", "error_message": "tickers and weights length mismatch"}
#     if abs(sum(weights) - 1.0) > 0.01:
#         return {"status": "error", "error_message": f"weights must sum to 1.0, got {sum(weights):.3f}"}

#     variance = 0.0
#     asset_vols = {}
#     for ticker, w in zip(tickers, weights):
#         result = get_asset_volatility(ticker)
#         if result["status"] == "error":
#             return {"status": "error", "error_message": f"{ticker}: {result['error_message']}"}
#         vol = result["volatility"]
#         asset_vols[ticker.upper()] = vol
#         variance += (w * vol) ** 2

#     return {
#         "status": "success",
#         "portfolio_volatility": math.sqrt(variance),
#         "asset_volatilities": asset_vols,
#         "method": "assumes zero correlation",
#     }

def compute_portfolio_volatility(tickers: list[str], weights: list[float]) -> dict:
    """Computes portfolio volatility using live market data and calculated correlations."""
    if len(tickers) != len(weights):
        return {"status": "error", "error_message": "tickers and weights length mismatch"}
    if abs(sum(weights) - 1.0) > 0.01:
        return {"status": "error", "error_message": f"weights must sum to 1.0, got {sum(weights):.3f}"}

    w = np.array(weights)

    try:
        aligned_data = _align_time_series(tickers)
        returns_matrix = _log_returns_matrix(aligned_data, tickers)
        cov_matrix = np.cov(returns_matrix, rowvar=False) * 252  # annualised covariance matrix
        portfolio_variance = w @ cov_matrix @ w.T
        portfolio_volatility = math.sqrt(portfolio_variance)

        asset_vols = np.sqrt(np.diag(cov_matrix))
        correlation_matrix = cov_matrix / np.outer(asset_vols, asset_vols)
        days_used = returns_matrix.shape[0] + 1  # +1 because returns are one less than prices
        method = "calculated from live data with correlations"
    except Exception as e:
        return {"status": "error", "error_message": str(e)}
    
    return {
        "status": "success",
        "portfolio_volatility": portfolio_volatility,
        "asset_volatilities": dict(zip(tickers, asset_vols.tolist())),
        "correlation_matrix": correlation_matrix.tolist(),
        "days_used": days_used,
        "method": method,
    }


root_agent = Agent(
    model=_build_model(),
    name="portfolio_risk_agent",
    description="Analyses portfolio risk using live market data.",
    instruction=(
        "You are a portfolio risk analyst. Help users understand the volatility "
        "of individual assets and portfolios. Use the tools provided to fetch "
        "live market data and compute risk. Always be precise about assumptions "
        "and limitations of any calculation you report."
    ),
    tools=[get_asset_volatility, compute_portfolio_volatility],
)
