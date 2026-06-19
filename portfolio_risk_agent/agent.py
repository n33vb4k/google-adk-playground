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

# note that this is not using factor models or other advanced risk models; it is a simple historical volatility calculation based on live market data and correlations.
def compute_portfolio_volatility(tickers: list[str], weights: list[float], benchmark_ticker: str = "SPY") -> dict:
    """Computes portfolio volatility using live market data and calculated correlations."""
    if len(tickers) != len(weights):
        return {"status": "error", "error_message": "tickers and weights length mismatch"}
    if abs(sum(weights) - 1.0) > 0.01:
        return {"status": "error", "error_message": f"weights must sum to 1.0, got {sum(weights):.3f}"}

    w = np.array(weights)

    try:
        all_tickers = tickers + [benchmark_ticker]

        aligned_data = _align_time_series(all_tickers)
        returns_matrix = _log_returns_matrix(aligned_data, all_tickers)

        asset_returns = returns_matrix[:, :-1]  # exclude benchmark returns
        benchmark_returns = returns_matrix[:, -1]  # benchmark returns

        cov_matrix = np.cov(asset_returns, rowvar=False) * 252  # annualised covariance matrix
        portfolio_variance = w @ cov_matrix @ w.T
        portfolio_volatility = math.sqrt(portfolio_variance)

        portfolio_returns = asset_returns @ w
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = float(np.std(active_returns, ddof=1) * np.sqrt(252))  # annualised tracking error

        sigma_w = cov_matrix @ w # margin contribution to risk
        component_contributions = w * sigma_w # element-wise: sums to portfolio variance
        risk_contributions_pct = (component_contributions / portfolio_volatility)
        risk_contributions = dict(zip(tickers, risk_contributions_pct.tolist()))

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
        "tracking_error": tracking_error,
        "risk_contributions_pct": risk_contributions,
    }


root_agent = Agent(
    model=_build_model(),
    name="portfolio_risk_agent",
    description="Analyses portfolio risk using live market data.",
    instruction=("You are a portfolio risk analyst. Report both total volatility "
    "(standalone risk) and tracking error (risk relative to a benchmark), "
    "and explain the difference — total vol matters for absolute drawdown, "
    "tracking error matters for active management mandates. Reference the "
    "correlation matrix to explain why portfolio vol differs from a naive "
    "weighted average of asset vols. Always be precise about assumptions "
    "and limitations of any calculation. When reporting portfolio risk," 
    "explain which assets contribute most to total variance using the "
    "risk_contribution_pct field."),
    tools=[get_asset_volatility, compute_portfolio_volatility],
)


