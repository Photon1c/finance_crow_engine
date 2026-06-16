import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_STOCK_DIR = "F:/inputs/stocks"
DEFAULT_OPTION_DIR = "F:/inputs/options/log"


def parse_price(value) -> float:
    """Parse a price cell that may include currency formatting."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace("$", "").replace(",", "").strip())


def get_stock_close_series(stock_df: pd.DataFrame) -> pd.Series:
    """Return numeric close prices sorted by date."""
    col = "Close/Last" if "Close/Last" in stock_df.columns else "Close"
    return stock_df[col].map(parse_price).dropna()


def get_latest_stock_row(ticker, base_dir=DEFAULT_STOCK_DIR) -> tuple[float, object]:
    """Return latest close price and date from stock CSV."""
    df = load_stock_data(ticker, base_dir=base_dir)
    if len(df) == 0:
        raise ValueError(f"No data found for {ticker}")

    closes = get_stock_close_series(df)
    latest_date = df["Date"].iloc[-1] if "Date" in df.columns else None
    return float(closes.iloc[-1]), latest_date


def load_stock_data(ticker, base_dir=DEFAULT_STOCK_DIR):
    ticker_upper = ticker.upper()
    filepath = Path(base_dir) / f"{ticker_upper}.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, parse_dates=["Date"])
        # Sort by date to ensure proper ordering
        if "Date" in df.columns:
            df = df.sort_values("Date", ascending=True)
        return df
    else:
        raise FileNotFoundError(f"Stock file not found: {filepath}")

def get_latest_price(ticker, base_dir=DEFAULT_STOCK_DIR, verbose=True):
    """
    Get the most recent closing price for a ticker.
    Returns the price from the row with the latest date.
    """
    price, latest_date = get_latest_stock_row(ticker, base_dir=base_dir)
    if verbose:
        if latest_date is not None:
            print(f"Latest {ticker} price: ${price:.2f} (as of {latest_date})")
        else:
            print(f"Latest {ticker} price: ${price:.2f}")
    return price

def get_most_recent_option_date(ticker, base_dir=DEFAULT_OPTION_DIR, verbose=True):
    ticker_lower = ticker.lower()
    ticker_dir = Path(base_dir) / ticker_lower

    date_dirs = [d for d in ticker_dir.iterdir() if d.is_dir()]
    if not date_dirs:
        raise FileNotFoundError(f"No date directories found for {ticker_lower}.")

    most_recent_dir = max(date_dirs, key=lambda d: d.stat().st_mtime)
    most_recent_date = most_recent_dir.name
    if verbose:
        print(f"Most recent date for {ticker_lower}: {most_recent_date}")
    return most_recent_date

def load_option_chain_data(ticker, date=None, base_dir=DEFAULT_OPTION_DIR):
    ticker_lower = ticker.lower()
    ticker_dir = Path(base_dir) / ticker_lower

    if date:
        date_dir = ticker_dir / date
    else:
        # Auto-discover the most recent date directory
        date_dirs = [d for d in ticker_dir.iterdir() if d.is_dir()]
        if not date_dirs:
            raise FileNotFoundError(f"No date directories found for {ticker_lower}.")
        date_dir = max(date_dirs, key=lambda d: d.stat().st_mtime)  # most recently modified

    filepath = date_dir / f"{ticker_lower}_quotedata.csv"
    if filepath.exists():
        return pd.read_csv(filepath, skiprows=3)
    else:
        raise FileNotFoundError(f"Option chain file not found: {filepath}")

    filepath = date_dir / f"{ticker_lower}_quotedata.csv"
    if filepath.exists():
        return pd.read_csv(filepath, skiprows=3)
    else:
        raise FileNotFoundError(f"Option chain file not found: {filepath}")


def load_option_data(ticker, date=None, base_dir=DEFAULT_OPTION_DIR):
    """Alias for load_option_chain_data kept for backward compatibility."""
    return load_option_chain_data(ticker, date=date, base_dir=base_dir)


def _parse_expiration(value) -> Optional[datetime]:
    if value in ("", None) or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _reference_date_from_chain(chain_date: Optional[str]) -> datetime:
    if not chain_date:
        return datetime.now()
    parsed = pd.to_datetime(chain_date.replace("_", "-"), errors="coerce")
    if pd.isna(parsed):
        return datetime.now()
    return parsed.to_pydatetime()


def _atm_option_row(option_df: pd.DataFrame, spot: float) -> pd.Series:
    strikes = pd.to_numeric(option_df["Strike"], errors="coerce")
    idx = (strikes - spot).abs().idxmin()
    return option_df.loc[idx]


def _select_expiry(
    option_df: pd.DataFrame,
    spot: float,
    reference_date: datetime,
    min_dte: int = 0,
    max_dte: Optional[int] = None,
) -> pd.DataFrame:
    """Keep rows for the best-matching expiration given DTE preferences."""
    df = option_df.copy()
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    df["expiration_dt"] = df["Expiration Date"].map(_parse_expiration)
    df = df[df["expiration_dt"].notna() & df["Strike"].notna()]
    if df.empty:
        return option_df.copy()

    df["dte"] = df["expiration_dt"].map(lambda dt: max((dt.date() - reference_date.date()).days, 0))
    candidates = df.copy()
    if max_dte is not None:
        in_range = candidates[(candidates["dte"] >= min_dte) & (candidates["dte"] <= max_dte)]
        if len(in_range) > 0:
            candidates = in_range
        elif min_dte > 0:
            candidates = candidates[candidates["dte"] >= min_dte]
            if candidates.empty:
                candidates = df

    target_dte = 7 if max_dte is not None else 0
    expiry_dtes = (
        candidates.groupby("Expiration Date")["dte"]
        .min()
        .reset_index()
        .sort_values("dte", key=lambda s: (s - target_dte).abs())
    )
    chosen_expiry = expiry_dtes.iloc[0]["Expiration Date"]
    return candidates[candidates["Expiration Date"] == chosen_expiry].copy()


def _option_side_columns(is_put: bool) -> dict:
    if is_put:
        return {
            "symbol": "Puts",
            "iv": "IV.1",
            "delta": "Delta.1",
            "gamma": "Gamma.1",
            "bid": "Bid.1",
            "ask": "Ask.1",
            "volume": "Volume.1",
            "oi": "Open Interest.1",
        }
    return {
        "symbol": "Calls",
        "iv": "IV",
        "delta": "Delta",
        "gamma": "Gamma",
        "bid": "Bid",
        "ask": "Ask",
        "volume": "Volume",
        "oi": "Open Interest",
    }


def _prepare_chain_rows(
    option_df: pd.DataFrame,
    reference_date: datetime,
    *,
    min_dte: int = 0,
    max_dte: Optional[int] = None,
) -> pd.DataFrame:
    df = option_df.copy()
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    df["expiration_dt"] = df["Expiration Date"].map(_parse_expiration)
    df = df[df["expiration_dt"].notna() & df["Strike"].notna()].copy()
    df["dte"] = df["expiration_dt"].map(
        lambda dt: max((dt.date() - reference_date.date()).days, 0)
    )
    if max_dte is not None:
        in_range = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]
        if len(in_range) > 0:
            return in_range
        if min_dte > 0:
            relaxed = df[df["dte"] >= min_dte]
            if len(relaxed) > 0:
                return relaxed
    return df


def _contract_from_row(row, side: dict) -> dict:
    bid = parse_price(row.get(side["bid"]))
    ask = parse_price(row.get(side["ask"]))
    iv = parse_price(row.get(side["iv"]))
    delta = parse_price(row.get(side["delta"]))
    gamma = parse_price(row.get(side["gamma"])) if side["gamma"] in row.index else float("nan")
    volume = parse_price(row.get(side["volume"]))
    oi = parse_price(row.get(side["oi"]))
    mid = (bid + ask) / 2 if not np.isnan(bid) and not np.isnan(ask) else (ask or bid or 0.0)
    return {
        "symbol": str(row.get(side["symbol"], "") or ""),
        "strike": float(row["Strike"]),
        "iv": float(iv) if not np.isnan(iv) else 0.0,
        "delta": float(delta) if not np.isnan(delta) else 0.0,
        "gamma": float(gamma) if not np.isnan(gamma) else 0.0,
        "bid": float(bid) if not np.isnan(bid) else 0.0,
        "ask": float(ask) if not np.isnan(ask) else 0.0,
        "mid": float(mid),
        "volume": float(volume) if not np.isnan(volume) else 0.0,
        "open_interest": float(oi) if not np.isnan(oi) else 0.0,
        "expiry": str(row.get("Expiration Date", "")),
        "dte": int(row.get("dte", 0)),
    }


def find_contract_by_max_premium(
    option_df: pd.DataFrame,
    spot: float,
    max_premium: float,
    is_put: bool = True,
    *,
    reference_date: Optional[datetime] = None,
    min_dte: int = 0,
    max_dte: Optional[int] = None,
) -> dict:
    """
    Pick the most liquid contract with ask price at or below max_premium.

    Prefers the highest premium under the cap (closest to ATM among cheap contracts).
    """
    ref = reference_date or datetime.now()
    side = _option_side_columns(is_put)
    df = _prepare_chain_rows(option_df, ref, min_dte=min_dte, max_dte=max_dte)

    best = None
    best_score = float("-inf")
    for _, row in df.iterrows():
        contract = _contract_from_row(row, side)
        ask = contract["ask"]
        if not contract["symbol"] or ask <= 0 or ask > max_premium:
            continue

        spread = contract["ask"] - contract["bid"] if contract["bid"] > 0 else contract["ask"]
        strike_distance = abs(contract["strike"] - spot) / max(spot, 1.0)
        target_dte = 7 if max_dte is not None else contract["dte"]
        dte_penalty = abs(contract["dte"] - target_dte) * 0.05

        score = (
            ask * 4.0
            + np.log1p(contract["volume"]) * 0.5
            + np.log1p(contract["open_interest"]) * 0.1
            - strike_distance * 8.0
            - spread * 2.0
            - dte_penalty
        )
        if score > best_score:
            best_score = score
            best = contract

    if best is None:
        raise ValueError(
            f"No {'put' if is_put else 'call'} found with ask <= ${max_premium:.2f} "
            f"at spot ${spot:.2f}. Try raising --max-premium or widening the DTE window."
        )
    return best


def find_optimal_contract(
    option_df: pd.DataFrame,
    spot: float,
    is_put: bool = True,
    *,
    reference_date: Optional[datetime] = None,
    min_dte: int = 0,
    max_dte: Optional[int] = None,
    near_atm_pct: float = 0.025,
    max_premium: Optional[float] = None,
) -> dict:
    """Pick the most liquid contract; optionally cap entry premium via max_premium."""
    if max_premium is not None:
        return find_contract_by_max_premium(
            option_df,
            spot,
            max_premium,
            is_put=is_put,
            reference_date=reference_date,
            min_dte=min_dte,
            max_dte=max_dte,
        )

    ref = reference_date or datetime.now()
    df = _select_expiry(option_df, spot, ref, min_dte=min_dte, max_dte=max_dte)
    side = _option_side_columns(is_put)
    target_delta = -0.45 if is_put else 0.45
    near_atm = df[df["Strike"].sub(spot).abs() / max(spot, 1.0) <= near_atm_pct]
    scan_df = near_atm if len(near_atm) > 0 else df

    best = None
    best_score = float("-inf")
    for _, row in scan_df.iterrows():
        strike = row["Strike"]
        if pd.isna(strike):
            continue

        symbol = str(row.get(side["symbol"], "") or "")
        iv = parse_price(row.get(side["iv"]))
        delta = parse_price(row.get(side["delta"]))
        gamma = parse_price(row.get(side["gamma"])) if side["gamma"] in row.index else float("nan")
        bid = parse_price(row.get(side["bid"]))
        ask = parse_price(row.get(side["ask"]))
        volume = parse_price(row.get(side["volume"]))
        oi = parse_price(row.get(side["oi"]))

        if not symbol or (np.isnan(bid) and np.isnan(ask)):
            continue

        mid = (bid + ask) / 2 if not np.isnan(bid) and not np.isnan(ask) else (bid or ask or 0.0)
        spread = (ask - bid) if not np.isnan(bid) and not np.isnan(ask) else 1.0
        moneyness = abs(strike - spot) / max(spot, 1.0)
        delta_penalty = abs((delta if not np.isnan(delta) else 0.0) - target_delta)

        score = (
            np.log1p(volume or 0) * 0.6
            + np.log1p(oi or 0) * 0.15
            - delta_penalty * 18.0
            - moneyness * 12.0
            - spread * 2.5
        )
        if score > best_score:
            best_score = score
            dte = int(row.get("dte", 0)) if "dte" in row.index else 0
            best = {
                "symbol": symbol,
                "strike": float(strike),
                "iv": float(iv) if not np.isnan(iv) else 0.0,
                "delta": float(delta) if not np.isnan(delta) else 0.0,
                "gamma": float(gamma) if not np.isnan(gamma) else 0.0,
                "bid": float(bid) if not np.isnan(bid) else 0.0,
                "ask": float(ask) if not np.isnan(ask) else 0.0,
                "mid": float(mid),
                "volume": float(volume) if not np.isnan(volume) else 0.0,
                "open_interest": float(oi) if not np.isnan(oi) else 0.0,
                "expiry": str(row.get("Expiration Date", "")),
                "dte": dte,
            }

    if best is None:
        row = _atm_option_row(df, spot)
        strike = float(row["Strike"])
        if is_put:
            bid = parse_price(row.get("Bid.1", 0))
            ask = parse_price(row.get("Ask.1", 0))
            best = {
                "symbol": str(row.get("Puts", "")),
                "strike": strike,
                "iv": parse_price(row.get("IV.1", 0)) or 0.0,
                "delta": parse_price(row.get("Delta.1", 0)) or 0.0,
                "gamma": parse_price(row.get("Gamma.1", 0)) or 0.0,
                "bid": bid if not np.isnan(bid) else 0.0,
                "ask": ask if not np.isnan(ask) else 0.0,
                "mid": (bid + ask) / 2 if not np.isnan(bid) and not np.isnan(ask) else 0.0,
                "volume": parse_price(row.get("Volume.1", 0)) or 0.0,
                "open_interest": parse_price(row.get("Open Interest.1", 0)) or 0.0,
                "expiry": str(row.get("Expiration Date", "")),
                "dte": 0,
            }
        else:
            bid = parse_price(row.get("Bid", 0))
            ask = parse_price(row.get("Ask", 0))
            best = {
                "symbol": str(row.get("Calls", "")),
                "strike": strike,
                "iv": parse_price(row.get("IV", 0)) or 0.0,
                "delta": parse_price(row.get("Delta", 0)) or 0.0,
                "gamma": parse_price(row.get("Gamma", 0)) or 0.0,
                "bid": bid if not np.isnan(bid) else 0.0,
                "ask": ask if not np.isnan(ask) else 0.0,
                "mid": (bid + ask) / 2 if not np.isnan(bid) and not np.isnan(ask) else 0.0,
                "volume": parse_price(row.get("Volume", 0)) or 0.0,
                "open_interest": parse_price(row.get("Open Interest", 0)) or 0.0,
                "expiry": str(row.get("Expiration Date", "")),
                "dte": 0,
            }

    return best


def validate_contract_near_spot(
    contract: dict,
    spot: float,
    *,
    near_atm_pct: float = 0.025,
    ticker: str = "",
) -> float:
    """
    Ensure selected strike is near current spot.

    Returns moneyness as a fraction of spot. Raises ValueError if too far OTM/ITM.
    """
    strike = float(contract.get("strike", 0))
    moneyness = abs(strike - spot) / max(spot, 1.0)
    if moneyness > near_atm_pct:
        label = ticker or contract.get("symbol", "contract")
        raise ValueError(
            f"{label} strike ${strike:.0f} is {moneyness:.1%} from spot ${spot:.2f} "
            f"(allowed <= {near_atm_pct:.1%}). "
            "Stock CSV and option chain may be out of sync, or chain strikes are stale."
        )
    return moneyness


def get_stock_close_on_date(stock_df: pd.DataFrame, target_date) -> Optional[float]:
    """Return the close on target_date, or the nearest prior trading close."""
    if "Date" not in stock_df.columns:
        return None
    dates = pd.to_datetime(stock_df["Date"]).dt.tz_localize(None).dt.normalize()
    target = pd.Timestamp(target_date).tz_localize(None).normalize()
    exact = stock_df.loc[dates == target]
    if len(exact) > 0:
        return float(get_stock_close_series(exact).iloc[-1])
    prior = stock_df.loc[dates <= target]
    if len(prior) > 0:
        return float(get_stock_close_series(prior).iloc[-1])
    return None


def find_optimal_exit_plan(
    option_df: pd.DataFrame,
    entry_contract: dict,
    *,
    reference_date: Optional[datetime] = None,
    is_put: bool = True,
    min_hold_days: int = 7,
    max_hold_days: int = 14,
) -> dict:
    """
    Choose an exit horizon between min_hold_days and max_hold_days.

    Uses chain bids on the same strike with remaining DTE after the hold window.
    When the entry contract would expire before exit, uses a floor bid of 0.01.
    """
    ref = reference_date or datetime.now()
    side = _option_side_columns(is_put)
    strike = float(entry_contract["strike"])
    entry_dte = int(entry_contract.get("dte", 0))
    df = _prepare_chain_rows(option_df, ref, min_dte=0, max_dte=60)

    best = None
    for hold_days in range(min_hold_days, max_hold_days + 1):
        remaining_dte = entry_dte - hold_days
        exit_bid = 0.01
        exit_expiry = entry_contract.get("expiry", "")
        exit_iv = entry_contract.get("iv", 0.0)
        exit_delta = entry_contract.get("delta", 0.0)

        if remaining_dte > 0:
            matches = df[(df["Strike"] == strike) & (df["dte"] == remaining_dte)]
            if len(matches) > 0:
                row = matches.iloc[0]
                contract = _contract_from_row(row, side)
                exit_bid = contract["bid"] if contract["bid"] > 0 else contract["mid"]
                exit_expiry = contract["expiry"]
                exit_iv = contract["iv"]
                exit_delta = contract["delta"]
        elif remaining_dte == 0:
            matches = df[(df["Strike"] == strike) & (df["dte"] == 0)]
            if len(matches) > 0:
                row = matches.iloc[0]
                contract = _contract_from_row(row, side)
                exit_bid = contract["bid"] if contract["bid"] > 0 else contract["mid"]
                exit_expiry = contract["expiry"]
                exit_iv = contract["iv"]
                exit_delta = contract["delta"]

        candidate = {
            "hold_days": hold_days,
            "exit_bid": float(max(exit_bid, 0.01)),
            "remaining_dte": max(remaining_dte, 0),
            "exit_expiry": exit_expiry,
            "exit_iv": exit_iv,
            "exit_delta": exit_delta,
        }
        if best is None or candidate["exit_bid"] > best["exit_bid"]:
            best = candidate

    if best is None:
        raise ValueError(
            f"No exit plan found for strike ${strike:.0f} between "
            f"{min_hold_days}-{max_hold_days} day hold."
        )
    return best


def load_market_snapshot(
    ticker: str,
    *,
    stock_dir: str = DEFAULT_STOCK_DIR,
    option_dir: str = DEFAULT_OPTION_DIR,
    is_put: bool = True,
    min_dte: int = 0,
    max_dte: Optional[int] = None,
    near_atm_pct: float = 0.025,
    max_premium: Optional[float] = None,
) -> dict:
    """
    Load synchronized stock + option chain data for the latest available snapshot.

    Uses explicit chain date selection so spot and option strikes align.
    """
    ticker_upper = ticker.upper()
    chain_date = get_most_recent_option_date(ticker_upper, base_dir=option_dir, verbose=False)
    stock_df = load_stock_data(ticker_upper, base_dir=stock_dir)
    option_df = load_option_chain_data(ticker_upper, date=chain_date, base_dir=option_dir)
    spot, latest_date = get_latest_stock_row(ticker_upper, base_dir=stock_dir)
    reference_date = _reference_date_from_chain(chain_date)
    contract = find_optimal_contract(
        option_df,
        spot,
        is_put=is_put,
        reference_date=reference_date,
        min_dte=min_dte,
        max_dte=max_dte,
        near_atm_pct=near_atm_pct,
        max_premium=max_premium,
    )
    if max_premium is not None:
        moneyness = abs(contract["strike"] - spot) / max(spot, 1.0)
        contract["moneyness_pct"] = round(moneyness * 100.0, 2)
        contract["entry_price"] = contract["ask"] if contract["ask"] > 0 else contract["mid"]
    else:
        moneyness = validate_contract_near_spot(
            contract,
            spot,
            near_atm_pct=near_atm_pct,
            ticker=ticker_upper,
        )
        contract["moneyness_pct"] = round(moneyness * 100.0, 2)
        contract["entry_price"] = contract["mid"] if contract["mid"] > 0 else contract["ask"]
    return {
        "ticker": ticker_upper,
        "spot": spot,
        "latest_date": latest_date,
        "chain_date": chain_date,
        "reference_date": reference_date,
        "stock_df": stock_df,
        "option_df": option_df,
        "contract": contract,
    }

