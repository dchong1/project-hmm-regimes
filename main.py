#!/usr/bin/env python3
"""Hidden Markov Model regime detection on a Yahoo Finance ticker.

Fits a Gaussian HMM to daily log returns, picks the number of hidden states by
BIC, and prints the learned regime structure to the console.
"""

import logging
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm

PERIOD = "5y"
MIN_OBS = 200
MAX_STATES = 5
TRADING_DAYS = 252
MAX_TRIES = 3

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def prompt_ticker() -> str | None:
    while True:
        raw = input("Ticker (e.g. AAPL, ^GSPC, BTC-USD, or 'q' to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if raw:
            return raw.upper()
        print("  please enter a symbol.")


def fetch_returns(symbol: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (dates, percent daily log returns) or raise ValueError."""
    data = yf.Ticker(symbol).history(period=PERIOD, interval="1d", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if "Close" not in data.columns or data["Close"].dropna().empty:
        raise ValueError(f"no price data returned for {symbol!r} (bad or delisted symbol?)")

    close = data["Close"].dropna()
    returns = np.log(close).diff().dropna() * 100
    if len(returns) < MIN_OBS:
        raise ValueError(f"only {len(returns)} usable rows for {symbol!r}, need {MIN_OBS}")
    return pd.Series(returns.index, index=returns.index), returns.to_numpy().reshape(-1, 1)


def fit_states(X: np.ndarray) -> tuple[hmm.GaussianHMM, list[dict]]:
    top_k = min(MAX_STATES, len(X) // 20)
    candidates = []
    for k in range(2, top_k + 1):
        model = hmm.GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=500,
            tol=1e-4,
            random_state=0,
        ).fit(X)
        candidates.append({
            "k": k,
            "n_params": sum(model._get_n_fit_scalars_per_param().values()),
            "loglik": model.score(X),
            "aic": model.aic(X),
            "bic": model.bic(X),
            "model": model,
            "converged": model.monitor_.converged,
        })
    return min(candidates, key=lambda c: c["bic"]), candidates


def relabel_by_volatility(model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Renumber states 0..K-1 by ascending volatility so state 0 is calmest.

    Also permutes the model's own parameter arrays to match, so the printed
    transition matrix and start probabilities stay consistent with the labels.
    """
    raw = model.predict(X)
    vols = [X[raw == s, 0].std() for s in range(raw.max() + 1)]
    order = np.argsort(vols)  # order[new_label] = old_label
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    model.startprob_ = model.startprob_[order]
    model.means_ = model.means_[order]
    return order[raw]


def runs(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (state_of_each_run, length_of_each_run)."""
    if len(states) == 0:
        return np.array([]), np.array([])
    breaks = np.flatnonzero(np.diff(states)) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(states)]])
    return states[starts], ends - starts


def table(headers: list[str], rows: list[list[str]], aligns: str) -> str:
    widths = [max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) for i in range(len(headers))]

    def fmt(cells: list[str]) -> str:
        parts = []
        for cell, width, align in zip(cells, widths, aligns):
            parts.append(cell.rjust(width) if align == ">" else cell.ljust(width))
        return "  ".join(parts).rstrip()

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([fmt(headers), sep, *(fmt(r) for r in rows)])


def load() -> tuple[str, pd.Series, np.ndarray] | None:
    """Resolve a ticker (argv or prompt, with retries) and fetch its returns."""
    if len(sys.argv) > 1:
        candidates = [sys.argv[1].strip().upper()]
    else:
        candidates = None

    for attempt in range(MAX_TRIES):
        symbol = candidates[0] if candidates else prompt_ticker()
        if symbol is None:
            return None
        try:
            dates, X = fetch_returns(symbol)
        except ValueError as exc:
            print(f"  {exc}")
            if candidates:
                return None
            print(f"  ({MAX_TRIES - attempt - 1} tries left)")
            continue
        return symbol, dates, X
    return None


def main() -> int:
    loaded = load()
    if loaded is None:
        return 1
    symbol, dates, X = loaded

    best, candidates = fit_states(X)
    model = best["model"]
    states = relabel_by_volatility(model, X)
    K = best["k"]

    print()
    print("=" * 68)
    print(f"  {symbol}  |  HMM regime model")
    print("=" * 68)
    print(f"  history      {dates.iloc[0]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d}")
    print(f"  observations {len(X)} daily log returns (%), after {PERIOD} of history")
    print(f"  model        GaussianHMM, diag covariance, selected {K} states by BIC")

    print("\n1) Model selection (lower AIC/BIC is better)\n")
    print(table(
        ["states", "params", "logLik", "AIC", "BIC", "converged"],
        [
            [
                f"{c['k']}{'  <-- picked' if c is best else ''}",
                str(c["n_params"]),
                f"{c['loglik']:.1f}",
                f"{c['aic']:.1f}",
                f"{c['bic']:.1f}",
                "yes" if c["converged"] else "no",
            ]
            for c in candidates
        ],
        ">>>>><",
    ))

    print("\n2) Transition matrix (%, rows = from, cols = to)\n")
    print(table(
        ["from \\ to", *[str(s) for s in range(K)]],
        [[str(i), *[f"{v * 100:.1f}" for v in row]] for i, row in enumerate(model.transmat_)],
        ">" + "<" * K,
    ))
    print("\n  diagonal = persistence: how likely a regime is to repeat tomorrow")

    print("\n3) Regime profile\n")
    run_states, lengths = runs(states)
    rows = []
    for s in range(K):
        mask = states == s
        rets = X[mask, 0]
        share = mask.mean() * 100
        vol = rets.std()
        avg_run = lengths[run_states == s].mean() if mask.any() else 0.0
        rows.append([
            str(s),
            f"{share:.1f}",
            f"{model.startprob_[s] * 100:.1f}",
            f"{rets.mean():+.3f}",
            f"{vol:.3f}",
            f"{vol * np.sqrt(TRADING_DAYS):.1f}",
            f"{avg_run:.0f}",
        ])
    print(table(
        ["state", "% days", "start%", "mean ret%", "vol%", "ann vol%", "avg run"],
        rows,
        "><>>><><",
    ))
    print("\n  states are numbered by ascending volatility: state 0 = calmest")

    print("\n4) Where is the market now\n")
    last = states[-1]
    tail = X[states == last, 0]
    print(f"  current regime  {last}  (vol {tail.std():.3f}%/day, mean {tail.mean():+.3f}%)")
    print(f"  days in regime  {int(lengths[-1])}")

    print("\n5) Last 10 trading days\n")
    print(table(
        ["date", "return%", "state"],
        [[f"{d:%Y-%m-%d}", f"{r:+.3f}", str(s)] for d, r, s in list(zip(dates, X[:, 0], states))[-10:]],
        "><>",
    ))

    print("\n6) Regime strip, last 120 days (one char = one day)\n")
    strip = "".join(str(s) for s in states[-120:])
    print(f"  {strip}")
    print(f"  {dates.iloc[-120]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
