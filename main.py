#!/usr/bin/env python3
"""Hidden Markov Model regime detection on a Yahoo Finance ticker.

Fits a Gaussian HMM to a configurable daily observation series, picks the number
of hidden states by BIC, and prints the learned regime structure to the console.
"""

import logging
import math
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm

YF_PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
OBSERVATIONS = ("close_logret", "parkinson", "abs_return", "delta_vix", "vrp")
VIX_SYMBOL = "^VIX"
RV_WINDOW = 21

OBSERVATION_LABELS = {
    "close_logret": "close_logret (daily log return %)",
    "parkinson": "parkinson (daily range vol %)",
    "abs_return": "abs_return (|daily log return| %)",
    "delta_vix": "delta_vix (daily change in VIX)",
    "vrp": "vrp (VIX minus 21d ann realized vol %)",
}

# (recipe key, one-line help for the selection menu)
OBSERVATION_MENU: tuple[tuple[str, str], ...] = (
    ("close_logret", "Daily log return from Close — same as the original model."),
    ("parkinson", "Range vol from High/Low — stress from the day's trading range."),
    ("abs_return", "Absolute daily return — magnitude only, no up/down sign."),
    ("delta_vix", "Day-over-day change in ^VIX — index vol shock, stock as sidecar."),
    ("vrp", "VIX minus 21d ann. realized vol — rich vs cheap implied vol."),
)

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


@dataclass(frozen=True)
class Config:
    period: str = "5y"
    min_obs: int = 200
    max_states: int = 5
    trading_days: int = 252
    max_tries: int = 3
    observation: str = "close_logret"


def _prompt(label: str, default: str) -> str:
    raw = input(f"  {label} [{default}]: ").strip()
    return raw if raw else default


def _resolve_observation_choice(raw: str, default: str) -> str | None:
    text = raw.strip().lower()
    if not text:
        return default
    if text.isdigit():
        n = int(text)
        if 1 <= n <= len(OBSERVATION_MENU):
            return OBSERVATION_MENU[n - 1][0]
        return None
    if text in OBSERVATIONS:
        return text
    return None


def _print_observation_menu(default: str) -> None:
    print("  Observation — what the HMM fits each day:\n")
    default_idx = next(i for i, (k, _) in enumerate(OBSERVATION_MENU) if k == default)
    for i, (key, help_text) in enumerate(OBSERVATION_MENU, start=1):
        mark = " (default)" if i - 1 == default_idx else ""
        print(f"    [{i}]  {key}{mark}")
        print(f"         {help_text}")
    print()


def _pick_observation_curses(default: str) -> str | None:
    import curses

    default_idx = next(i for i, (k, _) in enumerate(OBSERVATION_MENU) if k == default)
    idx = default_idx

    def _draw(stdscr: "curses._CursesWindow") -> str:
        nonlocal idx
        curses.curs_set(0)
        stdscr.keypad(True)
        title = "Observation — ↑/↓ to highlight, Enter to select, Esc for default"
        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            if height < len(OBSERVATION_MENU) + 4:
                raise curses.error("terminal too small")
            stdscr.addstr(0, 0, title[: max(width - 1, 0)])
            row = 2
            for i, (key, help_text) in enumerate(OBSERVATION_MENU):
                selected = i == idx
                attr = curses.A_REVERSE if selected else curses.A_NORMAL
                button = f" {key} "
                stdscr.addstr(row, 2, button[: width - 3], attr)
                hint = help_text[: max(width - len(button) - 5, 0)]
                stdscr.addstr(row, 2 + len(button), hint, curses.A_DIM if not selected else attr)
                row += 1
            stdscr.refresh()
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(OBSERVATION_MENU)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(OBSERVATION_MENU)
            elif key in (curses.KEY_ENTER, 10, 13):
                return OBSERVATION_MENU[idx][0]
            elif key in (27, ord("q")):  # Esc / q → default
                return default
            elif ord("1") <= key <= ord("9"):
                n = key - ord("0")
                if 1 <= n <= len(OBSERVATION_MENU):
                    return OBSERVATION_MENU[n - 1][0]

    try:
        return curses.wrapper(_draw)
    except curses.error:
        return None


def prompt_observation(default: str = "close_logret") -> str:
    if sys.stdin.isatty() and sys.stdout.isatty():
        picked = _pick_observation_curses(default)
        if picked is not None:
            print(f"  observation → {OBSERVATION_LABELS[picked]}\n")
            return picked

    _print_observation_menu(default)
    while True:
        raw = input(f"  Select 1–{len(OBSERVATION_MENU)} or recipe name [Enter = {default}]: ")
        choice = _resolve_observation_choice(raw, default)
        if choice is not None:
            return choice
        print(f"  pick 1–{len(OBSERVATION_MENU)}, a recipe name, or Enter for the default.")


def prompt_int(label: str, default: int, *, min_value: int) -> int:
    while True:
        raw = _prompt(label, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("  please enter a whole number.")
            continue
        if value < min_value:
            print(f"  please enter a number >= {min_value}.")
            continue
        return value


def prompt_config() -> Config:
    defaults = Config()
    print("Settings (Enter keeps the default)\n")
    while True:
        period = _prompt("history period", defaults.period).lower()
        if period in YF_PERIODS:
            break
        print(f"  use one of: {', '.join(YF_PERIODS)}")
    observation = prompt_observation(defaults.observation)
    return Config(
        period=period,
        min_obs=prompt_int("min observations", defaults.min_obs, min_value=40),
        max_states=prompt_int("max states", defaults.max_states, min_value=2),
        trading_days=prompt_int("trading days / year", defaults.trading_days, min_value=1),
        max_tries=prompt_int("ticker retries", defaults.max_tries, min_value=1),
        observation=observation,
    )


def prompt_ticker() -> str | None:
    while True:
        raw = input("Ticker (e.g. AAPL, ^GSPC, BTC-USD, or 'q' to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if raw:
            return raw.upper()
        print("  please enter a symbol.")


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    return data


def _calendar_index(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Strip timezone and normalize to calendar dates so joins work across tickers."""
    out = obj.copy()
    idx = out.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        out.index = idx.normalize()
    return out


def _download_history(symbol: str, config: Config) -> pd.DataFrame:
    data = yf.Ticker(symbol).history(period=config.period, interval="1d", auto_adjust=True)
    data = _flatten_columns(data)
    if "Close" not in data.columns or data["Close"].dropna().empty:
        raise ValueError(f"no price data returned for {symbol!r} (bad or delisted symbol?)")
    return data


def _vix_close_series(config: Config) -> pd.Series:
    data = _calendar_index(_download_history(VIX_SYMBOL, config))
    return data["Close"].dropna().rename("vix")


def _parkinson_pct(high: pd.Series, low: pd.Series) -> pd.Series:
    log_hl = np.log(high / low)
    daily = np.sqrt(log_hl.pow(2) / (4.0 * math.log(2.0)))
    return daily * 100.0


def fetch_observations(
    symbol: str, config: Config
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Return (dates, observation matrix, equity log returns %) aligned on the same days."""
    if config.observation not in OBSERVATIONS:
        raise ValueError(f"unknown observation {config.observation!r}")

    data = _calendar_index(_download_history(symbol, config))
    close = data["Close"].dropna()
    equity = (np.log(close).diff() * 100).rename("equity")

    if config.observation == "close_logret":
        obs = equity
    elif config.observation == "abs_return":
        obs = equity.abs().rename("obs")
    elif config.observation == "parkinson":
        if "High" not in data.columns or "Low" not in data.columns:
            raise ValueError(f"need High/Low for parkinson on {symbol!r}")
        hl = data[["High", "Low"]].dropna(how="any")
        park = _parkinson_pct(hl["High"], hl["Low"]).rename("obs")
        obs = park.reindex(equity.index)
    elif config.observation in ("delta_vix", "vrp"):
        if symbol in (VIX_SYMBOL, "VIX"):
            vix = close.rename("vix")
        else:
            vix = _vix_close_series(config)
            if vix.empty:
                raise ValueError(f"no VIX data from {VIX_SYMBOL!r} (needed for {config.observation})")
        frame = pd.concat([equity, vix], axis=1, join="inner").dropna()
        if frame.empty:
            raise ValueError(f"no overlapping dates between {symbol!r} and {VIX_SYMBOL!r}")
        if config.observation == "delta_vix":
            obs = frame["vix"].diff().rename("obs")
        else:
            ann_rv = (
                frame["equity"].rolling(RV_WINDOW).std()
                * math.sqrt(config.trading_days)
            )
            obs = (frame["vix"] - ann_rv).rename("obs")
        equity = frame["equity"]
    else:
        raise ValueError(f"unknown observation {config.observation!r}")

    aligned = pd.concat([obs.rename("obs"), equity.rename("equity")], axis=1).dropna()
    if len(aligned) < config.min_obs:
        raise ValueError(
            f"only {len(aligned)} usable rows for {symbol!r}, need {config.min_obs}"
        )

    dates = pd.Series(aligned.index, index=aligned.index)
    X = aligned["obs"].to_numpy().reshape(-1, 1)
    equity_col = aligned["equity"].to_numpy().reshape(-1, 1)
    return dates, X, equity_col


def fit_states(X: np.ndarray, config: Config) -> tuple[hmm.GaussianHMM, list[dict]]:
    top_k = min(config.max_states, len(X) // 20)
    if top_k < 2:
        raise ValueError(f"need at least 40 observations to fit 2 states, have {len(X)}")
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


def relabel_by_level(model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Renumber states 0..K-1 by ascending mean observation (state 0 = lowest / calmest)."""
    raw = model.predict(X)
    means = [X[raw == s, 0].mean() for s in range(raw.max() + 1)]
    order = np.argsort(means)
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    model.startprob_ = model.startprob_[order]
    model.means_ = model.means_[order]
    if hasattr(model, "covars_") and model.covars_ is not None:
        k = len(order)
        model.covars_ = np.take(model.covars_, order, axis=0).reshape(k, X.shape[1])
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


def load(config: Config) -> tuple[str, pd.Series, np.ndarray, np.ndarray] | None:
    """Resolve a ticker (argv or prompt, with retries) and fetch observations."""
    if len(sys.argv) > 1:
        candidates = [sys.argv[1].strip().upper()]
    else:
        candidates = None

    for attempt in range(config.max_tries):
        symbol = candidates[0] if candidates else prompt_ticker()
        if symbol is None:
            return None
        try:
            dates, X, equity = fetch_observations(symbol, config)
        except ValueError as exc:
            print(f"  {exc}")
            if candidates:
                return None
            print(f"  ({config.max_tries - attempt - 1} tries left)")
            continue
        return symbol, dates, X, equity
    return None


def main() -> int:
    config = prompt_config()
    loaded = load(config)
    if loaded is None:
        return 1
    symbol, dates, X, equity = loaded

    best, candidates = fit_states(X, config)
    model = best["model"]
    states = relabel_by_level(model, X)
    K = best["k"]
    obs_label = OBSERVATION_LABELS[config.observation]

    print()
    print("=" * 68)
    print(f"  {symbol}  |  HMM regime model")
    print("=" * 68)
    print(f"  history      {dates.iloc[0]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d}")
    print(f"  observations {len(X)} daily points, after {config.period} of history")
    print(f"  observation  {obs_label}")
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
        obs_vals = X[mask, 0]
        eq_vals = equity[mask, 0]
        share = mask.mean() * 100
        obs_vol = obs_vals.std()
        eq_vol = eq_vals.std()
        avg_run = lengths[run_states == s].mean() if mask.any() else 0.0
        rows.append([
            str(s),
            f"{share:.1f}",
            f"{model.startprob_[s] * 100:.1f}",
            f"{obs_vals.mean():+.3f}",
            f"{obs_vol:.3f}",
            f"{obs_vol * np.sqrt(config.trading_days):.1f}",
            f"{eq_vals.mean():+.3f}",
            f"{eq_vol:.3f}",
            f"{avg_run:.0f}",
        ])
    print(table(
        [
            "state",
            "% days",
            "start%",
            "obs mean",
            "obs vol",
            "ann obs vol",
            "eq mean%",
            "eq vol%",
            "avg run",
        ],
        rows,
        "><>>><>><><",
    ))
    print("\n  states are numbered by ascending observation mean: state 0 = lowest")

    print("\n4) Where is the market now\n")
    last = states[-1]
    tail_obs = X[states == last, 0]
    tail_eq = equity[states == last, 0]
    print(
        f"  current regime  {last}  "
        f"(obs mean {tail_obs.mean():+.3f}, eq mean {tail_eq.mean():+.3f}%)"
    )
    print(f"  days in regime  {int(lengths[-1])}")

    print("\n5) Last 10 trading days\n")
    print(table(
        ["date", "obs", "eq ret%", "state"],
        [
            [f"{d:%Y-%m-%d}", f"{o:+.3f}", f"{e:+.3f}", str(s)]
            for d, o, e, s in list(zip(dates, X[:, 0], equity[:, 0], states))[-10:]
        ],
        "><><",
    ))

    print("\n6) Regime strip, last 120 days (one char = one day)\n")
    strip = "".join(str(s) for s in states[-120:])
    print(f"  {strip}")
    print(f"  {dates.iloc[-120]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
