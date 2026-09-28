#!/usr/bin/env python3
"""Hidden Markov Model regime detection on a Yahoo Finance ticker.

Fits a Gaussian HMM to a configurable daily observation series, picks the number
of hidden states by BIC, and prints the learned regime structure to the console.
"""

import logging
import math
import sys
import textwrap
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

@dataclass(frozen=True)
class ObservationGuide:
    key: str
    menu: str
    on_select: str
    results: str
    state0: str
    markov_note: str


OBSERVATION_GUIDES: tuple[ObservationGuide, ...] = (
    ObservationGuide(
        "close_logret",
        "Daily log return from Close.",
        "The HMM clusters days by this ticker's return; eq columns are the same series.",
        "Regimes are return buckets on the ticker. obs mean/vol describe the signal; "
        "eq mean%/vol% show typical P&L in each state.",
        "state 0 = lowest average daily return (not always the quietest vol).",
        "Regime i = which Gaussian emission for daily returns applies; staying in i is a "
        "hidden-state label, not a forecast that tomorrow's return stays small.",
    ),
    ObservationGuide(
        "parkinson",
        "Range vol from High/Low (Parkinson).",
        "The HMM clusters by intraday range on this ticker; good for local index/single names.",
        "Regimes are wide-range vs narrow-range days for this symbol. "
        "eq stats show how the ticker tended to close-to-close in each range regime.",
        "state 0 = lowest average range-vol days.",
        "Regime i = range-vol bucket for this ticker; Markov part is only the hidden state, "
        "not a forecast of tomorrow's High/Low.",
    ),
    ObservationGuide(
        "abs_return",
        "Absolute daily return (size only, no sign).",
        "The HMM clusters quiet vs violent days for this ticker, ignoring up/down.",
        "Regimes are magnitude-of-move, not direction. eq mean% can still be positive or negative.",
        "state 0 = smallest typical |daily move|.",
        "Regime i = |return| bucket; large signed moves can still occur within a sticky regime label.",
    ),
    ObservationGuide(
        "delta_vix",
        f"Day-over-day change in {VIX_SYMBOL} (S&P 500 implied vol).",
        "The HMM fits US VIX shocks; your ticker is only the sidecar (eq columns).",
        f"Interpret as: when {VIX_SYMBOL} rose or fell, how did this ticker behave? "
        f"Not the ticker's own vol index unless it tracks US equities closely.",
        "state 0 = days when VIX tended to fall or drift down (calmest VIX-change bucket).",
        "Regime i = bucket for ^VIX daily change (obs); eq returns are not in the HMM state — "
        "only described in the sidecar columns.",
    ),
    ObservationGuide(
        "vrp",
        f"{VIX_SYMBOL} minus this ticker's 21d ann. realized vol.",
        "Mixes S&P implied vol (^VIX) with this ticker's realized vol — US-centric.",
        "High obs ≈ VIX rich vs the ticker's recent realized move; low obs ≈ realized hot vs VIX. "
        "Meaningful mainly for US names correlated with SPX; for e.g. ^HSI, prefer parkinson/abs_return.",
        "state 0 = lowest VIX-minus-realized (cheapest vs ticker's recent vol, in this formula).",
        "Regime i = bucket for the VRP formula (obs); Markov persistence is on that label only, "
        "not on ^VIX or the ticker's realized vol separately.",
    ),
)

MARKOV_HORIZONS = (1, 5, 10, 20)

OBSERVATION_MENU: tuple[tuple[str, str], ...] = tuple(
    (g.key, g.menu) for g in OBSERVATION_GUIDES
)

_GUIDE_BY_KEY = {g.key: g for g in OBSERVATION_GUIDES}

MODEL_SELECTION_NOTES: tuple[str, ...] = (
    "Each row is a separate fit with a different number of regimes (K) on the same history.",
    "states = K; params = model complexity; logLik = raw fit (rises with more states).",
    "AIC and BIC add a complexity penalty — lower is better. This run picks lowest BIC only.",
    "Sections 2–7 use the picked K. BIC often prefers 2 on ~5y daily data (parsimony, not proof).",
)

REGIME_PROFILE_NOTES: tuple[str, ...] = (
    "Each row = all days the decoder labeled that state (historical summary, not a forecast).",
    "% days = how often; avg run = typical streak length in trading days (persistence).",
    "obs mean / obs vol / ann obs vol = your selected observation recipe (banner line).",
    "eq mean% / eq vol% = this ticker's daily log returns on those same days (sidecar).",
    "State 0 is not 'good' or 'bad' — lowest average obs (recipe-specific line below).",
)

TRANSITION_MATRIX_NOTES: tuple[str, ...] = (
    "Row = regime today; column = regime tomorrow. Each row sums to ~100% (all next-day paths).",
    "Diagonal = stay in the same regime (persistence). Off-diagonal = switch to another regime.",
    "Estimated on full history for the picked K — typical in-sample dynamics, not a live forecast.",
)

MARKOV_REFERENCE_NOTES: tuple[str, ...] = (
    "HMM reference: P(still in regime i after h days) = (transition matrix)^h [i,i], "
    "assuming Markov hidden states and today's regime = i (from decoding the last day).",
    "Uses fitted transitions as if they hold forward — descriptive extrapolation, not validated forecast.",
    "Does not predict returns, VIX, or obs levels — only the discrete regime index. "
    "See banner 'interpret' for what obs/regime means for your recipe.",
)

WHERE_NOW_NOTES: tuple[str, ...] = (
    "current regime = decoded state on the last trading day in the window.",
    "obs mean / eq mean in parentheses = averages for that state over all history (see section 3),",
    "not just the current streak. days in regime = consecutive days in this state ending today.",
)

LAST_10_NOTES: tuple[str, ...] = (
    "Day-by-day view: obs = your recipe that day; eq ret% = ticker return; state = decoded regime.",
    "Use with section 6 to see recent switches vs long streaks. Labels are in-sample, descriptive.",
)

REGIME_STRIP_NOTES: tuple[str, ...] = (
    "One character = one trading day; digit = regime (0 = lowest obs mean, same as section 3).",
    "Long repeats = sticky regime; frequent digit changes = more switching (see section 2).",
    "Left = older, right = most recent — compare the tail to sections 4 and 5.",
)

SECTION_INTROS: tuple[tuple[str, str], ...] = (
    (
        "1) Model selection (lower AIC/BIC is better)",
        "How many hidden regimes best describe the past — without adding buckets that don't pay off?",
    ),
    (
        "2) Transition matrix (%, rows = from, cols = to)",
        "If today is regime i, where does the model usually place tomorrow? (stick vs switch.)",
    ),
    (
        "3) Regime profile",
        "When the decoder called each regime, what did the observation and ticker tend to look like?",
    ),
    (
        "4) Where is the market now",
        "Which regime label applies to the last day, and how long has this stretch lasted?",
    ),
    (
        "5) Last 10 trading days",
        "Recent calendar: each day's observation, ticker return, and assigned regime.",
    ),
    (
        "6) Regime strip, last 120 days (one char = one day)",
        "Quick timeline of regime labels — spot long calm runs vs choppy switching at a glance.",
    ),
    (
        "7) Overall summary",
        "Plain-English wrap-up: how many regimes, which one you're in, history mix, and stickiness.",
    ),
)

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _print_section(heading: str, intro: str) -> None:
    print(f"\n{heading}")
    print(f"  → {intro}\n")


def _print_notes(lines: tuple[str, ...]) -> None:
    print()
    for line in lines:
        print(f"  {line}")


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


def _observation_intro() -> None:
    print("  Observation — one number per day that the HMM clusters into regimes.")
    print("  State 0 is always the lowest average observation; eq mean%/vol% = your ticker.\n")


def _vix_proxy_caveat(symbol: str, observation: str) -> str | None:
    if observation not in ("delta_vix", "vrp"):
        return None
    if symbol.upper() in (VIX_SYMBOL, "^GSPC", "SPY", "IVV", "VOO"):
        return None
    return (
        f"  ^VIX is S&P 500 options vol — for {symbol!r}, read {observation} as a US fear "
        "signal applied to this ticker, not as its native vol regime."
    )


def _print_observation_followup(key: str, *, symbol: str | None = None) -> None:
    guide = _GUIDE_BY_KEY[key]
    print(f"  → {guide.on_select}")
    if symbol:
        caveat = _vix_proxy_caveat(symbol, key)
        if caveat:
            print(caveat)
    print()


def _print_observation_menu(default: str) -> None:
    _observation_intro()
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
        title = "Observation — ↑/↓ Enter to select, Esc = default"
        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            detail_row = len(OBSERVATION_MENU) + 3
            if height < detail_row + 2:
                raise curses.error("terminal too small")
            stdscr.addstr(0, 0, title[: max(width - 1, 0)])
            stdscr.addstr(1, 0, "HMM fits obs; eq columns = your ticker (state 0 = lowest obs mean)"[
                : max(width - 1, 0)
            ], curses.A_DIM)
            row = 3
            for i, (key, help_text) in enumerate(OBSERVATION_MENU):
                selected = i == idx
                attr = curses.A_REVERSE if selected else curses.A_NORMAL
                button = f" {key} "
                stdscr.addstr(row, 2, button[: width - 3], attr)
                hint = help_text[: max(width - len(button) - 5, 0)]
                stdscr.addstr(row, 2 + len(button), hint, curses.A_DIM if not selected else attr)
                row += 1
            detail = _GUIDE_BY_KEY[OBSERVATION_MENU[idx][0]].on_select
            stdscr.addstr(detail_row, 2, detail[: max(width - 4, 0)], curses.A_DIM)
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
        print()
        _observation_intro()
        picked = _pick_observation_curses(default)
        if picked is not None:
            print(f"  observation → {OBSERVATION_LABELS[picked]}")
            _print_observation_followup(picked)
            return picked

    _print_observation_menu(default)
    while True:
        raw = input(f"  Select 1–{len(OBSERVATION_MENU)} or recipe name [Enter = {default}]: ")
        choice = _resolve_observation_choice(raw, default)
        if choice is not None:
            print(f"  observation → {OBSERVATION_LABELS[choice]}")
            _print_observation_followup(choice)
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


def _print_paragraphs(paragraphs: tuple[str, ...]) -> None:
    for para in paragraphs:
        print(textwrap.fill(para, width=66, initial_indent="  ", subsequent_indent="  "))
        print()


def _markov_stay_prob(transmat: np.ndarray, state: int, horizon: int) -> float:
    if horizon <= 1:
        return float(transmat[state, state])
    return float(np.linalg.matrix_power(transmat, horizon)[state, state])


def _print_markov_reference(
    model: hmm.GaussianHMM,
    regime: int,
    guide: ObservationGuide,
) -> None:
    print("\n  2b) HMM Markov reference (discrete regime persistence)")
    print(
        "  → If hidden state today equals the decoded regime below, "
        "fitted one-day transitions imply:"
    )
    print("  Recipe (what regime i means here):")
    for line in textwrap.wrap(guide.markov_note, width=62):
        print(f"    {line}")
    rows = []
    for h in MARKOV_HORIZONS:
        p_stay = _markov_stay_prob(model.transmat_, regime, h)
        rows.append([str(h), f"{p_stay * 100:.1f}", f"{(1.0 - p_stay) * 100:.1f}"])
    print()
    print(
        table(
            [
                "days ahead (h)",
                f"P(still regime {regime})",
                f"P(not in {regime} at h)",
            ],
            rows,
            ">>>",
        )
    )
    print(f"\n  Decoded regime on last day (used as i above): {regime}")
    _print_notes(MARKOV_REFERENCE_NOTES)


def _overall_summary_paragraphs(
    symbol: str,
    K: int,
    config: Config,
    model: hmm.GaussianHMM,
    states: np.ndarray,
    X: np.ndarray,
    guide: ObservationGuide,
    lengths: np.ndarray,
) -> tuple[str, ...]:
    current = int(states[-1])
    streak = int(lengths[-1])
    persist = float(model.transmat_[current, current] * 100.0)

    parts: list[str] = []
    parts.append(
        f"BIC picked {K} regimes for {symbol} using {config.observation}. "
        "The story below is descriptive: it labels past days and summarizes "
        "patterns in-sample; it does not forecast tomorrow's return or regime."
    )

    state_bits: list[str] = []
    for s in range(K):
        mask = states == s
        share = mask.mean() * 100.0
        obs_mean = X[mask, 0].mean()
        obs_vol = X[mask, 0].std()
        state_bits.append(
            f"regime {s} about {share:.0f}% of days "
            f"(obs mean {obs_mean:+.2f}, daily obs vol {obs_vol:.2f}%)"
        )
    parts.append(
        "Historically, time split roughly as: " + "; ".join(state_bits) + ". "
        + guide.state0
    )

    share_current = (states == current).mean() * 100.0
    parts.append(
        f"As of the last trading day, the decoder assigns regime {current} "
        f"and has kept that label for {streak} consecutive days. "
        f"Regime {current} accounted for {share_current:.1f}% of all days in this window — "
        "section 5 lists the last 10 days; section 6 shows the last 120."
    )

    parts.append(
        f"From section 2: whenever the model was in regime {current}, "
        f"the next day stayed in {current} about {persist:.0f}% of the time "
        f"(row {current}, diagonal) — in-sample one-step stickiness on the observation "
        f"({config.observation})."
    )

    h5 = _markov_stay_prob(model.transmat_, current, 5) * 100.0
    h10 = _markov_stay_prob(model.transmat_, current, 10) * 100.0
    parts.append(
        f"Section 2b (Markov reference, same recipe): if hidden state today were {current}, "
        f"fitted transitions imply about {persist:.0f}% still in {current} after 1 day, "
        f"{h5:.0f}% after 5 days, {h10:.0f}% after 10 days — regime label only, "
        f"not a return/VIX forecast; assumes estimated transitions hold."
    )

    if K == 2:
        other = 1 - current
        other_persist = float(model.transmat_[other, other] * 100.0)
        parts.append(
            f"The other regime ({other}) was rarer or more episodic when active; "
            f"its own diagonal persistence is about {other_persist:.0f}%. "
            "Large single-day moves can still occur inside a sticky regime — "
            "see section 5."
        )

    return tuple(parts)


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
    current_regime = int(states[-1])
    obs_label = OBSERVATION_LABELS[config.observation]

    print()
    print("=" * 68)
    print(f"  {symbol}  |  HMM regime model")
    print("=" * 68)
    print(f"  history      {dates.iloc[0]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d}")
    print(f"  observations {len(X)} daily points, after {config.period} of history")
    print(f"  observation  {obs_label}")
    print(f"  model        GaussianHMM, diag covariance, selected {K} states by BIC")
    guide = _GUIDE_BY_KEY[config.observation]
    print(f"  interpret    {guide.results}")
    caveat = _vix_proxy_caveat(symbol, config.observation)
    if caveat:
        print(caveat)
    print(
        "  readout      Descriptive in-sample regimes on your observation — not a forward forecast."
    )

    _print_section(*SECTION_INTROS[0])
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
    _print_notes(MODEL_SELECTION_NOTES)

    _print_section(*SECTION_INTROS[1])
    print(table(
        ["from \\ to", *[str(s) for s in range(K)]],
        [[str(i), *[f"{v * 100:.1f}" for v in row]] for i, row in enumerate(model.transmat_)],
        ">" + "<" * K,
    ))
    _print_notes(TRANSITION_MATRIX_NOTES)
    _print_markov_reference(model, current_regime, guide)

    _print_section(*SECTION_INTROS[2])
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
    _print_notes(REGIME_PROFILE_NOTES)
    print(f"  Recipe-specific: {guide.state0}")

    _print_section(*SECTION_INTROS[3])
    last = current_regime
    tail_obs = X[states == last, 0]
    tail_eq = equity[states == last, 0]
    print(
        f"  current regime  {last}  "
        f"(obs mean {tail_obs.mean():+.3f}, eq mean {tail_eq.mean():+.3f}%)"
    )
    print(f"  days in regime  {int(lengths[-1])}")
    _print_notes(WHERE_NOW_NOTES)

    _print_section(*SECTION_INTROS[4])
    print(table(
        ["date", "obs", "eq ret%", "state"],
        [
            [f"{d:%Y-%m-%d}", f"{o:+.3f}", f"{e:+.3f}", str(s)]
            for d, o, e, s in list(zip(dates, X[:, 0], equity[:, 0], states))[-10:]
        ],
        "><><",
    ))
    _print_notes(LAST_10_NOTES)

    _print_section(*SECTION_INTROS[5])
    strip = "".join(str(s) for s in states[-120:])
    print(f"  {strip}")
    print(f"  {dates.iloc[-120]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d}")
    _print_notes(REGIME_STRIP_NOTES)

    _print_section(*SECTION_INTROS[6])
    _print_paragraphs(
        _overall_summary_paragraphs(
            symbol, K, config, model, states, X, guide, lengths
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
