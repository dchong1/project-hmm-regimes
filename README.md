# project-hmm-regimes

Fits a [hmmlearn](https://github.com/hmmlearn/hmmlearn) Hidden Markov Model to a
Yahoo Finance ticker and prints the regime structure it finds.

You choose what the HMM observes (daily log returns, range vol, VIX change, VRP,
and so on). States are relabeled by **ascending mean of that observation** (state 0
is the lowest / calmest for vol-style recipes). The same run also tracks **equity
log returns** as a sidecar so regime profiles and the last-10 table show what the
stock did inside each state. The number of states is chosen by BIC.

## Observation recipes

At the interactive prompt, pick one of:

| Name | What `X` is |
|------|-------------|
| `close_logret` | `100 × ln(C_t / C_{t-1})` (default) |
| `parkinson` | Parkinson daily range vol from High/Low, in % |
| `abs_return` | Absolute value of `close_logret` |
| `delta_vix` | Day-over-day change in `^VIX` Close, aligned to the ticker calendar |
| `vrp` | Variance risk premium: `VIX_t − √(trading_days) × σ_21d(equity returns)` |

For `vrp`, equity returns are the same daily log returns in %; the 21-day rolling
std is annualized with your **trading days / year** setting (default 252). Rows
without a full rolling window are dropped. `^VIX` is downloaded when the ticker
is not already VIX.

Example: try NVDA with `vrp`, then again with `close_logret`, and compare the
regime strips.

## Setup

```bash
cd ~/projects/project-hmm-regimes
python3.12 -m venv .venv          # skip if .venv already exists
./.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python main.py                    # prompts for a ticker
python main.py AAPL               # or pass it as an argument
python main.py ^GSPC
```

Without activating:

```bash
./.venv/bin/python main.py AAPL
```

## Output

1. Model selection table for 2–5 states, scored by AIC and BIC
2. Transition matrix (persistence on the diagonal)
3. Per-regime profile: share of days, observation mean/vol, **equity mean% / eq vol%**, average run length
4. Which regime the market is in right now
5. Last 10 trading days: date, observation value, equity return %, state
6. A 120-day regime strip, one character per day

## Notes

- Data is split/dividend adjusted (`auto_adjust=True`); default history is 5 years of daily bars.
- State numbers follow ascending **observation mean**, not volatility of returns alone.
- BIC is strict on daily data: with ~1250 observations its penalty for extra states
  often outweighs the fit gain, so it usually picks 2 states. The comparison table
  shows the criterion working, not a failure. Switching to `model.aic(X)` in
  `fit_states()` selects richer regimes instead.
- Labels are arbitrary up to permutation; compare regimes by their statistics across runs.
- The model is descriptive, not predictive.
