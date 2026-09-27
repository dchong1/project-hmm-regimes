# project-hmm-regimes

Fits a [hmmlearn](https://github.com/hmmlearn/hmmlearn) Hidden Markov Model to a
Yahoo Finance ticker and prints the regime structure it finds.

The model reads daily log returns, not price levels. Each hidden state comes out
as a volatility regime: state 0 is the calmest, higher states are progressively
turbulent. The number of states is chosen by BIC.

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

1. Model selection table for 2-5 states, scored by AIC and BIC
2. Transition matrix (persistence on the diagonal)
3. Per-regime profile: share of days, mean return, daily and annualised vol, average run length
4. Which regime the market is in right now
5. Last 10 trading days with decoded states
6. A 120-day regime strip, one character per day

## Notes

- Data is split/dividend adjusted (`auto_adjust=True`), 5 years of daily bars.
- State numbers are assigned by ascending volatility, so state 0 is always calmest.
- BIC is strict on daily data: with ~1250 observations its penalty for extra states
  outweighs the fit gain, so it usually picks 2 states (calm vs. turbulent). The
  comparison table shows this is the criterion working, not a failure. Switching to
  `model.aic(X)` in `fit_states()` selects 3-4 richer regimes instead.
- Labels are arbitrary up to permutation, so compare regimes by their statistics,
  not by number, across runs.
- The model is descriptive, not predictive. It clusters historical volatility; it
  does not forecast the next state.
