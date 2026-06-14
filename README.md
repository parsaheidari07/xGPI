# xG Power Index (xGPI)

A football analytics tool that estimates team strength using
Elo-weighted expected goals (xG).

## How It Works

Each of the last 5 matches is assigned a weight based on the
opponent's Elo rating relative to the team's Elo:

$$w_i = \frac{1}{1 + e^{-(R_{opp} - R_{team}) / 400}}$$

Stronger opponents contribute more to the final score.

**Attacking xG** — weighted average of the team's own xG across 5 matches  
**Defensive xG** — weighted average of opponents' xG against the team  
**Team Strength** — Attacking xG × Defensive xG

## Installation

pip install streamlit

## Run

streamlit run app.py

## Usage

1. Enter your team's Elo rating.
2. Fill in the last 5 matches:
   - Opponent Elo
   - Your xG
   - Opponent xG
3. Click **Calculate Team Strength**.

## Requirements

- Python 3.8+
- Streamlit

## License

MIT
