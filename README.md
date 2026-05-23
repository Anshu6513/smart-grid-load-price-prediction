# Smart Grid AI — Load & Price Forecasting

AI-based system for predicting electricity consumption and market 
prices to enable intelligent load distribution in smart grids.

## Problem Statement
Grid operators currently over or under-procure power due to 
inaccurate demand forecasting, leading to load shedding and 
financial losses. This system predicts 24-hour ahead load and 
price to enable proactive, optimized distribution.

## Project Structure
smart-grid-ai/
├── data/               # Raw and processed datasets (not committed)
├── notebooks/          # EDA and experimentation
├── pipeline/           # Data fetching, cleaning, feature engineering
├── models/             # Trained model artifacts
├── dashboard/          # Web app for demo
└── docs/               # Reports and documentation

## Team
| Role | Name |
|------|------|
| Data Engineering | Your Name |
| Load Forecasting | Person 2 |
| Price Forecasting | Person 3 |
| Dashboard | Person 4 |
| Research & Integration | Person 5 |

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Current Status
- [x] Repository setup
- [ ] Data collection
- [ ] Preprocessing
- [ ] Model development
- [ ] Dashboard
