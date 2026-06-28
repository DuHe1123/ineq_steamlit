# Global Inequality Explorer

Streamlit app for exploring the `ADM0_ALL.csv` country-year inequality panel.

The app focuses on:

- Overview and data coverage
- Distribution and descriptive statistics
- Within-country and between-country variation
- Time trends
- Relationships with GDP, population, and selected development indicators
- Dynamics and persistence

The primary measure is `GINIW_gdppc`, with alternative measures:

- `GE_m1W_gdppc`
- `GE_0W_gdppc`
- `GE_1W_gdppc`
- `GE_2W_gdppc`
- `COVW_gdppc`

## Local Run

From the project folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects the data file here:

```text
data/ADM0_ALL.csv
```

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy.

The data is loaded from the repository itself, so no Streamlit secrets are required for the current version.

## Notes

The app attempts to import `expdpy` and uses its exploration helpers where available. The main charts also include pandas and Plotly fallbacks so the public app remains usable if an experimental helper changes.
