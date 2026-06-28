from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import expdpy as ex
except Exception:  # pragma: no cover - the app has pandas/plotly fallbacks.
    ex = None


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "ADM0_ALL.csv"

ENTITY_COL = "iso3"
ENTITY_NAME_COL = "country"
TIME_COL = "year"
WEIGHT_COL = "pop"

INEQUALITY_LABELS = {
    "GINIW_gdppc": "Population-weighted Gini",
    "GE_m1W_gdppc": "GE alpha -1",
    "GE_0W_gdppc": "GE alpha 0",
    "GE_1W_gdppc": "GE alpha 1",
    "GE_2W_gdppc": "GE alpha 2",
    "COVW_gdppc": "Coefficient of variation",
}

CORE_COLUMNS = [
    ENTITY_COL,
    ENTITY_NAME_COL,
    TIME_COL,
    "gdp_per_capita",
    "gdp_total",
    WEIGHT_COL,
    *INEQUALITY_LABELS.keys(),
]

DEFAULT_RELATION_COLUMNS = [
    "gdp_per_capita",
    "pop",
    "gdp_total",
    "SP_URB_TOTL_IN_ZS",
    "SP_DYN_LE00_IN",
    "SE_ADT_LITR_ZS",
    "IT_NET_USER_ZS",
    "NE_TRD_GNFS_ZS",
    "VA_EST",
    "RQ_EST",
]


@dataclass(frozen=True)
class AppState:
    df: pd.DataFrame
    metric: str
    years: tuple[int, int]
    countries: list[str]
    treat_zero_as_missing: bool


def page_config() -> None:
    st.set_page_config(
        page_title="Global Inequality Explorer",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;}
        [data-testid="stMetricValue"] {font-size: 1.35rem;}
        div[data-testid="stExpander"] div[role="button"] p {font-size: 0.95rem;}
        .small-note {color: #5b6472; font-size: 0.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for col in df.columns:
        if col not in {ENTITY_COL, ENTITY_NAME_COL}:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").astype("Int64")
    df = df.dropna(subset=[ENTITY_COL, ENTITY_NAME_COL, TIME_COL]).copy()
    df[TIME_COL] = df[TIME_COL].astype(int)

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    if ex is not None and hasattr(ex, "set_panel"):
        try:
            df = ex.set_panel(df, entity=ENTITY_COL, time=TIME_COL)
        except Exception:
            pass
    return df


def metric_label(metric: str) -> str:
    return INEQUALITY_LABELS.get(metric, metric)


def available_columns(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def numeric_options(df: pd.DataFrame) -> list[str]:
    keep = []
    for col in df.select_dtypes(include=np.number).columns:
        if col != TIME_COL and df[col].notna().sum() > 10:
            keep.append(col)
    return keep


def filtered_data(raw: pd.DataFrame, state: AppState) -> pd.DataFrame:
    df = raw.loc[
        raw[TIME_COL].between(state.years[0], state.years[1])
        & raw[ENTITY_NAME_COL].isin(state.countries)
    ].copy()
    if state.treat_zero_as_missing:
        cols = available_columns(df, INEQUALITY_LABELS.keys())
        df[cols] = df[cols].replace(0, np.nan)
    return df


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna() & (weights > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(values[ok], weights=weights[ok]))


def add_download(df: pd.DataFrame, label: str, file_name: str) -> None:
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def safe_expdpy_figure(func_name: str, *args, **kwargs):
    if ex is None or not hasattr(ex, func_name):
        return None
    try:
        result = getattr(ex, func_name)(*args, **kwargs)
        return getattr(result, "fig", None)
    except Exception:
        return None


def sidebar(raw: pd.DataFrame) -> tuple[str, AppState]:
    st.sidebar.title("Inequality Explorer")
    st.sidebar.caption("ADM0 country-year panel")

    section = st.sidebar.radio(
        "Section",
        [
            "Overview",
            "Descriptive statistics",
            "Within and between",
            "Trends",
            "Relationships",
            "Dynamics",
        ],
    )

    st.sidebar.divider()
    metric = st.sidebar.selectbox(
        "Inequality measure",
        available_columns(raw, INEQUALITY_LABELS.keys()),
        format_func=metric_label,
    )

    min_year, max_year = int(raw[TIME_COL].min()), int(raw[TIME_COL].max())
    years = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))

    countries_all = sorted(raw[ENTITY_NAME_COL].dropna().unique().tolist())
    default_countries = countries_all
    countries = st.sidebar.multiselect(
        "Countries",
        options=countries_all,
        default=default_countries,
        help="Clear this field to temporarily hide all countries, or select a subset for focused analysis.",
    )
    if not countries:
        countries = countries_all

    treat_zero = st.sidebar.checkbox(
        "Treat zero inequality values as missing",
        value=False,
        help="Some countries report zeros for all inequality measures in early years. Turn this on for sensitivity checks.",
    )

    st.sidebar.divider()
    st.sidebar.caption("Package status")
    if ex is None:
        st.sidebar.warning("expdpy is not loaded; pandas/plotly fallbacks are active.")
    else:
        st.sidebar.success("expdpy loaded")

    return section, AppState(raw, metric, years, countries, treat_zero)


def render_header(df: pd.DataFrame, state: AppState) -> None:
    st.title("Global Inequality Explorer")
    st.caption(
        "Country-year exploration of population-weighted inequality measures from ADM0_ALL.csv."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Countries", f"{df[ENTITY_COL].nunique():,}")
    c3.metric("Years", f"{df[TIME_COL].min()}-{df[TIME_COL].max()}")
    c4.metric("Measure", metric_label(state.metric))


def overview_page(df: pd.DataFrame, state: AppState) -> None:
    render_header(df, state)

    with st.expander("Preview analysis sample"):
        st.dataframe(df[available_columns(df, CORE_COLUMNS)].head(200), use_container_width=True)
        add_download(df, "Download filtered sample", "adm0_filtered_sample.csv")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Panel coverage")
        coverage = (
            df.groupby(TIME_COL, as_index=False)
            .agg(countries=(ENTITY_COL, "nunique"), observations=(ENTITY_COL, "size"))
            .sort_values(TIME_COL)
        )
        fig = px.area(
            coverage,
            x=TIME_COL,
            y="countries",
            markers=True,
            labels={"countries": "Countries with observations", TIME_COL: "Year"},
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Current-year map")
        selected_year = st.select_slider(
            "Map year",
            options=sorted(df[TIME_COL].unique()),
            value=int(df[TIME_COL].max()),
        )
        map_df = df[df[TIME_COL] == selected_year].dropna(subset=[state.metric])
        fig = px.choropleth(
            map_df,
            locations=ENTITY_COL,
            color=state.metric,
            hover_name=ENTITY_NAME_COL,
            hover_data={state.metric: ":.4f", "gdp_per_capita": ":,.0f", WEIGHT_COL: ":,.0f"},
            color_continuous_scale="Viridis",
            labels={state.metric: metric_label(state.metric)},
        )
        fig.update_geos(showframe=False, showcoastlines=True, projection_type="natural earth")
        fig.update_layout(height=390, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Missingness and zeros")
    cols = available_columns(df, [*INEQUALITY_LABELS.keys(), "gdp_per_capita", WEIGHT_COL])
    quality = pd.DataFrame(
        {
            "variable": cols,
            "missing_pct": [df[col].isna().mean() * 100 for col in cols],
            "zero_pct": [(df[col] == 0).mean() * 100 for col in cols],
        }
    )
    fig = px.bar(
        quality.melt("variable", var_name="status", value_name="percent"),
        x="variable",
        y="percent",
        color="status",
        barmode="group",
        labels={"percent": "Percent of filtered rows", "variable": ""},
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)


def describe_page(df: pd.DataFrame, state: AppState) -> None:
    render_header(df, state)

    st.subheader("Distribution")
    fig = safe_expdpy_figure("explore_histogram", df, state.metric, kde=True)
    if fig is None:
        fig = px.histogram(
            df,
            x=state.metric,
            nbins=45,
            marginal="box",
            color_discrete_sequence=["#2f6f73"],
            labels={state.metric: metric_label(state.metric)},
        )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    stats = (
        df.groupby(TIME_COL)[state.metric]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    stats["weighted_mean"] = [
        weighted_mean(g[state.metric], g[WEIGHT_COL]) for _, g in df.groupby(TIME_COL)
    ]

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Descriptive table by year")
        st.dataframe(stats.round(4), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Top and bottom countries")
        year = st.select_slider(
            "Ranking year",
            options=sorted(df[TIME_COL].unique()),
            value=int(df[TIME_COL].max()),
        )
        n = st.slider("Countries per side", 5, 25, 10)
        ranked = (
            df[df[TIME_COL] == year]
            .dropna(subset=[state.metric])
            .sort_values(state.metric)
        )
        extremes = pd.concat([ranked.head(n), ranked.tail(n)])
        fig = px.bar(
            extremes,
            x=state.metric,
            y=ENTITY_NAME_COL,
            color=np.where(extremes.index.isin(ranked.head(n).index), "Lowest", "Highest"),
            orientation="h",
            labels={state.metric: metric_label(state.metric), ENTITY_NAME_COL: ""},
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


def within_between_page(df: pd.DataFrame, state: AppState) -> None:
    render_header(df, state)
    st.subheader("Within-country and between-country variation")

    panel = df[[ENTITY_COL, ENTITY_NAME_COL, TIME_COL, state.metric]].dropna().copy()
    country_stats = (
        panel.groupby([ENTITY_COL, ENTITY_NAME_COL])[state.metric]
        .agg(country_mean="mean", within_sd="std", observations="count")
        .reset_index()
    )
    between_sd = float(country_stats["country_mean"].std())
    within_sd = float(country_stats["within_sd"].mean())
    total_sd = float(panel[state.metric].std())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total standard deviation", f"{total_sd:.4f}")
    c2.metric("Between-country SD", f"{between_sd:.4f}")
    c3.metric("Average within-country SD", f"{within_sd:.4f}")

    left, right = st.columns([1, 1])
    with left:
        fig = px.bar(
            pd.DataFrame(
                {
                    "component": ["Between countries", "Within countries"],
                    "standard_deviation": [between_sd, within_sd],
                }
            ),
            x="component",
            y="standard_deviation",
            color="component",
            labels={"standard_deviation": "Standard deviation", "component": ""},
        )
        fig.update_layout(height=390, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            country_stats,
            x="country_mean",
            y="within_sd",
            size="observations",
            hover_name=ENTITY_NAME_COL,
            labels={
                "country_mean": f"Country mean: {metric_label(state.metric)}",
                "within_sd": "Within-country SD",
            },
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Country-year heatmap")
    top_n = st.slider("Countries to display", 10, 80, 35)
    order = country_stats.sort_values("country_mean", ascending=False).head(top_n)
    heat = panel[panel[ENTITY_COL].isin(order[ENTITY_COL])].pivot_table(
        index=ENTITY_NAME_COL, columns=TIME_COL, values=state.metric, aggfunc="mean"
    )
    heat = heat.reindex(order[ENTITY_NAME_COL])
    fig = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels=dict(color=metric_label(state.metric), x="Year", y="Country"),
    )
    fig.update_layout(height=max(450, 18 * len(heat)), margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def trends_page(df: pd.DataFrame, state: AppState) -> None:
    render_header(df, state)

    yearly = (
        df.groupby(TIME_COL)
        .agg(
            mean=(state.metric, "mean"),
            median=(state.metric, "median"),
            p10=(state.metric, lambda x: x.quantile(0.10)),
            p90=(state.metric, lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    yearly["population_weighted_mean"] = [
        weighted_mean(g[state.metric], g[WEIGHT_COL]) for _, g in df.groupby(TIME_COL)
    ]

    st.subheader("Global trend")
    fig = go.Figure()
    for col, name in [
        ("mean", "Mean"),
        ("median", "Median"),
        ("population_weighted_mean", "Population-weighted mean"),
    ]:
        fig.add_trace(go.Scatter(x=yearly[TIME_COL], y=yearly[col], mode="lines+markers", name=name))
    fig.add_trace(
        go.Scatter(
            x=pd.concat([yearly[TIME_COL], yearly[TIME_COL][::-1]]),
            y=pd.concat([yearly["p90"], yearly["p10"][::-1]]),
            fill="toself",
            fillcolor="rgba(47,111,115,0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="P10-P90 band",
        )
    )
    fig.update_layout(
        height=440,
        yaxis_title=metric_label(state.metric),
        xaxis_title="Year",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Country trajectories")
    max_lines = st.slider("Maximum country lines", 5, 80, 25)
    country_order = (
        df.groupby(ENTITY_NAME_COL)[state.metric]
        .mean()
        .sort_values(ascending=False)
        .head(max_lines)
        .index
    )
    line_df = df[df[ENTITY_NAME_COL].isin(country_order)].sort_values([ENTITY_NAME_COL, TIME_COL])
    fig = px.line(
        line_df,
        x=TIME_COL,
        y=state.metric,
        color=ENTITY_NAME_COL,
        labels={state.metric: metric_label(state.metric), TIME_COL: "Year"},
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=20, b=10), legend_title="")
    st.plotly_chart(fig, use_container_width=True)


def relationships_page(df: pd.DataFrame, state: AppState) -> None:
    render_header(df, state)

    relation_candidates = available_columns(df, DEFAULT_RELATION_COLUMNS)
    all_numeric = numeric_options(df)
    x_var = st.selectbox(
        "Relationship variable",
        options=relation_candidates + [c for c in all_numeric if c not in relation_candidates],
        index=0,
    )

    plot_df = df.dropna(subset=[x_var, state.metric]).copy()
    if x_var in {"gdp_per_capita", "gdp_total", "pop"}:
        plot_df[f"log_{x_var}"] = np.log10(plot_df[x_var].where(plot_df[x_var] > 0))
        use_log = st.checkbox(f"Use log10({x_var})", value=True)
        x_plot = f"log_{x_var}" if use_log else x_var
    else:
        x_plot = x_var

    st.subheader("Bivariate relationship")
    fig = px.scatter(
        plot_df,
        x=x_plot,
        y=state.metric,
        color=TIME_COL,
        hover_name=ENTITY_NAME_COL,
        hover_data={TIME_COL: True, state.metric: ":.4f", x_var: ":.4f"},
        labels={state.metric: metric_label(state.metric), x_plot: x_plot},
    )

    smooth = plot_df[[x_plot, state.metric]].dropna().sort_values(x_plot)
    if len(smooth) >= 5 and smooth[x_plot].nunique() >= 3:
        degree = 2 if smooth[x_plot].nunique() > 10 else 1
        coef = np.polyfit(smooth[x_plot], smooth[state.metric], degree)
        xs = np.linspace(smooth[x_plot].min(), smooth[x_plot].max(), 120)
        ys = np.polyval(coef, xs)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Polynomial fit", line=dict(color="#d1495b", width=3)))

    fig.update_layout(height=500, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation matrix")
    corr_vars = available_columns(df, [*INEQUALITY_LABELS.keys(), *DEFAULT_RELATION_COLUMNS])
    corr = df[corr_vars].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation"),
    )
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def dynamics_page(df: pd.DataFrame, state: AppState) -> None:
    render_header(df, state)

    panel = df.sort_values([ENTITY_COL, TIME_COL]).copy()
    panel["lag_metric"] = panel.groupby(ENTITY_COL)[state.metric].shift(1)
    panel["delta_metric"] = panel[state.metric] - panel["lag_metric"]
    dyn = panel.dropna(subset=[state.metric, "lag_metric", "delta_metric"])

    st.subheader("Persistence")
    left, right = st.columns([1, 1])
    with left:
        fig = px.scatter(
            dyn,
            x="lag_metric",
            y=state.metric,
            color=TIME_COL,
            hover_name=ENTITY_NAME_COL,
            labels={
                "lag_metric": f"Lagged {metric_label(state.metric)}",
                state.metric: f"Current {metric_label(state.metric)}",
            },
        )
        max_val = np.nanmax([dyn["lag_metric"].max(), dyn[state.metric].max()])
        min_val = np.nanmin([dyn["lag_metric"].min(), dyn[state.metric].min()])
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="No change"))
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            dyn,
            x="delta_metric",
            nbins=50,
            marginal="box",
            labels={"delta_metric": f"Annual change in {metric_label(state.metric)}"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#30343f")
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Largest changes")
    year = st.select_slider(
        "Change ending in year",
        options=sorted(dyn[TIME_COL].unique()),
        value=int(dyn[TIME_COL].max()),
    )
    n = st.slider("Rows", 5, 30, 15)
    changes = dyn[dyn[TIME_COL] == year].copy()
    changes["abs_change"] = changes["delta_metric"].abs()
    changes = changes.sort_values("abs_change", ascending=False).head(n)
    fig = px.bar(
        changes.sort_values("delta_metric"),
        x="delta_metric",
        y=ENTITY_NAME_COL,
        orientation="h",
        color="delta_metric",
        color_continuous_scale="RdBu",
        labels={"delta_metric": f"Change since previous year", ENTITY_NAME_COL: ""},
    )
    fig.update_layout(height=max(420, 24 * n), margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Dynamics data"):
        st.dataframe(
            dyn[[ENTITY_COL, ENTITY_NAME_COL, TIME_COL, "lag_metric", state.metric, "delta_metric"]].round(5),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    page_config()
    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()

    raw = load_data()
    section, state = sidebar(raw)
    df = filtered_data(raw, state)

    if df.empty:
        st.warning("No rows match the current filters.")
        st.stop()

    if section == "Overview":
        overview_page(df, state)
    elif section == "Descriptive statistics":
        describe_page(df, state)
    elif section == "Within and between":
        within_between_page(df, state)
    elif section == "Trends":
        trends_page(df, state)
    elif section == "Relationships":
        relationships_page(df, state)
    elif section == "Dynamics":
        dynamics_page(df, state)


if __name__ == "__main__":
    main()
