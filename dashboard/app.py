"""Public, read-only dashboard for historical India crop-production risk."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crop_risk.live_data import LiveDataError, fetch_current_upag, read_snapshot  # noqa: E402
from crop_risk.production_analysis import (  # noqa: E402
    RISK_THRESHOLD_PCT,
    build_historical_panel,
    summarize_source,
    walk_forward_evaluation,
)


SNAPSHOT_PATH = PROJECT_ROOT / "data" / "public" / "upag_crop_production.csv.gz"
APP_VERSION = "0.2.0"
DATA_CACHE_SECONDS = 6 * 60 * 60

st.set_page_config(
    page_title="CropLens India | Harvest Risk Monitor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=DATA_CACHE_SECONDS, max_entries=1, show_spinner=False)
def load_source_data() -> tuple[pd.DataFrame, dict[str, Any], str, str | None]:
    """Prefer current source rows; fall back to the checksum-verified snapshot."""
    try:
        frame, metadata = fetch_current_upag()
        return frame, metadata, "live_source", None
    except LiveDataError as exc:
        try:
            frame, metadata = read_snapshot(SNAPSHOT_PATH)
            return frame, metadata, "snapshot_fallback", str(exc)
        except LiveDataError as snapshot_error:
            raise RuntimeError(
                "The live crop-data feed and bundled snapshot are both unavailable. "
                "Please try again later."
            ) from snapshot_error


def display_year(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    year = str(value)
    return year if "-" in year else f"{year}"


def make_source_version(metadata: dict[str, Any], mode: str) -> str:
    return "|".join(
        str(value or "")
        for value in (
            mode,
            metadata.get("resource_id"),
            metadata.get("source_updated_at"),
            metadata.get("fetched_at_utc"),
            metadata.get("snapshot_sha256"),
        )
    )


def load_panel_for_session(
    source: pd.DataFrame, metadata: dict[str, Any], mode: str
) -> tuple[pd.DataFrame, str]:
    version = make_source_version(metadata, mode)
    cached_version = st.session_state.get("panel_version")
    if cached_version != version:
        st.session_state["analysis_panel"] = build_historical_panel(source)
        st.session_state["panel_version"] = version
        st.session_state.pop("model_evaluation", None)
    return st.session_state["analysis_panel"], version


def apply_filters(frame: pd.DataFrame, state: str, crop: str, season: str) -> pd.DataFrame:
    selected = frame
    if state != "All India":
        selected = selected[selected["state_name"].astype(str) == state]
    if crop != "All crops":
        selected = selected[selected["crop_name"].astype(str) == crop]
    if season != "All seasons":
        selected = selected[selected["season"].astype(str) == season]
    return selected.copy()


def render_data_quality(profile: dict[str, Any]) -> None:
    columns = st.columns(4)
    columns[0].metric("Source rows", f"{profile['rows']:,}")
    columns[1].metric("States / UTs", f"{profile['states']:,}")
    columns[2].metric("Districts", f"{profile['districts']:,}")
    columns[3].metric("Crop identifiers", f"{profile['crops']:,}")


st.title("🌾 CropLens India")
st.subheader("District crop-production and harvest-risk monitor")
st.caption(
    "A public data product for retrospective yield-risk screening, statistical analysis, and model validation. "
    "Source: Ministry of Agriculture & Farmers Welfare APY data published through UPAg."
)

with st.sidebar:
    st.header("Data controls")
    if st.button("Refresh source now", use_container_width=True):
        load_source_data.clear()
        st.session_state.pop("analysis_panel", None)
        st.session_state.pop("panel_version", None)
        st.session_state.pop("model_evaluation", None)
        st.rerun()
    st.caption("Automatic source cache: 6 hours. Refresh is read-only.")

try:
    with st.spinner("Loading crop production records…"):
        source, metadata, data_mode, source_error = load_source_data()
        panel, source_version = load_panel_for_session(source, metadata, data_mode)
except (RuntimeError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

profile = summarize_source(source)
if data_mode == "live_source":
    status_text = "Live source connected"
    status_caption = (
        f"Fetched {int(metadata['row_count']):,} rows from the public APY/UPAG data API. "
        f"Source vintage: {metadata.get('source_updated_at') or 'not supplied by the source'}; "
        f"retrieved {metadata.get('fetched_at_utc', 'time not supplied')} UTC."
    )
    st.success(status_caption)
else:
    status_text = "Verified snapshot fallback"
    st.warning(
        "The live source could not be reached, so this page is using its bundled, checksum-verified "
        f"data snapshot ({metadata.get('fetched_at_utc', 'retrieval date unavailable')}). "
        "The dashboard is showing historical records, not current field conditions."
    )
    if source_error:
        with st.expander("Source connection details"):
            st.code(source_error)

st.info(
    "This is a retrospective production-data monitor. It identifies observed yield declines against a "
    "prior 5-year median and evaluates a history-only model on later years. It does not forecast the "
    "current season, ingest live weather/satellite conditions, or issue farmer, insurance, or relief advice."
)

left_filter, crop_filter, season_filter = st.columns(3)
state_values = sorted(source["state_name"].dropna().astype(str).unique().tolist())
crop_values = sorted(source["crop_name"].dropna().astype(str).unique().tolist())
season_values = sorted(source["season"].dropna().astype(str).unique().tolist())
with left_filter:
    selected_state = st.selectbox("State / UT", ["All India", *state_values])
with crop_filter:
    default_crop = crop_values.index("Rice") + 1 if "Rice" in crop_values else 0
    selected_crop = st.selectbox("Crop", ["All crops", *crop_values], index=default_crop)
with season_filter:
    selected_season = st.selectbox("Season", ["All seasons", *season_values])

scope = apply_filters(panel, selected_state, selected_crop, selected_season)
if scope.empty:
    st.warning("No source rows match these filters.")
    st.stop()

available_years = (
    scope[["season_year", "year"]]
    .drop_duplicates()
    .sort_values("season_year", kind="stable")
)
latest_year = int(available_years["season_year"].max())
latest_label = str(available_years.loc[available_years["season_year"] == latest_year, "year"].iloc[-1])
latest_scope = scope[scope["season_year"] == latest_year].copy()
latest_with_baseline = latest_scope.dropna(subset=["yield_anomaly_pct"])
latest_flags = latest_with_baseline[latest_with_baseline["yield_anomaly_pct"] <= RISK_THRESHOLD_PCT]
latest_median_anomaly = latest_with_baseline["yield_anomaly_pct"].median()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Records in view", f"{len(scope):,}")
kpi2.metric("Latest source season", latest_label)
kpi3.metric("Latest rows with baseline", f"{len(latest_with_baseline):,}")
kpi4.metric(
    "Observed low-yield flags",
    f"{len(latest_flags):,}",
    help=f"Yield at least {abs(RISK_THRESHOLD_PCT):.0f}% below the prior 5-year median.",
)
if pd.notna(latest_median_anomaly):
    st.caption(
        f"Median yield anomaly in {latest_label}: {latest_median_anomaly:+.1f}% across rows with a "
        "baseline of at least 3 observations from the preceding 5 years. This is retrospective, not a "
        "probability of future loss."
    )

overview_tab, hotspot_tab, model_tab, source_tab = st.tabs(
    ["Production overview", "Observed declines", "Model validation", "Data & limitations"]
)

with overview_tab:
    st.subheader("Reported yield over time")
    annual = (
        scope.dropna(subset=["yield"])
        .groupby(["season_year", "year"], as_index=False)
        .agg(
            median_yield_kg_ha=("yield", "median"),
            p25_yield_kg_ha=("yield", lambda values: values.quantile(0.25)),
            p75_yield_kg_ha=("yield", lambda values: values.quantile(0.75)),
            reporting_rows=("yield", "size"),
        )
        .sort_values("season_year")
    )
    if not annual.empty:
        chart = annual.set_index("year")[["median_yield_kg_ha", "p25_yield_kg_ha", "p75_yield_kg_ha"]]
        st.line_chart(chart, y_label="Reported yield (kg/ha)", x_label="Season")
        with st.expander("Annual summary values"):
            st.dataframe(annual, hide_index=True, width="stretch")
        st.caption(
            "Each point is the unweighted median of reported district rows in the selected view. "
            "The 25th and 75th percentiles describe the cross-sectional spread; districts are not equally sized."
        )
    else:
        st.info("There are no numeric yield values for the selected view.")

    st.subheader(f"Latest observed records · {latest_label}")
    latest_columns = [
        "state_name",
        "district_name",
        "crop_name",
        "season",
        "area",
        "production",
        "yield",
        "prior_median",
        "yield_anomaly_pct",
        "observed_low_yield",
    ]
    display_latest = latest_scope[latest_columns].copy()
    display_latest = display_latest.rename(
        columns={
            "state_name": "State / UT",
            "district_name": "District",
            "crop_name": "Crop",
            "season": "Season",
            "area": "Area (ha)",
            "production": "Production (tonnes)",
            "yield": "Yield (kg/ha)",
            "prior_median": "Prior 5-year median (kg/ha)",
            "yield_anomaly_pct": "Yield anomaly (%)",
            "observed_low_yield": "Observed decline flag",
        }
    )
    st.dataframe(display_latest.head(100), hide_index=True, width="stretch")
    st.caption("The table is capped at 100 rows; export the filtered records below to review the full selection.")
    st.download_button(
        "Download filtered records as CSV",
        data=scope.to_csv(index=False).encode("utf-8"),
        file_name="croplens_filtered_apy_records.csv",
        mime="text/csv",
    )

with hotspot_tab:
    st.subheader(f"Observed low-yield anomalies · {latest_label}")
    st.caption(
        f"A row is flagged when its observed yield is at least {abs(RISK_THRESHOLD_PCT):.0f}% below "
        "its own district/crop/season median of observations in the preceding 5 years. At least 3 prior "
        "observations are required."
    )
    if latest_with_baseline.empty:
        st.info("No rows in this selection have enough earlier observations to calculate a baseline.")
    else:
        state_summary = (
            latest_with_baseline.assign(
                flagged=latest_with_baseline["yield_anomaly_pct"] <= RISK_THRESHOLD_PCT
            )
            .groupby("state_name", as_index=False)
            .agg(
                district_crop_rows=("yield_anomaly_pct", "size"),
                flagged_rows=("flagged", "sum"),
                median_anomaly_pct=("yield_anomaly_pct", "median"),
            )
            .sort_values(["flagged_rows", "median_anomaly_pct"], ascending=[False, True])
        )
        st.markdown("**State / UT summary**")
        st.dataframe(state_summary.head(25), hide_index=True, width="stretch")

        st.markdown("**Largest observed declines**")
        anomaly_cols = [
            "year",
            "state_name",
            "district_name",
            "crop_name",
            "season",
            "yield",
            "prior_median",
            "yield_anomaly_pct",
        ]
        declines = latest_with_baseline.sort_values("yield_anomaly_pct").head(50)[anomaly_cols]
        st.dataframe(declines, hide_index=True, width="stretch")
        st.caption(
            "Flags are computed from reported historical yield only. They are not hazard alerts, causal "
            "attributions, or evidence that a specific intervention is needed."
        )

with model_tab:
    st.subheader("History-only low-yield classifier")
    st.write(
        "The random forest uses lagged yield and area history to score whether a later reported yield "
        "falls below the prior median threshold. Every validation year is held out; the model only sees "
        "earlier years during that fold."
    )
    st.warning(
        "These are historical backtest scores, not current-season forecasts. Probabilities are uncalibrated, "
        "the model has no weather, soil, satellite, irrigation, or crop-condition features, and this model "
        "is not used to issue operational decisions."
    )
    evaluation = st.session_state.get("model_evaluation")
    evaluation_version = st.session_state.get("model_evaluation_version")
    if evaluation is None or evaluation_version != source_version:
        if st.button("Run walk-forward backtest for current filters", type="primary"):
            with st.spinner("Fitting and scoring year-block models…"):
                evaluation = walk_forward_evaluation(scope)
            st.session_state["model_evaluation"] = evaluation
            st.session_state["model_evaluation_version"] = source_version
            evaluation_version = source_version
    if evaluation is not None and evaluation_version == source_version:
        if evaluation.get("status") != "complete":
            st.info(evaluation.get("message", "More history is required for evaluation."))
        else:
            overall = evaluation["overall"]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Out-of-time rows", f"{overall['holdout_rows']:,}")
            metric_cols[1].metric(
                "Average precision",
                "Not available" if overall["average_precision"] is None else f"{overall['average_precision']:.3f}",
            )
            metric_cols[2].metric(
                "Prevalence baseline",
                f"{overall['baseline_average_precision']:.3f}",
                help="Average precision of a constant score equals the positive class prevalence.",
            )
            metric_cols[3].metric(
                "Brier score",
                f"{overall['brier_score']:.3f}",
                delta=f"{overall['brier_score'] - overall['baseline_brier_score']:+.3f} vs prevalence",
                delta_color="inverse",
                help="Lower is better. Delta compares this score with a constant score set to the holdout event prevalence.",
            )
            year_labels = (
                scope[["season_year", "year"]]
                .drop_duplicates()
                .set_index("season_year")["year"]
                .astype(str)
                .to_dict()
            )
            holdout_labels = [year_labels.get(int(year), str(year)) for year in overall["holdout_years"]]
            st.caption(
                f"Holdout seasons: {', '.join(holdout_labels)}. "
                f"Target: {overall['target']}. Lower Brier is better; average precision should be compared "
                "with the prevalence baseline. Scores are historical and uncalibrated."
            )
            fold_table = pd.DataFrame(evaluation["folds"])
            fold_table["holdout_season"] = fold_table["holdout_year"].map(
                lambda year: year_labels.get(int(year), str(year))
            )
            fold_table = fold_table.drop(columns="holdout_year")
            st.dataframe(fold_table, hide_index=True, width="stretch")
            with st.expander("Evaluation protocol"):
                st.markdown(
                    "- Expanding time split: training rows satisfy `season_year < holdout_year`.\n"
                    "- Validation is the next full year block; random row splitting is not used.\n"
                    "- Features use only prior yield/area reports, with at least three observations for the baseline.\n"
                    "- Target: current reported yield at least 15% below the prior 5-year median.\n"
                    "- The latest three available holdout years are pooled for summary metrics."
                )
    else:
        st.caption("Run the backtest to calculate out-of-time metrics for the selected state, crop, and season.")

with source_tab:
    st.subheader("Source and lineage")
    render_data_quality(profile)
    st.markdown(
        f"""
        **Dataset:** Ministry of Agriculture & Farmers Welfare, Area, Production and Yield (APY),
        crop-wise data published through UPAg and the [India Data Portal]({metadata.get('source_url', '')}).

        **License:** {metadata.get('license_name', 'See source metadata')} —
        [ODC-BY 1.0 terms]({metadata.get('license_url', '')}). Attribution is retained in this app and
        the downloadable project snapshot.

        **Source vintage:** {metadata.get('source_updated_at') or 'Not supplied'}

        **Retrieved at:** {metadata.get('fetched_at_utc') or 'Not supplied'}

        **Coverage in loaded rows:** {profile['year_start']}–{profile['year_end']}

        **App version:** {APP_VERSION}
        """
    )
    st.markdown("**Source-quality checks**")
    quality1, quality2, quality3, quality4, quality5 = st.columns(5)
    quality1.metric("Rows with missing yield", f"{profile['missing_yield_rows']:,}")
    quality2.metric("Duplicate panel keys", f"{profile['duplicate_key_rows']:,}")
    quality3.metric("Negative reported yields", f"{profile['negative_yield_rows']:,}")
    quality4.metric("Missing district codes", f"{profile['missing_district_code_rows']:,}")
    quality5.metric("Missing crop codes", f"{profile['missing_crop_code_rows']:,}")
    st.markdown("**Known limitations**")
    st.markdown(
        """
        - The source is annual administrative crop-production reporting. New districts, revised boundaries,
          changing crop codes, and incomplete reporting can break long time-series comparisons.
        - Yield is reported in kilograms per hectare. The dashboard preserves source values and does not
          infer missing yields or aggregate duplicate records.
        - The risk screen uses prior observed yields only. It cannot explain a decline or establish that a
          weather event caused it.
        - The history-only model is an analyst benchmark. It is not an operational forecast and has not been
          field-validated or reviewed for farmer, insurance, credit, relief, or public-policy decisions.
        - NLP bulletin extraction and plant-leaf computer-vision experiments remain independent research
          modules. They are not fused into this district-level view because aligned, point-in-time inputs are
          not available in this source.
        """
    )
    st.markdown("**Project materials**")
    st.markdown(
        "See `README.md`, `docs/model_card.md`, `docs/data_sources.md`, and `docs/runbook.md` "
        "for setup, evaluation, source terms, deployment steps, and operational limits."
    )

st.divider()
st.caption(
    f"{status_text} · Read-only analyst tool · No farmer-level recommendations or automated decisions · "
    "Data attribution: Ministry of Agriculture & Farmers Welfare / India Data Portal"
)
