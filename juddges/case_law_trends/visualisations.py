import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from juddges.case_law_trends.constants import DZIUBAK_JUDGMENT_DATE


def _prepare_data(extractions_df, column_name, min_year=2015):
    """Prepare and filter data for plotting."""
    # Filter by minimum year if specified
    if min_year is not None:
        extractions_df = extractions_df[pd.to_datetime(extractions_df["date"]).dt.year >= min_year]

    # Filter out empty values
    return extractions_df[
        (extractions_df[column_name].notna()) & (extractions_df[column_name] != "")
    ]


def _setup_subplot(ax, period_type, column_name, ylabel):
    """Setup common subplot properties."""
    ax.set_xlabel("Rok" if period_type == "year" else "Rok-Kwartał")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Rozkład {period_type}y {column_name}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)


def _add_dziubak_line(ax, counts, period_type):
    """Add Dziubak judgment vertical line to plot."""
    if period_type == "year":
        dziubak_date = pd.to_datetime(DZIUBAK_JUDGMENT_DATE).year
    else:
        dziubak_date = pd.to_datetime(DZIUBAK_JUDGMENT_DATE).to_period("Q")

    dziubak_idx = np.where(counts.index == dziubak_date)[0][0]
    ax.axvline(x=dziubak_idx, color="red", linestyle="--", label="Wyrok Dziubaka")


def _get_counts(extractions_df, dates, column_name, period_type):
    """Calculate counts based on period type."""
    if period_type == "year":
        extractions_df["period"] = dates.dt.year
    else:
        extractions_df["period"] = dates.dt.to_period("Q")

    return extractions_df.groupby(["period", column_name]).size().unstack(fill_value=0)


def plot_distributions_stacked(extractions_df, column_name, min_year=2015):
    """Plot stacked percentage distributions by year and quarter."""
    filtered_df = _prepare_data(extractions_df, column_name, min_year)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    dates = pd.to_datetime(filtered_df["date"])

    # Yearly plot
    yearly_counts = _get_counts(filtered_df, dates, column_name, "year")
    yearly_pct = yearly_counts.div(yearly_counts.sum(axis=1), axis=0) * 100
    yearly_pct.plot(kind="bar", stacked=True, ax=ax1, alpha=0.7)
    _add_dziubak_line(ax1, yearly_counts, "year")
    _setup_subplot(ax1, "roczn", column_name, "Procent wyroków")

    # Quarterly plot
    quarterly_counts = _get_counts(filtered_df, dates, column_name, "quarter")
    quarterly_pct = quarterly_counts.div(quarterly_counts.sum(axis=1), axis=0) * 100
    quarterly_pct.plot(kind="bar", stacked=True, ax=ax2, alpha=0.7)
    _add_dziubak_line(ax2, quarterly_counts, "quarter")
    _setup_subplot(ax2, "kwartaln", column_name, "Procent wyroków")

    plt.tight_layout()
    return fig


def plot_distributions(extractions_df, column_name, min_year=2015):
    """Plot absolute count distributions by year and quarter."""
    filtered_df = _prepare_data(extractions_df, column_name, min_year)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    dates = pd.to_datetime(filtered_df["date"])

    # Yearly plot
    yearly_counts = _get_counts(filtered_df, dates, column_name, "year")
    yearly_counts.plot(kind="bar", ax=ax1, alpha=0.7)
    _add_dziubak_line(ax1, yearly_counts, "year")
    _setup_subplot(ax1, "roczn", column_name, "Liczba wyroków")

    # Quarterly plot
    quarterly_counts = _get_counts(filtered_df, dates, column_name, "quarter")
    quarterly_counts.plot(kind="bar", ax=ax2, alpha=0.7)
    _add_dziubak_line(ax2, quarterly_counts, "quarter")
    _setup_subplot(ax2, "kwartaln", column_name, "Liczba wyroków")

    plt.tight_layout()
    return fig
