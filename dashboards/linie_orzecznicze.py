from pathlib import Path

import pandas as pd
import streamlit as st

from juddges.case_law_trends.visualisations import (
    plot_distributions,
    plot_distributions_stacked,
)
from juddges.prompts.information_extraction import SWISS_FRANC_LOAN_SCHEMA
from juddges.settings import FRANKOWICZE_DATA_PATH

st.title("Analiza Linii Orzeczniczych")


# Load data
@st.cache_data
def load_data(file_path):
    df = pd.read_pickle(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


# Get list of available data files
data_files = list(FRANKOWICZE_DATA_PATH.glob("extractions_df_*.pkl"))
file_options = {f.name: f for f in data_files}

# File selection widget
selected_filename = st.selectbox(
    "Wybierz plik z danymi:",
    options=list(file_options.keys()),
    index=len(file_options) - 1,  # Default to most recent file
)

df = load_data(file_options[selected_filename])

# Data overview section
st.header("Przegląd Danych")
st.write(f"Liczba spraw w bazie: {len(df):,}")
st.write(f"Zakres dat: od {df['date'].min():%Y-%m-%d} do {df['date'].max():%Y-%m-%d}")

# Show sample of the data
st.subheader("Przykładowe Dane")
st.dataframe(df.head())

# Schema description
st.header("Schema Danych / Kodowanie")
with st.expander("Show Schema"):
    st.markdown(
        f"""```yaml
{SWISS_FRANC_LOAN_SCHEMA}
```
"""
    )

# Visualization section
st.header("Wizualizacje")
# Get list of available columns for visualization
viz_columns = [col for col in df.columns if df[col].dtype == "object"]

# Column selection widget
# Sort columns to put wynik_sprawy and typ_rozstrzygniecia first, rest alphabetically
priority_columns = ["wynik_sprawy", "typ_rozstrzygniecia"]
other_columns = sorted([col for col in viz_columns if col not in priority_columns])
sorted_columns = priority_columns + other_columns

selected_column = st.selectbox(
    "Wybierz kolumnę do wizualizacji:",
    options=sorted_columns,
    index=0,
)

# Plot type selection
plot_type = st.radio(
    "Wybierz typ wykresu:", ["Wartości bezwzględne", "Wartości procentowe (stacked)"]
)

# Add minimum year selection
min_year = st.slider(
    "Wybierz minimalny rok orzeczenia:",
    min_value=int(df["date"].dt.year.min()),
    max_value=int(df["date"].dt.year.max()),
    value=2015,
)

if plot_type == "Wartości bezwzględne":
    st.pyplot(plot_distributions(df, selected_column, min_year=min_year))
else:
    st.pyplot(plot_distributions_stacked(df, selected_column, min_year=min_year))
