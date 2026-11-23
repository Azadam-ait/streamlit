import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


def benford_statistics(series, digits):
    """Return observed and expected Benford frequencies for positive numeric values."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[(numeric > 0) & np.isfinite(numeric)]

    if numeric.empty:
        return pd.DataFrame(columns=["digit", "count", "percent", "benford", "difference"])

    def extract_prefix(value):
        mantissa = abs(float(value))
        if mantissa == 0 or not np.isfinite(mantissa):
            return None

        while mantissa >= 10:
            mantissa /= 10
        while mantissa < 1:
            mantissa *= 10

        return int(np.floor(mantissa * (10 ** (digits - 1))))

    prefixes = numeric.apply(extract_prefix)

    if digits == 1:
        prefixes = prefixes[prefixes.between(1, 9)]
        possible = np.arange(1, 10)
    else:
        prefixes = prefixes[prefixes.between(10, 99)]
        possible = np.arange(10, 100)

    observed = prefixes.value_counts().reindex(possible, fill_value=0)
    percentages = observed / observed.sum() if observed.sum() else observed.astype(float)
    expected = pd.Series(
        [np.log10(1 + 1 / int(digit)) for digit in possible],
        index=possible,
    )

    result = pd.DataFrame({
        "digit": possible,
        "count": observed.values,
        "percent": percentages.values,
        "benford": expected.values,
    })
    result["difference"] = result["percent"] - result["benford"]
    return result


def ks_test(statistics):
    if statistics.empty:
        return None, None

    observed_cumulative = np.cumsum(statistics["percent"])
    expected_cumulative = np.cumsum(statistics["benford"])
    maximum_difference = float(np.max(np.abs(observed_cumulative - expected_cumulative)))
    critical_value = 1.36 / np.sqrt(statistics["count"].sum())
    return maximum_difference, critical_value


def plot_benford(statistics, title):
    is_two_digits = len(statistics) > 20
    fig_width = 16 if is_two_digits else 10
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = np.arange(len(statistics))
    width = 0.42 if not is_two_digits else 0.78

    observed = statistics["percent"] * 100
    expected = statistics["benford"] * 100

    ax.bar(x - width / 2, observed, width, label="Observed", color="steelblue")
    ax.bar(x + width / 2, expected, width, label="Benford expected", color="darkorange")

    if is_two_digits:
        tick_positions = np.arange(0, len(statistics), 5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(statistics.iloc[tick_positions]["digit"].astype(str), rotation=45, ha="right")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(statistics["digit"].astype(str))

    ax.set_xlabel("Leading digit(s)")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Data Filter", layout="wide")
st.title("Filter uploaded data")

uploaded_file = st.file_uploader(
    "Upload data", accept_multiple_files=False, type=["csv", "xlsx"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    st.write("### Dataset preview")
    st.dataframe(df, use_container_width=True)

    num_filters = int(st.number_input(
        "Number of filters",
        min_value=3,
        max_value=min(10, len(df.columns)),
        value=3,
        step=1,
    ))

    criteria = []
    st.write("### Filter criteria")

    for i in range(num_filters):
        column = st.selectbox(
            f"Filter {i + 1}: choose a column",
            df.columns,
            key=f"column_{i}",
        )
        series = df[column].dropna()

        if pd.api.types.is_numeric_dtype(series):
            minimum = float(series.min())
            maximum = float(series.max())
            value = st.slider(
                f"Select range for {column}",
                minimum,
                maximum,
                (minimum, maximum),
                key=f"value_{i}",
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            minimum = pd.to_datetime(series.min())
            maximum = pd.to_datetime(series.max())
            value = st.date_input(
                f"Select date range for {column}",
                (minimum.date(), maximum.date()),
                key=f"value_{i}",
            )
        else:
            options = sorted(series.astype(str).unique().tolist())
            value = st.multiselect(
                f"Select values for {column}",
                options,
                key=f"value_{i}",
            )

        criteria.append((column, value))

    if st.button("Apply all filters"):
        filtered_df = df.copy()

        for column, value in criteria:
            series = filtered_df[column]

            if pd.api.types.is_numeric_dtype(series):
                filtered_df = filtered_df[
                    series.between(value[0], value[1])
                ]
            elif pd.api.types.is_datetime64_any_dtype(series):
                dates = pd.to_datetime(series)
                filtered_df = filtered_df[
                    dates.dt.date.between(value[0], value[1])
                ]
            elif value:
                filtered_df = filtered_df[
                    series.astype(str).isin(value)
                ]

        st.session_state["filtered_df"] = filtered_df

    filtered_df = st.session_state.get("filtered_df")

    if filtered_df is not None:
        st.write(f"### Filtered data ({len(filtered_df)} rows)")
        st.dataframe(filtered_df, use_container_width=True)
        st.download_button(
            "Download filtered CSV",
            filtered_df.to_csv(index=False).encode("utf-8"),
            "filtered_data.csv",
            "text/csv",
        )

        st.write("### Benford's Law analysis")
        numeric_columns = filtered_df.select_dtypes(include=["number"]).columns.tolist()

        if numeric_columns:
            column = st.selectbox(
                "Select a numeric column for Benford's analysis",
                numeric_columns,
                key="benford_column",
            )

            digits = st.radio(
                "Select number of digits for analysis",
                [1, 2],
                horizontal=True,
                key="benford_digits",
            )

            statistics = benford_statistics(filtered_df[column], digits)

            if statistics.empty:
                st.info("No positive numeric values are available for this analysis.")
            else:
                st.write("#### Statistics")
                display_statistics = statistics.copy()
                display_statistics["percent"] = display_statistics["percent"] * 100
                display_statistics["benford"] = display_statistics["benford"] * 100
                display_statistics["difference"] = display_statistics["difference"] * 100

                st.dataframe(
                    display_statistics,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "digit": st.column_config.TextColumn("Leading digits", width="small"),
                        "count": st.column_config.NumberColumn("Count", format="%d"),
                        "percent": st.column_config.NumberColumn("Observed %", format="%.2f%%"),
                        "benford": st.column_config.NumberColumn("Benford %", format="%.2f%%"),
                        "difference": st.column_config.NumberColumn("Difference %", format="%.2f%%"),
                    },
                )

                st.write("#### Largest deviations")
                deviations = display_statistics.copy()
                deviations["Absolute difference"] = deviations["difference"].abs()
                largest_deviations = deviations.nlargest(10, "Absolute difference")
                st.dataframe(
                    largest_deviations[["digit", "count", "Observed %" if "Observed %" in largest_deviations else "percent"]],
                    use_container_width=True,
                    hide_index=True,
                )

                maximum_difference, critical_value = ks_test(statistics)
                if maximum_difference is not None:
                    result_column, critical_column = st.columns(2)
                    result_column.metric("Maximum difference", f"{maximum_difference:.4f}")
                    critical_column.metric("Critical value (alpha=0.05)", f"{critical_value:.4f}")

                    if maximum_difference > critical_value:
                        st.warning("The distribution differs significantly from Benford's Law.")
                    else:
                        st.success("The distribution does not differ significantly from Benford's Law.")

                st.write("#### Download analysis")

                chart_title = f"Benford's Law Analysis for {column} (Digits: {digits})"
                figure = plot_benford(statistics, chart_title)
                st.pyplot(figure)

                download_csv = display_statistics.rename(columns={
                    "digit": "leading_digits",
                    "count": "count",
                    "percent": "observed_percent",
                    "benford": "benford_percent",
                    "difference": "difference_percent",
                }).to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download summary statistics (CSV)",
                    download_csv,
                    f"benford_{digits}_digit_summary.csv",
                    "text/csv",
                )

                conclusion = (
                    "The distribution differs significantly from Benford's Law."
                    if maximum_difference > critical_value
                    else "The distribution does not differ significantly from Benford's Law."
                )

                report = (
                    "BENFORD ANALYSIS REPORT\n"
                    "=======================\n\n"
                    f"Numeric column: {column}\n"
                    f"Leading digits analyzed: {digits}\n"
                    f"Number of positive values: {statistics['count'].sum()}\n"
                    f"Maximum KS difference: {maximum_difference:.6f}\n"
                    f"Critical value (alpha=0.05): {critical_value:.6f}\n"
                    f"Conclusion: {conclusion}\n\n"
                    "SUMMARY STATISTICS\n"
                    f"{display_statistics.to_string(index=False)}\n\n"
                    "10 LARGEST DEVIATIONS\n"
                    f"{largest_deviations[['digit', 'count', 'percent', 'benford', 'difference']].to_string(index=False)}\n"
                )

                st.download_button(
                    "Download analysis report (TXT)",
                    report.encode("utf-8"),
                    f"benford_{digits}_digit_report.txt",
                    "text/plain",
                )

                png_buffer = BytesIO()
                figure.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
                png_buffer.seek(0)
                st.download_button(
                    "Download visualization (PNG)",
                    png_buffer.getvalue(),
                    f"benford_{digits}_digit_chart.png",
                    "image/png",
                )

                pdf_buffer = BytesIO()
                figure.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
                pdf_buffer.seek(0)
                st.download_button(
                    "Download visualization (PDF)",
                    pdf_buffer.getvalue(),
                    f"benford_{digits}_digit_chart.pdf",
                    "application/pdf",
                )
        else:
            st.info("No numeric columns are available for Benford's Law analysis.")


