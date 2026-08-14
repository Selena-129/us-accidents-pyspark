
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "results"
OUT = PROJECT / "charts"
OUT.mkdir(exist_ok=True)

BLUE = "#176B87"
LIGHT_BLUE = "#64A6BD"
ORANGE = "#E69F00"
GREEN = "#2A9D8F"
GREY = "#6B7280"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 160,
})


def comma(x, _):
    return f"{int(x):,}"


def finish(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def horizontal_top(data, label, value, title, filename, n=10):
    plot = data.nlargest(n, value).sort_values(value)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.barh(plot[label], plot[value], color=BLUE)
    ax.set_title(title)
    ax.set_xlabel("Recorded accident count")
    ax.xaxis.set_major_formatter(FuncFormatter(comma))
    ax.grid(axis="x", alpha=0.2)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in plot[value]], padding=4, fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    finish(fig, filename)


# 1. States
states = pd.read_csv(ROOT / "state_summary.csv")
horizontal_top(states, "State", "Accident_Count", "Top 10 States by Recorded Accident Count", "01_top_states.png")

# 2. Cities
cities = pd.read_csv(ROOT / "top_cities.csv")
horizontal_top(cities, "City_State", "Accident_Count", "Top 10 City–State Groups by Recorded Accident Count", "02_top_cities.png")

# 3. Annual counts; partial years are visually distinguished.
years = pd.read_csv(ROOT / "year_summary.csv")
fig, ax = plt.subplots(figsize=(8.5, 5.2))
colors = [GREY if year in (2016, 2023) else BLUE for year in years["Year"]]
bars = ax.bar(years["Year"].astype(str), years["Accident_Count"], color=colors)
ax.set_title("Recorded Accident Count by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Recorded accident count")
ax.yaxis.set_major_formatter(FuncFormatter(comma))
ax.grid(axis="y", alpha=0.2)
ax.bar_label(bars, labels=[f"{v/1_000_000:.2f}M" for v in years["Accident_Count"]], padding=3, fontsize=8)
ax.text(0.01, -0.18, "Note: 2016 and 2023 are partial-coverage years (shown in grey).", transform=ax.transAxes, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "03_yearly_counts.png")

# 4. Weekdays
weekdays = pd.read_csv(ROOT / "weekday_summary.csv")
fig, ax = plt.subplots(figsize=(8.5, 5.2))
bars = ax.bar(weekdays["Weekday"], weekdays["Accident_Count"], color=BLUE)
ax.set_title("Recorded Accident Count by Weekday")
ax.set_ylabel("Recorded accident count")
ax.yaxis.set_major_formatter(FuncFormatter(comma))
ax.grid(axis="y", alpha=0.2)
ax.bar_label(bars, labels=[f"{v:.1f}%" for v in weekdays["Percentage_of_All_Records"]], padding=3, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "04_weekday_distribution.png")

# 5. Hours
hours = pd.read_csv(ROOT / "hour_summary.csv")
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(hours["Hour"], hours["Accident_Count"], color=BLUE, linewidth=2.5, marker="o", markersize=4)
ax.fill_between(hours["Hour"], hours["Accident_Count"], alpha=0.12, color=LIGHT_BLUE)
ax.set_title("Recorded Accident Count by Hour of Day")
ax.set_xlabel("Hour of day (recorded local time)")
ax.set_ylabel("Recorded accident count")
ax.set_xticks(range(0, 24))
ax.yaxis.set_major_formatter(FuncFormatter(comma))
ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "05_hourly_distribution.png")

# 6. Severity
severity = pd.read_csv(ROOT / "severity_distribution.csv")
fig, ax = plt.subplots(figsize=(7.5, 5.2))
bars = ax.bar(severity["Severity"].astype(str), severity["Percentage_of_All_Records"], color=[LIGHT_BLUE, BLUE, ORANGE, "#C44E52"])
ax.set_title("Distribution of Recorded Severity Levels")
ax.set_xlabel("Severity level")
ax.set_ylabel("Percentage of records (%)")
ax.set_ylim(0, max(severity["Percentage_of_All_Records"]) * 1.18)
ax.grid(axis="y", alpha=0.2)
ax.bar_label(bars, labels=[f"{v:.2f}%" for v in severity["Percentage_of_All_Records"]], padding=4)
ax.text(0.01, -0.18, "Severity describes traffic impact/delay, not injury severity.", transform=ax.transAxes, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "06_severity_distribution.png")

# 7. Weather frequency
weather = pd.read_csv(ROOT / "weather_summary.csv")
weather = weather[weather["Weather_Condition_Group"] != "Missing"]
horizontal_top(weather, "Weather_Condition_Group", "Accident_Count", "Top 10 Weather Conditions by Recorded Accident Count", "07_weather_conditions.png")

# 8. Road-feature prevalence (present records only).
road = pd.read_csv(ROOT / "road_feature_summary.csv")
present = road[road["Feature_Present"].astype(str).str.lower() == "true"].copy()
present = present[present["Feature"] != "Turning_Loop"].sort_values("Accident_Count")
fig, ax = plt.subplots(figsize=(8.5, 6.2))
bars = ax.barh(present["Feature"].str.replace("_", " "), present["Accident_Count"], color=GREEN)
ax.set_title("Recorded Accidents with Selected Road-Feature Indicators")
ax.set_xlabel("Records where feature indicator was present")
ax.xaxis.set_major_formatter(FuncFormatter(comma))
ax.grid(axis="x", alpha=0.2)
ax.bar_label(bars, labels=[f"{v:,.0f}" for v in present["Accident_Count"]], padding=3, fontsize=8)
ax.text(0.01, -0.12, "Feature indicators can overlap; counts are not mutually exclusive.", transform=ax.transAxes, fontsize=9)
ax.spines[["top", "right", "left"]].set_visible(False)
finish(fig, "08_road_features.png")

# 9. Three-run runtime comparison.
pandas_runs = pd.read_csv(ROOT / "pandas_benchmark.csv")[["run", "total_seconds"]].rename(columns={"total_seconds": "Pandas"})
spark_runs = pd.read_csv(ROOT / "pyspark_benchmark.csv")[["run", "total_seconds"]].rename(columns={"total_seconds": "PySpark"})
runtime = pandas_runs.merge(spark_runs, on="run")
fig, ax = plt.subplots(figsize=(8, 5.2))
x = range(len(runtime))
width = 0.36
p1 = ax.bar([i - width/2 for i in x], runtime["Pandas"], width, label="Pandas", color=BLUE)
p2 = ax.bar([i + width/2 for i in x], runtime["PySpark"], width, label="PySpark", color=ORANGE)
ax.set_title("Pandas and PySpark End-to-End Runtime")
ax.set_xlabel("Benchmark run")
ax.set_ylabel("Runtime (seconds)")
ax.set_xticks(list(x), [str(v) for v in runtime["run"]])
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False)
ax.bar_label(p1, fmt="%.2f", padding=3, fontsize=8)
ax.bar_label(p2, fmt="%.2f", padding=3, fontsize=8)
ax.text(0.01, -0.18, "Median: Pandas 15.6320 s; PySpark 15.4447 s. Cluster provisioning excluded.", transform=ax.transAxes, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "09_runtime_comparison.png")

print(f"Created 9 charts in: {OUT}")
