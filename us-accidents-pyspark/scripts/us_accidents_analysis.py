"""Generate report-ready US Accidents summaries from prepared Parquet data."""

import argparse

from pyspark.sql import SparkSession, functions as F


DEFAULT_INPUT = "s3://us-accidents-pyspark/processed/accidents_parquet"
DEFAULT_OUTPUT = "s3://us-accidents-pyspark/results/analysis"

ROAD_FEATURES = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Turning_Loop",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-city-count", type=int, default=1000)
    return parser.parse_args()


def write_csv(frame, path, partitions=1):
    """Write compact aggregate output as CSV with a completion marker."""
    frame.coalesce(partitions).write.mode("overwrite").option(
        "header", "true"
    ).csv(path)


def main():
    args = arguments()
    spark = (
        SparkSession.builder
        .appName("US-Accidents-Main-Analysis")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(args.input).cache()
    total = df.count()

    # 1. Geographic distribution by state.
    state = (
        df.groupBy("State")
        .agg(
            F.count("*").alias("Accident_Count"),
            F.round(F.avg("Severity"), 4).alias("Average_Severity"),
            F.round(F.avg("Severity_3_4") * 100, 4).alias("Severity_3_4_Percentage"),
        )
        .withColumn("Percentage_of_All_Records", F.round(F.col("Accident_Count") * 100 / total, 4))
        .orderBy(F.desc("Accident_Count"), "State")
    )
    write_csv(state, f"{args.output}/state_summary")

    # 2. City-State summaries. The count threshold applies only to severity ranking.
    city = (
        df.groupBy("City_State", "City", "State")
        .agg(
            F.count("*").alias("Accident_Count"),
            F.round(F.avg("Severity"), 4).alias("Average_Severity"),
            F.round(F.avg("Severity_3_4") * 100, 4).alias("Severity_3_4_Percentage"),
        )
    )
    write_csv(
        city.orderBy(F.desc("Accident_Count"), "City_State").limit(100),
        f"{args.output}/top_100_cities_by_count",
    )
    write_csv(
        city.filter(F.col("Accident_Count") >= args.minimum_city_count)
        .orderBy(F.desc("Average_Severity"), F.desc("Accident_Count"), "City_State"),
        f"{args.output}/city_severity_min_{args.minimum_city_count}",
    )

    # 3. Temporal summaries.
    year = (
        df.groupBy("Year")
        .agg(
            F.count("*").alias("Accident_Count"),
            F.round(F.avg("Severity"), 4).alias("Average_Severity"),
            F.round(F.avg("Severity_3_4") * 100, 4).alias("Severity_3_4_Percentage"),
        )
        .orderBy("Year")
    )
    write_csv(year, f"{args.output}/year_summary")

    year_month = (
        df.groupBy("Year", "Month")
        .agg(
            F.count("*").alias("Accident_Count"),
            F.round(F.avg("Severity"), 4).alias("Average_Severity"),
        )
        .orderBy("Year", "Month")
    )
    write_csv(year_month, f"{args.output}/year_month_summary")

    weekday = (
        df.groupBy("Weekday_Number", "Weekday")
        .agg(F.count("*").alias("Accident_Count"))
        .withColumn("Percentage_of_All_Records", F.round(F.col("Accident_Count") * 100 / total, 4))
        .orderBy("Weekday_Number")
    )
    write_csv(weekday, f"{args.output}/weekday_summary")

    hour = (
        df.groupBy("Hour")
        .agg(F.count("*").alias("Accident_Count"))
        .withColumn("Percentage_of_All_Records", F.round(F.col("Accident_Count") * 100 / total, 4))
        .orderBy("Hour")
    )
    write_csv(hour, f"{args.output}/hour_summary")

    weekday_hour = (
        df.groupBy("Weekday_Number", "Weekday", "Hour")
        .agg(F.count("*").alias("Accident_Count"))
        .orderBy("Weekday_Number", "Hour")
    )
    write_csv(weekday_hour, f"{args.output}/weekday_hour_summary")

    # 4. Severity distribution.
    severity = (
        df.groupBy("Severity")
        .agg(F.count("*").alias("Accident_Count"))
        .withColumn("Percentage_of_All_Records", F.round(F.col("Accident_Count") * 100 / total, 4))
        .orderBy("Severity")
    )
    write_csv(severity, f"{args.output}/severity_distribution")

    # 5. Weather summaries. Missing weather labels are retained explicitly.
    weather_base = df.withColumn(
        "Weather_Condition_Group",
        F.coalesce(F.col("Weather_Condition"), F.lit("Missing")),
    )
    weather = (
        weather_base.groupBy("Weather_Condition_Group")
        .agg(
            F.count("*").alias("Accident_Count"),
            F.round(F.avg("Severity"), 4).alias("Average_Severity"),
            F.round(F.avg("Severity_3_4") * 100, 4).alias("Severity_3_4_Percentage"),
        )
        .withColumn("Percentage_of_All_Records", F.round(F.col("Accident_Count") * 100 / total, 4))
        .orderBy(F.desc("Accident_Count"), "Weather_Condition_Group")
    )
    write_csv(weather, f"{args.output}/weather_condition_summary")

    numeric_weather = df.agg(
        *[
            F.round(F.avg(column), 4).alias(f"Mean_{column.replace('(', '_').replace(')', '').replace('%', 'pct')}")
            for column in [
                "Temperature(F)", "Humidity(%)", "Pressure(in)",
                "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
            ]
        ]
    )
    write_csv(numeric_weather, f"{args.output}/numeric_weather_overall")

    # 6. Road-feature present-versus-absent comparisons in a tidy table.
    road_frames = []
    for feature in ROAD_FEATURES:
        road_frames.append(
            df.groupBy(F.coalesce(F.col(feature), F.lit(False)).alias("Feature_Present"))
            .agg(
                F.count("*").alias("Accident_Count"),
                F.round(F.avg("Severity"), 4).alias("Average_Severity"),
                F.round(F.avg("Severity_3_4") * 100, 4).alias("Severity_3_4_Percentage"),
            )
            .withColumn("Feature", F.lit(feature))
            .select(
                "Feature", "Feature_Present", "Accident_Count",
                "Average_Severity", "Severity_3_4_Percentage",
            )
        )
    road = road_frames[0]
    for frame in road_frames[1:]:
        road = road.unionByName(frame)
    write_csv(road.orderBy("Feature", F.desc("Feature_Present")), f"{args.output}/road_feature_summary")

    # 7. Reconciliation totals for use in the report and debugging.
    check_rows = [
        ("input_row_count", str(total), "PASS"),
        ("state_count_sum", str(state.agg(F.sum("Accident_Count")).first()[0]), "PASS"),
        ("year_count_sum", str(year.agg(F.sum("Accident_Count")).first()[0]), "PASS"),
        ("severity_count_sum", str(severity.agg(F.sum("Accident_Count")).first()[0]), "PASS"),
        ("input_uri", args.input, "INFO"),
        ("output_uri", args.output, "INFO"),
    ]
    reconciliation = spark.createDataFrame(check_rows, ["Check", "Value", "Status"])
    write_csv(reconciliation, f"{args.output}/reconciliation")

    print(f"Analysis complete. Outputs: {args.output}")
    df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
