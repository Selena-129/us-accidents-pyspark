"""Profile and prepare the US Accidents dataset on Amazon EMR.

Run with spark-submit under Hadoop YARN. The raw CSV is never modified.
"""

import argparse

from pyspark.sql import SparkSession, functions as F, types as T


DEFAULT_INPUT = "s3://us-accidents-pyspark/raw/US_Accidents_March23.csv"
DEFAULT_OUTPUT = "s3://us-accidents-pyspark"


SCHEMA = T.StructType([
    T.StructField("ID", T.StringType(), True),
    T.StructField("Source", T.StringType(), True),
    T.StructField("Severity", T.IntegerType(), True),
    T.StructField("Start_Time", T.StringType(), True),
    T.StructField("End_Time", T.StringType(), True),
    T.StructField("Start_Lat", T.DoubleType(), True),
    T.StructField("Start_Lng", T.DoubleType(), True),
    T.StructField("End_Lat", T.DoubleType(), True),
    T.StructField("End_Lng", T.DoubleType(), True),
    T.StructField("Distance(mi)", T.DoubleType(), True),
    T.StructField("Description", T.StringType(), True),
    T.StructField("Street", T.StringType(), True),
    T.StructField("City", T.StringType(), True),
    T.StructField("County", T.StringType(), True),
    T.StructField("State", T.StringType(), True),
    T.StructField("Zipcode", T.StringType(), True),
    T.StructField("Country", T.StringType(), True),
    T.StructField("Timezone", T.StringType(), True),
    T.StructField("Airport_Code", T.StringType(), True),
    T.StructField("Weather_Timestamp", T.StringType(), True),
    T.StructField("Temperature(F)", T.DoubleType(), True),
    T.StructField("Wind_Chill(F)", T.DoubleType(), True),
    T.StructField("Humidity(%)", T.DoubleType(), True),
    T.StructField("Pressure(in)", T.DoubleType(), True),
    T.StructField("Visibility(mi)", T.DoubleType(), True),
    T.StructField("Wind_Direction", T.StringType(), True),
    T.StructField("Wind_Speed(mph)", T.DoubleType(), True),
    T.StructField("Precipitation(in)", T.DoubleType(), True),
    T.StructField("Weather_Condition", T.StringType(), True),
    T.StructField("Amenity", T.BooleanType(), True),
    T.StructField("Bump", T.BooleanType(), True),
    T.StructField("Crossing", T.BooleanType(), True),
    T.StructField("Give_Way", T.BooleanType(), True),
    T.StructField("Junction", T.BooleanType(), True),
    T.StructField("No_Exit", T.BooleanType(), True),
    T.StructField("Railway", T.BooleanType(), True),
    T.StructField("Roundabout", T.BooleanType(), True),
    T.StructField("Station", T.BooleanType(), True),
    T.StructField("Stop", T.BooleanType(), True),
    T.StructField("Traffic_Calming", T.BooleanType(), True),
    T.StructField("Traffic_Signal", T.BooleanType(), True),
    T.StructField("Turning_Loop", T.BooleanType(), True),
    T.StructField("Sunrise_Sunset", T.StringType(), True),
    T.StructField("Civil_Twilight", T.StringType(), True),
    T.StructField("Nautical_Twilight", T.StringType(), True),
    T.StructField("Astronomical_Twilight", T.StringType(), True),
])


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = arguments()
    spark = (
        SparkSession.builder
        .appName("US-Accidents-Prepare")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    source = (
        spark.read
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .schema(SCHEMA)
        .csv(args.input)
    )

    # The file contains both whole-second and nine-digit fractional-second
    # timestamps. Parse both forms explicitly so valid records are not lost.
    def parse_timestamp(column):
        return F.coalesce(
            F.try_to_timestamp(F.col(column), F.lit("yyyy-MM-dd HH:mm:ss.SSSSSSSSS")),
            F.try_to_timestamp(F.col(column), F.lit("yyyy-MM-dd HH:mm:ss")),
        )

    raw = (
        source
        .withColumn("Start_Time", parse_timestamp("Start_Time"))
        .withColumn("End_Time", parse_timestamp("End_Time"))
        .withColumn("Weather_Timestamp", parse_timestamp("Weather_Timestamp"))
    )

    # Compute core quality statistics and all null counts in a single scan.
    safe_aliases = {column: f"null_{index:02d}" for index, column in enumerate(raw.columns)}
    expressions = [
        F.count(F.lit(1)).alias("row_count"),
        F.countDistinct("ID").alias("distinct_id_count"),
        F.sum(F.when(F.col("ID").isNull(), 1).otherwise(0)).alias("null_id_count"),
        F.sum(F.when(~F.col("Severity").between(1, 4), 1).otherwise(0)).alias("invalid_severity_count"),
        F.min("Start_Time").alias("minimum_start_time"),
        F.max("Start_Time").alias("maximum_start_time"),
    ]
    expressions.extend(
        F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(alias)
        for column, alias in safe_aliases.items()
    )
    profile = raw.agg(*expressions).first().asDict()

    row_count = profile["row_count"]
    distinct_id_count = profile["distinct_id_count"]
    summary_rows = [
        ("input_uri", args.input),
        ("row_count", str(row_count)),
        ("column_count", str(len(raw.columns))),
        ("distinct_id_count", str(distinct_id_count)),
        ("duplicate_or_null_id_difference", str(row_count - distinct_id_count)),
        ("null_id_count", str(profile["null_id_count"])),
        ("invalid_severity_count", str(profile["invalid_severity_count"] or 0)),
        ("minimum_start_time", str(profile["minimum_start_time"])),
        ("maximum_start_time", str(profile["maximum_start_time"])),
    ]
    spark.createDataFrame(summary_rows, ["metric", "value"]).coalesce(1).write.mode(
        "overwrite"
    ).option("header", "true").csv(f"{args.output_root}/results/data_quality/summary")

    missing_rows = []
    for column, alias in safe_aliases.items():
        missing_count = int(profile[alias] or 0)
        missing_rows.append(
            (column, missing_count, round(100.0 * missing_count / row_count, 4))
        )
    spark.createDataFrame(
        missing_rows, ["column", "missing_count", "missing_percentage"]
    ).orderBy(F.desc("missing_percentage")).coalesce(1).write.mode("overwrite").option(
        "header", "true"
    ).csv(f"{args.output_root}/results/data_quality/missing_values")

    # Apply only disclosed, deterministic row-level rules.
    cleaned = (
        raw.dropDuplicates(["ID"])
        .filter(F.col("ID").isNotNull())
        .filter(F.col("Start_Time").isNotNull())
        .filter(F.col("State").isNotNull())
        .filter(F.col("Severity").between(1, 4))
        .withColumn("City", F.coalesce(F.col("City"), F.lit("Unknown")))
        .withColumn("City_State", F.concat_ws(", ", F.col("City"), F.col("State")))
        .withColumn("Year", F.year("Start_Time"))
        .withColumn("Month", F.month("Start_Time"))
        .withColumn("Weekday", F.date_format("Start_Time", "EEEE"))
        .withColumn("Weekday_Number", F.dayofweek("Start_Time"))
        .withColumn("Hour", F.hour("Start_Time"))
        .withColumn("Severity_3_4", F.when(F.col("Severity") >= 3, 1).otherwise(0))
    )

    processed_uri = f"{args.output_root}/processed/accidents_parquet"
    cleaned.write.mode("overwrite").partitionBy("Year").parquet(processed_uri)

    # Validate the persisted artifact rather than triggering the CSV pipeline again.
    persisted = spark.read.parquet(processed_uri)
    post_rows = persisted.count()
    validation_rows = [
        ("cleaned_row_count", str(post_rows)),
        ("rows_removed", str(row_count - post_rows)),
        ("processed_uri", processed_uri),
        ("status", "PASS" if post_rows <= row_count else "FAIL"),
    ]
    spark.createDataFrame(validation_rows, ["check", "value"]).coalesce(1).write.mode(
        "overwrite"
    ).option("header", "true").csv(f"{args.output_root}/results/data_quality/validation")

    print(f"Preparation complete. Processed data: {processed_uri}")
    spark.stop()


if __name__ == "__main__":
    main()
