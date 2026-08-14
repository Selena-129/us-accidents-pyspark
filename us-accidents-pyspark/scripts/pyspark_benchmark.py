"""Three-run EMR PySpark benchmark equivalent to pandas_benchmark.py.

Each timed run reads the raw CSV, validates State/Severity/Start_Time,
calculates state, year, and severity counts in one grouping-sets aggregation,
and writes the result to S3. 
"""

import argparse
import json
import statistics
import time

from pyspark.sql import SparkSession, functions as F, types as T


EXPECTED_ROWS = 7_728_394
DEFAULT_INPUT = "s3://us-accidents-pyspark/raw/US_Accidents_March23.csv"
DEFAULT_OUTPUT = "s3://us-accidents-pyspark/results/benchmark/pyspark"


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
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def parsed_timestamp():
    return F.coalesce(
        F.try_to_timestamp("Start_Time", F.lit("yyyy-MM-dd HH:mm:ss.SSSSSSSSS")),
        F.try_to_timestamp("Start_Time", F.lit("yyyy-MM-dd HH:mm:ss")),
    )


def benchmark_run(spark, input_uri, output_uri, run_number):
    spark.catalog.clearCache()
    start = time.perf_counter()

    source = (
        spark.read.option("header", "true").option("mode", "PERMISSIVE")
        .schema(SCHEMA).csv(input_uri)
        .select("State", "Severity", "Start_Time")
        .withColumn("Parsed_Start_Time", parsed_timestamp())
        .filter(F.col("State").isNotNull())
        .filter(F.col("Severity").between(1, 4))
        .filter(F.col("Parsed_Start_Time").isNotNull())
        .withColumn("Year", F.year("Parsed_Start_Time"))
        .select("State", "Severity", "Year")
    )
    source.createOrReplaceTempView("benchmark_source")

    # GROUPING SETS calculates all three summaries through one Spark query.
    result = spark.sql(
        """
        SELECT
          CASE
            WHEN grouping(State) = 0 THEN 'State'
            WHEN grouping(Year) = 0 THEN 'Year'
            ELSE 'Severity'
          END AS Dimension,
          COALESCE(State, CAST(Year AS STRING), CAST(Severity AS STRING)) AS Value,
          COUNT(*) AS Accident_Count
        FROM benchmark_source
        GROUP BY GROUPING SETS ((State), (Year), (Severity))
        """
    )

    run_uri = f"{output_uri}/run_{run_number}/counts"
    result.coalesce(1).write.mode("overwrite").option("header", "true").csv(run_uri)
    total_seconds = time.perf_counter() - start

    # Read the small persisted result to reconcile every aggregation dimension.
    saved = spark.read.option("header", "true").option("inferSchema", "true").csv(run_uri)
    totals = {
        row["Dimension"]: int(row["Total"])
        for row in saved.groupBy("Dimension").agg(F.sum("Accident_Count").alias("Total")).collect()
    }
    status = "PASS" if all(
        totals.get(key) == EXPECTED_ROWS for key in ("State", "Year", "Severity")
    ) else "FAIL"
    return {
        "run": run_number,
        "total_seconds": round(total_seconds, 4),
        "state_total": totals.get("State"),
        "year_total": totals.get("Year"),
        "severity_total": totals.get("Severity"),
        "reconciliation_status": status,
        "output_uri": run_uri,
    }


def main():
    args = arguments()
    spark = (
        SparkSession.builder.appName("US-Accidents-PySpark-Benchmark")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    results = []
    for run_number in range(1, args.runs + 1):
        print(f"Starting PySpark run {run_number} of {args.runs}...")
        result = benchmark_run(spark, args.input, args.output, run_number)
        results.append(result)
        print(json.dumps(result, indent=2))

    median_seconds = round(statistics.median(r["total_seconds"] for r in results), 4)
    metadata = {
        "framework": "PySpark",
        "spark_version": spark.version,
        "spark_master": spark.sparkContext.master,
        "default_parallelism": spark.sparkContext.defaultParallelism,
        "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        "input_uri": args.input,
        "runs": args.runs,
        "median_total_seconds": median_seconds,
        "all_runs_reconciled": all(r["reconciliation_status"] == "PASS" for r in results),
        "timing_scope": "raw CSV read, validation, three grouped counts, and S3 output write",
        "cluster_provisioning_time_included": False,
    }

    rows = [
        (
            r["run"], r["total_seconds"], r["state_total"], r["year_total"],
            r["severity_total"], r["reconciliation_status"], r["output_uri"],
        )
        for r in results
    ]
    spark.createDataFrame(
        rows,
        ["run", "total_seconds", "state_total", "year_total", "severity_total", "reconciliation_status", "output_uri"],
    ).coalesce(1).write.mode("overwrite").option("header", "true").csv(
        f"{args.output}/benchmark_runs"
    )
    spark.createDataFrame(
        [(key, json.dumps(value) if isinstance(value, (bool, list, dict)) else str(value)) for key, value in metadata.items()],
        ["metric", "value"],
    ).coalesce(1).write.mode("overwrite").option("header", "true").csv(
        f"{args.output}/benchmark_summary"
    )

    print(json.dumps(metadata, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
