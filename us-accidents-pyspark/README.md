# US Accidents Analysis with PySpark

This repository contains the code, aggregated results, charts, and report for an IST3134 group assignment analysing the **US Accidents (2016–2023)** dataset with PySpark on Amazon EMR and a chunked Pandas benchmark.

## Dataset

- Source: [US Accidents (2016–2023) on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- Downloaded file: `US_Accidents_March23.csv`
- Size: 3,058,183,727 bytes (approximately 3.06 GB)
- Records: 7,728,394
- Columns: 46
- Observed date range: 14 January 2016 to 31 March 2023

The raw dataset is not included in this repository because of its size and distribution conditions. Download it from Kaggle and store it separately.

## Computing environment

- Amazon EMR release: `emr-spark-8.0.0`
- Spark: 4.0.2 (`4.0.2-amzn-0` in the benchmark output)
- Resource manager: Hadoop YARN
- Region: US East (N. Virginia)
- Cluster: one `m5.xlarge` primary and one `m5.xlarge` core node
- Persistent storage: Amazon S3
- Pandas benchmark: Apple M1 Mac, 8 GB RAM, Python 3.12.13, Pandas 2.2.3

## Repository structure

```text
scripts/
  us_accidents_prepare.py     Data profiling, timestamp parsing, validation, and Parquet preparation
  us_accidents_analysis.py    Geographic, temporal, severity, weather, and road-feature analyses
  pandas_benchmark.py         Three-run chunked Pandas benchmark
  pyspark_benchmark.py        Three-run PySpark benchmark on Amazon EMR
  create_charts.py            Generates the report charts from the result CSV files

results/                      Validated aggregate CSV outputs
charts/                       Report-ready PNG charts
report/                       Current Word report
```

## Workflow

1. Upload the raw CSV to Amazon S3.
2. Run `us_accidents_prepare.py` on EMR to validate the data and create Year-partitioned Parquet.
3. Run `us_accidents_analysis.py` to create the aggregate analytical outputs.
4. Run `pandas_benchmark.py` locally and `pyspark_benchmark.py` on EMR.
5. Place the aggregate CSV files in `results/` and run `create_charts.py` to regenerate the figures.

Regenerate the charts from the repository root:

```bash
python3 scripts/create_charts.py
```

## Main benchmark result

| Framework | Median runtime |
|---|---:|
| Pandas | 15.6320 seconds |
| PySpark | 15.4447 seconds |

PySpark was approximately 1.2% faster, but the 0.1873-second difference was not substantial. The result is a platform-level comparison between a chunked single-machine Pandas implementation and a small Amazon EMR deployment.

## Privacy and security

This repository intentionally excludes:

- The raw 3.06 GB dataset and processed Parquet files
- AWS access keys, tokens, account identifiers, cluster identifiers, VPC IDs, and subnet IDs
- EMR/S3 logs that may contain environment identifiers

## Licence and attribution

The dataset is used for academic, non-commercial analysis. Refer to the Kaggle data page and the dataset creators' publications for the applicable attribution and reuse conditions.
