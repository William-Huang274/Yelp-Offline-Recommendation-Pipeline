import os
from pathlib import Path
from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "Yelp JSON" / "yelp_dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "parquet"
OVERWRITE_EXISTING = False

DATASETS = ["business", "checkin", "tip", "user", "review"]

def find_hadoop_bin() -> Path | None:
    candidates: list[Path] = []

    project_bin = PROJECT_ROOT / "tools" / "hadoop" / "bin"
    candidates.append(project_bin)

    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home:
        candidates.append(Path(hadoop_home) / "bin")

    for bin_dir in candidates:
        if (bin_dir / "winutils.exe").exists() and (bin_dir / "hadoop.dll").exists():
            return bin_dir
    return None

def configure_windows_hadoop() -> Path:
    bin_dir = find_hadoop_bin()
    if bin_dir is None:
        raise RuntimeError(
            "Missing winutils.exe / hadoop.dll.\n"
            "Please put them under one of these paths:\n"
            f"1) {PROJECT_ROOT / 'tools' / 'hadoop' / 'bin'}\n"
            "2) %HADOOP_HOME%\\bin\n"
            "Current Spark(Hadoop) version is 4.1.1 / 3.4.2.\n"
            "Prefer matching binaries; if you cannot find 3.4.2, try 3.4.0."
        )

    hadoop_home = bin_dir.parent
    os.environ["HADOOP_HOME"] = str(hadoop_home)
    os.environ["hadoop.home.dir"] = str(hadoop_home)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    return bin_dir

def build_spark() -> SparkSession:
    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", driver_mem)
    local_master = os.environ.get("SPARK_LOCAL_MASTER", "local[2]")

    builder = (
        SparkSession.builder
        .appName("yelp-json-to-parquet")
        .master(local_master)
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .config("spark.default.parallelism", "4")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.cleanup.skipped", "true")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.LocalFs")
        .config("spark.hadoop.fs.file.impl.disable.cache", "true")
    )

    return builder.getOrCreate()


def target_output_partitions(name: str) -> int:
    if name == "review":
        return 12
    if name == "user":
        return 8
    return 2

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    bin_dir = configure_windows_hadoop()
    print(f"[INFO] Using Hadoop native binaries: {bin_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    for name in DATASETS:
        src = INPUT_DIR / f"yelp_academic_dataset_{name}.json"
        dst = OUTPUT_DIR / f"yelp_academic_dataset_{name}"

        if not src.exists():
            print(f"[SKIP] missing file: {src}")
            continue
        if dst.exists() and not OVERWRITE_EXISTING:
            print(f"[SKIP] output exists: {dst}")
            continue

        print(f"[READ ] {src}")
        df = spark.read.option("multiLine", "false").json(str(src))

        current_parts = df.rdd.getNumPartitions()
        target_parts = target_output_partitions(name)
        if target_parts < current_parts:
            out_df = df.coalesce(target_parts)
            part_msg = f"coalesce {current_parts}->{target_parts}"
        else:
            out_df = df
            part_msg = f"keep {current_parts}"

        print(f"[WRITE] {dst} ({part_msg})")
        dst_path = dst.resolve().as_posix()
        (
            out_df.write
            .mode("overwrite")
            .option("maxRecordsPerFile", 300000)
            .parquet(dst_path)
        )

        print(f"[DONE ] {name}")
        spark.catalog.clearCache()

    spark.stop()
    print(f"All done. Output folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
