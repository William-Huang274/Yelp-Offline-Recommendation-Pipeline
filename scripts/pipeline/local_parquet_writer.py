from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def _field_type_to_arrow(data_type: Any) -> pa.DataType:
    if isinstance(data_type, BooleanType):
        return pa.bool_()
    if isinstance(data_type, ByteType):
        return pa.int8()
    if isinstance(data_type, ShortType):
        return pa.int16()
    if isinstance(data_type, IntegerType):
        return pa.int32()
    if isinstance(data_type, LongType):
        return pa.int64()
    if isinstance(data_type, FloatType):
        return pa.float32()
    if isinstance(data_type, DoubleType):
        return pa.float64()
    if isinstance(data_type, StringType):
        return pa.string()
    if isinstance(data_type, BinaryType):
        return pa.binary()
    if isinstance(data_type, DateType):
        return pa.date32()
    if isinstance(data_type, TimestampType):
        return pa.timestamp("us")
    if isinstance(data_type, DecimalType):
        return pa.decimal128(int(data_type.precision), int(data_type.scale))
    if isinstance(data_type, ArrayType):
        return pa.list_(_field_type_to_arrow(data_type.elementType))
    if isinstance(data_type, StructType):
        return pa.struct([_struct_field_to_arrow(field) for field in data_type.fields])
    if isinstance(data_type, MapType):
        return pa.map_(_field_type_to_arrow(data_type.keyType), _field_type_to_arrow(data_type.valueType))
    raise TypeError(f"unsupported spark type for local parquet writer: {data_type}")


def _struct_field_to_arrow(field: StructField) -> pa.Field:
    return pa.field(field.name, _field_type_to_arrow(field.dataType), nullable=bool(field.nullable))


def spark_schema_to_arrow(schema: StructType) -> pa.Schema:
    return pa.schema([_struct_field_to_arrow(field) for field in schema.fields])


def _prepare_row(row: dict[str, Any], schema: StructType) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in schema.fields:
        value = row.get(field.name)
        if value is None:
            out[field.name] = None
        elif isinstance(field.dataType, ArrayType):
            out[field.name] = list(value)
        else:
            out[field.name] = value
    return out


def write_spark_df_to_parquet_dir(
    df: DataFrame,
    output_dir: str | Path,
    *,
    chunk_rows: int = 50000,
    compression: str = "snappy",
) -> dict[str, Any]:
    target_dir = Path(output_dir)
    parent_dir = target_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = parent_dir / f'.{target_dir.name}.tmp_{uuid.uuid4().hex}'
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    schema = df.schema
    arrow_schema = spark_schema_to_arrow(schema)
    chunk_rows = max(1, int(chunk_rows))
    buffered: list[dict[str, Any]] = []
    file_count = 0
    row_count = 0

    def flush() -> None:
        nonlocal buffered, file_count, row_count
        if not buffered:
            return
        file_path = temp_dir / f'part-{file_count:05d}.parquet'
        table = pa.Table.from_pylist(buffered, schema=arrow_schema)
        pq.write_table(
            table,
            file_path,
            compression=compression,
            coerce_timestamps="us",
            allow_truncated_timestamps=True,
        )
        row_count += len(buffered)
        file_count += 1
        buffered = []

    for row in df.toLocalIterator(prefetchPartitions=False):
        buffered.append(_prepare_row(row.asDict(recursive=False), schema))
        if len(buffered) >= chunk_rows:
            flush()

    if buffered or file_count == 0:
        flush()
    (temp_dir / '_SUCCESS').write_text('', encoding='utf-8')

    shutil.rmtree(target_dir, ignore_errors=True)
    temp_dir.replace(target_dir)
    return {
        'output_dir': str(target_dir),
        'row_count': int(row_count),
        'file_count': int(file_count),
        'chunk_rows': int(chunk_rows),
        'compression': str(compression),
    }
