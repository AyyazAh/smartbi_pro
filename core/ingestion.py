"""Ingestion: load CSV/Excel + auto-detect semantic column types."""
import re
from pathlib import Path
import pandas as pd
import chardet
from utils.logger import get_logger

log = get_logger("ingestion")

# --- Semantic hints (regex on column name) ---
DATE_HINTS     = re.compile(r"(date|time|month|year|day|dt|timestamp)", re.I)
REVENUE_HINTS  = re.compile(r"(revenue|sales|amount|price|total|income|profit|cost|value)", re.I)
ID_HINTS       = re.compile(r"(^id$|_id$|^code$|sku|serial|order_id|customer_id)", re.I)
GEO_HINTS      = re.compile(r"(region|country|city|state|area|zone|territory|location)", re.I)
CATEGORY_HINTS = re.compile(r"(category|segment|type|class|product|group|department)", re.I)


def _detect_encoding(path: str, n_bytes: int = 100_000) -> str:
    with open(path, "rb") as f:
        raw = f.read(n_bytes)
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    return enc


def read_any_file(path: str) -> dict:
    """
    Read CSV or Excel. Returns {sheet_name: dataframe}.
    CSV -> {'data': df}; Excel -> {sheet1: df, sheet2: df, ...}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    log.info(f"Reading file: {p.name} ({ext})")

    if ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        log.info(f"Excel loaded: {len(sheets)} sheet(s) -> {list(sheets.keys())}")
        return sheets

    # CSV
    enc = _detect_encoding(path)
    log.info(f"Detected encoding: {enc}")
    try:
        df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
    except Exception as e:
        log.warning(f"Auto-sep failed ({e}), falling back to comma")
        df = pd.read_csv(path, encoding=enc)

    log.info(f"CSV loaded: {len(df):,} rows x {len(df.columns)} cols")
    return {"data": df}


def infer_semantic_type(col: str, series: pd.Series) -> str:
    """Classify a column into: id / date / revenue / geo / category / numeric / text."""
    name = str(col).lower()
    n = len(series)
    nunique = series.nunique(dropna=True)

    # 1. ID: name matches AND almost-unique values
    if ID_HINTS.search(name) and n > 0 and nunique > 0.85 * n:
        return "id"

    # 2. Date: pandas dtype or parseable
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if DATE_HINTS.search(name):
        try:
            pd.to_datetime(series.dropna().head(50), errors="raise")
            return "date"
        except Exception:
            pass
    if series.dtype == object and n > 0:
        sample = series.dropna().head(20)
        if len(sample) >= 5:
            try:
                pd.to_datetime(sample, errors="raise")
                return "date"
            except Exception:
                pass

    # 3. Numeric-based hints
    is_numeric = pd.api.types.is_numeric_dtype(series)
    if is_numeric and REVENUE_HINTS.search(name):
        return "revenue"
    if GEO_HINTS.search(name):
        return "geo"
    if CATEGORY_HINTS.search(name):
        return "category"
    if is_numeric:
        return "numeric"

    return "text"


def build_schema(df: pd.DataFrame) -> dict:
    """Return {column: semantic_type}."""
    schema = {c: infer_semantic_type(c, df[c]) for c in df.columns}
    log.info(f"Schema detected: {schema}")
    return schema


__all__ = ["read_any_file", "build_schema", "infer_semantic_type"]