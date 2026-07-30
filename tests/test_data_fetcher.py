"""Tests for the data fetcher and validation layer."""

import pandas as pd
import pytest

from src.data.fetcher import (
    _flatten_columns,
    _validate_schema,
    detect_close_column,
)


class TestFlattenColumns:
    def test_multiindex_flattened(self):
        arrays = [["Close", "Open"], ["AAPL", "AAPL"]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[1, 2], [3, 4]], columns=index)
        result = _flatten_columns(df)
        assert not isinstance(result.columns, pd.MultiIndex)
        assert len(result.columns) == 2

    def test_flat_columns_unchanged(self):
        df = pd.DataFrame({"Close": [1, 2], "Open": [3, 4]})
        result = _flatten_columns(df)
        assert list(result.columns) == ["Close", "Open"]


class TestValidateSchema:
    def test_valid_schema(self):
        df = pd.DataFrame({
            "Open": [1], "High": [2], "Low": [3], "Close": [4], "Volume": [5]
        })
        _validate_schema(df, "TEST")  # should not raise

    def test_valid_schema_with_ticker_suffix(self):
        """Columns like 'Close_AAPL' (from flattened MultiIndex) should pass."""
        df = pd.DataFrame({
            "Open_AAPL": [1], "High_AAPL": [2], "Low_AAPL": [3],
            "Close_AAPL": [4], "Volume_AAPL": [5], "Adj Close_AAPL": [6],
        })
        _validate_schema(df, "AAPL")  # should not raise

    def test_missing_column_raises(self):
        df = pd.DataFrame({"Open": [1], "High": [2], "Low": [3]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_schema(df, "TEST")


class TestDetectCloseColumn:
    def test_standard(self):
        df = pd.DataFrame({"Open": [1], "Close": [2]})
        assert detect_close_column(df) == "Close"

    def test_case_insensitive(self):
        df = pd.DataFrame({"open": [1], "close_AAPL": [2]})
        assert "close" in detect_close_column(df).lower()

    def test_missing_raises(self):
        df = pd.DataFrame({"Open": [1], "High": [2]})
        with pytest.raises(ValueError, match="Close"):
            detect_close_column(df)
