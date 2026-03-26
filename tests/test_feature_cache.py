"""Unit tests for feature caching utilities."""

import pandas as pd

from virgo_trader.data import feature_cache


def test_load_cached_features_for_range_falls_back_to_superset(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_cache, "FEATURE_CACHE_DIR", tmp_path)

    index = pd.date_range("2022-01-01", periods=10, freq="D")
    superset_df = pd.DataFrame({"value": range(10)}, index=index)

    superset_path = feature_cache.feature_cache_path(
        symbol="AAA.SH",
        start_date="20220101",
        end_date="20220110",
        use_calendar_features=False,
    )
    feature_cache.save_cached_features(superset_path, superset_df)

    loaded = feature_cache.load_cached_features_for_range(
        symbol="AAA.SH",
        start_date="20220103",
        end_date="20220105",
        use_calendar_features=False,
    )
    assert loaded is not None
    assert len(loaded) == 3
    assert loaded.index.min().date().isoformat() == "2022-01-03"
    assert loaded.index.max().date().isoformat() == "2022-01-05"

    expected_slice_path = feature_cache.feature_cache_path(
        symbol="AAA.SH",
        start_date="20220103",
        end_date="20220105",
        use_calendar_features=False,
    )
    assert expected_slice_path.exists()
