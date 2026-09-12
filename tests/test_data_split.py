"""Tests kiểm tra phân chia dữ liệu độc lập theo đối tượng (Subject-Level Split) chống Data Leakage."""

import pandas as pd
import pytest

from scripts.prepare_data import split_by_subject


def test_subject_split_disjointness():
    """Đảm bảo các tập train, validation và test có danh sách subject rời rạc tuyệt đối."""
    subjects = [f"subject_{i:03d}" for i in range(1, 11)]
    df = pd.DataFrame({
        "sample_id": [f"s_{i}" for i in range(100)],
        "subject_id": [subjects[i % len(subjects)] for i in range(100)],
        "session_id": ["s1"] * 100,
        "gesture": ["Fist"] * 100,
        "handedness": ["Right"] * 100,
    })

    splits = split_by_subject(df, val_ratio=0.2, test_ratio=0.2, seed=42)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert train_set.isdisjoint(val_set), "Data Leakage: Train và Val có chung Subject!"
    assert train_set.isdisjoint(test_set), "Data Leakage: Train và Test có chung Subject!"
    assert val_set.isdisjoint(test_set), "Data Leakage: Val và Test có chung Subject!"
    assert len(train_set | val_set | test_set) == len(subjects)


def test_subject_split_requires_minimum_subjects():
    df = pd.DataFrame({
        "subject_id": ["sub_01", "sub_02"],
    })
    with pytest.raises(ValueError, match="Cần ít nhất 3 subjects"):
        split_by_subject(df)
