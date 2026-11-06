from pathlib import Path

from scripts import cls_pos_recency_e79_v1 as runner


def test_recovery_targets_official_classification_test_csv() -> None:
    relative = runner.OFFICIAL_TEST_REL
    assert relative.as_posix() == "data/raw/classification/Test.csv"
    assert relative.suffix.lower() == ".csv"
    assert runner.EXPECTED_TEST_ROWS == 29_421
    assert runner.OFFICIAL_TEST_SHA256 == "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"

