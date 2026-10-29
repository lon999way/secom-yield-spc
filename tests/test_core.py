"""Offline tests: synthetic data with known answers, no download needed."""
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import load, model, screen, spc

SQL = Path(__file__).resolve().parent.parent / "sql"


# ---------- load ----------
def test_parse_label_maps_minus_one_to_pass():
    assert load.parse_label('-1 "19/07/2008 11:55:00"') == (0, "2008-07-19 11:55:00")
    assert load.parse_label('1 "01/10/2008 09:05:00"') == (1, "2008-10-01 09:05:00")
    with pytest.raises(ValueError):
        load.parse_label('0 "01/10/2008 09:05:00"')


def test_parse_row_keeps_nan_as_none():
    assert load.parse_row("1.5 NaN -2") == [1.5, None, -2.0]


def test_load_rejects_mismatched_files(tmp_path):
    (tmp_path / "secom.data").write_text("1 2\n3 4\n")
    (tmp_path / "secom_labels.data").write_text('-1 "19/07/2008 11:55:00"\n')
    with pytest.raises(ValueError):
        load.load(tmp_path, tmp_path / "x.db")


# ---------- SQL ----------
@pytest.fixture
def tiny_db(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # 2008-07-20 is a Sunday, 2008-07-21 a Monday
    rows = ["1 NaN", "2 NaN", "3 5", "4 5"]
    labels = ['-1 "20/07/2008 10:00:00"', '1 "20/07/2008 11:00:00"',
              '-1 "20/07/2008 12:00:00"', '-1 "21/07/2008 11:00:00"']
    (raw / "secom.data").write_text("\n".join(rows) + "\n")
    (raw / "secom_labels.data").write_text("\n".join(labels) + "\n")
    db = tmp_path / "t.db"
    load.load(raw, db)
    con = sqlite3.connect(db)
    yield con
    con.close()


def run(con, name):
    return con.execute((SQL / name).read_text()).fetchall()


def test_daily_yield_rolling_is_volume_weighted(tiny_db):
    rows = run(tiny_db, "03_daily_yield.sql")
    assert rows[0][:4] == ("2008-07-20", 3, 1, 0.6667)
    assert rows[1][4] == 0.75  # (4 runs - 1 fail) / 4 runs, not the mean of 0.667 and 1.0


def test_weekly_yield_puts_monday_in_its_own_week(tiny_db):
    weeks = [r[0] for r in run(tiny_db, "04_weekly_yield.sql")]
    assert weeks == ["2008-07-14", "2008-07-21"]


def test_missing_and_constant_audit(tiny_db):
    buckets = dict((b, n) for b, n, _ in run(tiny_db, "01_missing_by_sensor.sql"))
    assert buckets == {"0 none": 1, "3 50% or more": 1}
    assert run(tiny_db, "02_constant_sensors.sql") == [(1,)]  # sensor 1 has only the value 5


# ---------- SPC ----------
def test_imr_limits_use_moving_range():
    lim = spc.imr_limits([10, 12, 10, 12])
    assert lim["center"] == 11
    assert lim["sigma"] == pytest.approx(2 / 1.128)


def test_p_chart_limits_widen_for_small_days():
    pc = spc.p_chart([1, 10], [10, 100])
    assert pc["center"] == pytest.approx(11 / 110)
    assert pc["ucl"][0] > pc["ucl"][1]
    assert np.all(pc["lcl"] >= 0)


@pytest.mark.parametrize("z, rule, at", [
    ([0, 0, 3.5], 1, 2),
    ([0, 2.5, 0.5, 2.5], 2, 3),
    ([1.5, 1.5, 0.2, 1.5, 1.5], 3, 4),
    ([0.3] * 8, 4, 7),
    ([-0.3] * 8, 4, 7),
])
def test_western_electric_rules_fire(z, rule, at):
    out = spc.western_electric(z)
    assert rule in out[at]
    assert all(rule not in s for s in out[:at])


def test_western_electric_requires_same_side():
    out = spc.western_electric([2.5, -2.5, 2.5 * 0, -1.5, 1.5, -1.5, 1.5, -1.5])
    assert not any({2, 3, 4} & s for s in out)


# ---------- screen ----------
def test_auc_matches_sklearn_and_drops_nan():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    x = rng.normal(size=200) + y
    a, _ = screen.auc(x, y)
    assert a == pytest.approx(roc_auc_score(y, x))
    x2 = x.copy()
    x2[:10] = np.nan
    assert screen.auc(x2, y)[0] == pytest.approx(roc_auc_score(y[10:], x[10:]))


def test_within_block_auc_removes_pure_time_trend():
    # fails are common early and rare late; the sensor only drifts with time, nothing else
    rng = np.random.default_rng(2)
    n = 1600
    t = np.arange(n)
    y = (rng.random(n) < np.where(t < n // 2, 0.2, 0.03)).astype(int)
    x = t + rng.normal(scale=5, size=n)
    blocks = t * 8 // n
    pooled, _ = screen.auc(x, y)
    within = screen.within_block_auc(x, y, blocks)
    assert pooled < 0.35            # looks strongly "related" when pooled
    assert abs(within - 0.5) < 0.07  # nothing left once time is held fixed


def test_permute_within_keeps_fail_count_per_block():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 100)
    blocks = np.arange(100) // 25
    yp = screen.permute_within(y, blocks, rng)
    for b in range(4):
        assert yp[blocks == b].sum() == y[blocks == b].sum()


# ---------- model ----------
def test_model_learns_signal_and_not_noise():
    rng = np.random.default_rng(4)
    n = 600
    y = (rng.random(n) < 0.1).astype(int)
    noise = rng.normal(size=(n, 30))
    signal = noise.copy()
    signal[:, 0] += 2.5 * y
    assert model.evaluate(signal, y, k=5)["cv10_ber"] < 0.2
    assert model.evaluate(noise, y, k=5)["cv10_ber"] > 0.35
