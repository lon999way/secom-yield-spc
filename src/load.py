"""Load the raw UCI SECOM files into SQLite.

Tables
  runs(run_id, ts, fail)                 one row per production entity; fail = 1, pass = 0
  measurements(run_id, sensor, value)    long format, value is NULL where the raw file has NaN

Usage: python -m src.load            # reads data/raw/, writes data/secom.db
"""
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "secom.db"

SCHEMA = """
DROP TABLE IF EXISTS runs;
DROP TABLE IF EXISTS measurements;
CREATE TABLE runs (run_id INTEGER PRIMARY KEY, ts TEXT NOT NULL, fail INTEGER NOT NULL CHECK (fail IN (0, 1)));
CREATE TABLE measurements (run_id INTEGER NOT NULL REFERENCES runs(run_id), sensor INTEGER NOT NULL, value REAL,
                           PRIMARY KEY (run_id, sensor));
"""


def parse_label(line):
    """'-1 "19/07/2008 11:55:00"' -> (0, '2008-07-19 11:55:00'); the source uses -1 = pass, 1 = fail."""
    code, stamp = line.split(" ", 1)
    if code not in ("-1", "1"):
        raise ValueError(f"unexpected label {code!r}")
    ts = datetime.strptime(stamp.strip().strip('"'), "%d/%m/%Y %H:%M:%S")
    return int(code == "1"), ts.strftime("%Y-%m-%d %H:%M:%S")


def parse_row(line):
    vals = [float(x) for x in line.split()]
    return [None if math.isnan(v) else v for v in vals]


def load(raw=RAW, db=DB):
    data = (raw / "secom.data").read_text().splitlines()
    labels = (raw / "secom_labels.data").read_text().splitlines()
    if len(data) != len(labels):
        raise ValueError(f"{len(data)} data rows but {len(labels)} label rows")
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    width = None
    for run_id, (row, lab) in enumerate(zip(data, labels)):
        values = parse_row(row)
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError(f"row {run_id} has {len(values)} values, expected {width}")
        fail, ts = parse_label(lab)
        con.execute("INSERT INTO runs VALUES (?, ?, ?)", (run_id, ts, fail))
        con.executemany("INSERT INTO measurements VALUES (?, ?, ?)",
                        ((run_id, s, v) for s, v in enumerate(values)))
    con.commit()
    con.close()
    return len(data), width


if __name__ == "__main__":
    n, w = load()
    print(f"loaded {n} runs x {w} sensors into {DB}", file=sys.stderr)
