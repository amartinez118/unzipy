"""Tests of the name-based raw trace reader."""

import numpy as np
import pytest

from unzipy.io import read_trace

FULL = [
    "CycleCount/n",
    "A_PsdX",
    "X_force",
    "Y_force",
    "A_dist-Y",
    "B_dist-Y",
    "time(sec)",
    "Status",
]
COMPACT = [
    "CycleCount/n",
    "X_force",
    "Y_force",
    "Z_force",
    "A_dist-Y",
    "B_dist-Y",
    "time(sec)",
]


def write(path, header, rows, preamble=("#Begin Write:",), extra=()):
    lines = list(preamble) + ["#" + "\t".join(header)]
    lines += ["\t".join(f"{v:.4f}" for v in r) for r in rows] + list(extra)
    path.write_text("\n".join(lines) + "\n")


def table(n, header, seed=0, start=0):
    rng = np.random.default_rng(seed)
    d = {h: rng.normal(size=n) for h in header}
    d["CycleCount/n"] = np.arange(start, start + n) + 1.0
    d["Y_force"] = -10.0 - rng.random(n)  # the instrument reports negative Y
    d["A_dist-Y"] = np.linspace(-500, -400, n)
    d["B_dist-Y"] = np.linspace(-1000, -900, n)
    d["time(sec)"] = (np.arange(n) + start) * 0.001
    if "Status" in d:
        d["Status"] = np.full(n, 130.0)
    return d, np.column_stack([d[h] for h in header])


def test_reads_columns_by_name_in_two_layouts(tmp_path):
    for header, name in ((FULL, "full"), (COMPACT, "compact")):
        d, rows = table(50, header)
        p = tmp_path / f"{name}.txt"
        write(p, header, rows)
        tr = read_trace(p)
        np.testing.assert_allclose(tr.t, d["time(sec)"], atol=1e-4)
        np.testing.assert_allclose(tr.force, d["Y_force"], atol=1e-4)
        np.testing.assert_allclose(tr.f, -d["Y_force"], atol=1e-4)
        np.testing.assert_allclose(
            tr.x, 0.5 * (d["A_dist-Y"] + d["B_dist-Y"]), atol=1e-4
        )
        assert np.isfinite(tr.count).all()
        assert np.isfinite(tr.status).all() == (name == "full")


def test_file_without_a_counter_and_header_not_on_the_first_line(tmp_path):
    header = [h for h in FULL if h != "CycleCount/n"]
    d, rows = table(30, header)
    p = tmp_path / "nocount.txt"
    write(p, header, rows, preamble=("#Begin Write:", "#second comment"))
    tr = read_trace(p)
    assert np.isnan(tr.count).all()
    np.testing.assert_allclose(tr.f, -d["Y_force"], atol=1e-4)


def test_continuation_files_are_concatenated_in_order(tmp_path):
    d1, r1 = table(40, COMPACT, seed=1, start=0)
    d2, r2 = table(25, COMPACT, seed=2, start=40)
    write(tmp_path / "a.txt", COMPACT, r1)
    write(tmp_path / "b.txt", COMPACT, r2)
    tr = read_trace([tmp_path / "a.txt", tmp_path / "b.txt"])
    assert tr.t.size == 65 and np.all(np.diff(tr.t) > 0)
    np.testing.assert_allclose(tr.count, np.arange(1, 66))
    assert [p.endswith(("a.txt", "b.txt")) for p in tr.files] == [True, True]


def test_truncated_rows_and_comment_lines_are_dropped(tmp_path):
    d, rows = table(60, COMPACT)
    p = tmp_path / "ragged.txt"
    write(
        p,
        COMPACT,
        rows,
        extra=("# a stray comment", "61\t1.0\t2.0", "62\tnot\ta\trow"),
    )
    tr = read_trace(p)
    assert tr.t.size == 60


def test_custom_column_names(tmp_path):
    header = ["tt", "force_y", "p1", "p2"]
    rows = np.column_stack(
        [np.arange(10) * 0.1, -np.arange(10.0), np.zeros(10), np.full(10, 4.0)]
    )
    p = tmp_path / "custom.txt"
    write(p, header, rows)
    tr = read_trace(p, force="force_y", positions=("p1", "p2"), time="tt")
    np.testing.assert_allclose(tr.x, 2.0)
    np.testing.assert_allclose(tr.f, np.arange(10.0), atol=1e-4)


def test_missing_required_columns_raise(tmp_path):
    header = [h for h in COMPACT if h != "B_dist-Y"]
    d, rows = table(10, header)
    p = tmp_path / "nopos.txt"
    write(p, header, rows)
    with pytest.raises(ValueError, match="not found"):
        read_trace(p)
    q = tmp_path / "nohead.txt"
    q.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="header"):
        read_trace(q)
    r = tmp_path / "empty.txt"
    write(r, COMPACT, np.empty((0, len(COMPACT))))
    with pytest.raises(ValueError):
        read_trace(r)
