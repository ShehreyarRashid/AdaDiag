"""v1.22 README section 6.22: reading TVDiag's own logs (no data or TVDiag checkout needed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_tvdiag_official as r                                    # noqa: E402


def test_parse_log_reads_the_released_metrics(tmp_path):
    log = tmp_path / "gaia.log"
    log.write_text(
        "2026-11-04 12:30:01 [INFO]: [Root localization] HR@1: 75.932%, HR@2: 85.623%, "
        "HR@3: 90.948%, HR@4: 93.291%, HR@5: 95.208%, avg@3: 0.845, MRR@3: 0.827\n"
        "2026-11-04 12:30:01 [INFO]: [Failure type classification] precision: 92.512%, "
        "recall: 94.782%, f1-score: 93.633%\n")
    m = r.parse_log(log)
    assert abs(m["HR@1"] - 0.75932) < 1e-9 and abs(m["f1"] - 0.93633) < 1e-9
    assert m["MRR@3"] == 0.827 and m["avg@3"] == 0.845


def test_committed_main_run_matches_its_log():
    import json
    recs = json.load(open(ROOT / "results" / "tvdiag_official" / "runs_main.json"))
    for rec in recs:
        m = r.parse_log(ROOT / "results" / "tvdiag_official" / "logs" / f"{rec['dataset']}_seed2_main.log")
        assert abs(m["HR@1"] - rec["HR@1"]) < 1e-9 and abs(m["f1"] - rec["f1"]) < 1e-9
