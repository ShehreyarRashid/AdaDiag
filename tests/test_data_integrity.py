"""The bundled data/ is exactly what TVDiag published, and loads with the expected shape.

Skipped (not failed) if data/ is absent, so the rest of the suite still runs on a partial
download.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(not (DATA / "SHA256SUMS").exists(), reason="data/ not present")


def _entries():
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        yield digest, rel.lstrip("./")


def test_every_file_matches_its_published_hash():
    bad = []
    for digest, rel in _entries():
        p = DATA / rel
        if not p.exists():
            bad.append(f"missing: {rel}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(f"changed: {rel}")
    assert not bad, "data/ differs from TVDiag's release:\n  " + "\n  ".join(bad)


def test_all_three_datasets_are_listed():
    listed = {rel.split("/")[0] for _, rel in _entries()}
    assert listed == {"gaia", "hotel", "sockshop"}
    assert len(list(_entries())) == 27            # 9 files per dataset


@pytest.mark.parametrize("name", ["gaia", "hotel", "sockshop"])
def test_loads_from_the_repository_by_default(name):
    """No --repo needed: data.load finds data/ in this repository and asserts EXPECTED."""
    from adadiag import config, data
    ds = data.load(name)                                  # strict=True asserts the counts
    config.check_against(config.get(name), ds)
