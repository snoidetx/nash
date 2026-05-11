import sys, pathlib


def setup_roots():
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    VINFO = ROOT / "freeshap" / "vinfo"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(VINFO))
