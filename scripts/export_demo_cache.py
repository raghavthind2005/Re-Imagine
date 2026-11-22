"""Write the demo's examples: dev ARKitScenes views that carry a human-labelled claim.

    python scripts/export_demo_cache.py

Only ARKitScenes imagery is exported (non-commercial licence that allows redistribution
with attribution; see docs/LICENSES.md). ScanNet and ScanNet++ frames are never exported.
Output: `demo/examples/` (committed): three JPEGs per view and `index.json`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from human_label_extra import labelled

ROOT = Path(__file__).resolve().parents[1]
CATEGORY = {
    "hallucination": "hallucination",
    "real_object": "real object",
    "detector_error": "detector error",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--views", default=str(ROOT / "runs/_kaggle/hallucination_v2/runs/heldout_imagine")
    )
    ap.add_argument("--out", default=str(ROOT / "demo/examples"))
    args = ap.parse_args()
    from PIL import Image

    rows = labelled(ROOT / "results/hallucination_v2", ROOT / "data/labels/claim_sample.json",
                    ROOT / "data/labels/human_labels.csv", "dev")  # fmt: skip
    human = {(r["scene"], r["target"], r["label"]): r["category"] for r in rows}
    claims = {}
    for f in sorted((ROOT / "results/hallucination_v2").glob("views*.jsonl")):
        for line in f.read_text().splitlines():
            v = json.loads(line)
            if not v["is_control"] and v["scene"].startswith("arkitscenes"):
                claims[(v["scene"], v["target"])] = v
    keep = sorted({(s, t) for (s, t, _l) in human if s.startswith("arkitscenes")})
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    index = []
    for scene, t in keep:
        v = claims[(scene, t)]
        arr = np.load(Path(args.views) / scene / "views.npz")
        d = out / f"{scene}_{t}"
        d.mkdir(parents=True)
        for name in ("imagined", "real", "geo"):
            Image.fromarray(arr[name][t]).save(d / f"{name}.jpg", quality=80)
        index.append(
            {
                "id": f"{scene.replace('__', '/')} · camera {t}",
                "dir": d.name,
                "map_coverage": v["geo_coverage"],
                "claims": [
                    {
                        "label": c["label"],
                        "map_check": "kept" if c["verdict"] == "supported" else "flagged",
                        "map_verdict": c["verdict"].replace("_", " "),
                        "region_covered": round(c["region_covered"], 2),
                        "human": CATEGORY.get(human.get((scene, t, c["label"])), "not labelled"),
                    }
                    for c in sorted(v["claims"], key=lambda c: c["label"])
                ],
            }
        )
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} views -> {out}")


if __name__ == "__main__":
    main()
