"""Export a stratified, blind sample of object claims as an HTML labelling sheet.

    python scripts/make_label_sheet.py --records results/hallucination_v2 \\
        --views runs/_kaggle/hallucination_v2/runs/heldout_imagine \\
        --sample data/labels/claim_sample.json --out labelling/claims_sheet.html

The sample (claim keys and strata) is saved once and never regenerated. The sheet shows
the imagined view and the real held-out frame of the same camera, and asks whether the
claimed object is visible in each. No model outputs are shown. The sheet is
self-contained; answers are kept in the browser and exported as CSV with a button.
The sheet holds benchmark frames, so it stays local (gitignored).
"""

# ruff: noqa: E501  (embedded HTML template)
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import numpy as np

from mindseye.data.splits import save_split

BINS = ((0.0, 0.2, "low"), (0.2, 0.6, "mid"), (0.6, 1.01, "high"))


def coverage_bin(x: float) -> str:
    return next(name for lo, hi, name in BINS if lo <= x < hi)


def jpeg_b64(image: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def draw_sample(records: Path, per_cell: int, seed: int) -> list[dict]:
    claims = []
    for f in sorted(records.glob("views*.jsonl")):
        for line in f.read_text().splitlines():
            v = json.loads(line)
            if v["is_control"]:
                continue
            for c in v["claims"]:
                claims.append(
                    {
                        "key": f"{v['scene']}|{v['target']}|{c['label']}",
                        "stratum": f"{'real' if c['true_in_real_view'] else 'invented'}"
                        f"/{coverage_bin(v['geo_coverage'])}",
                    }
                )
    claims.sort(key=lambda c: c["key"])
    rng = np.random.default_rng(seed)
    strata = sorted({c["stratum"] for c in claims})
    chosen, spare = [], []
    for s in strata:
        members = [c for c in claims if c["stratum"] == s]
        order = rng.permutation(len(members))
        chosen += [members[i] for i in order[:per_cell]]
        spare += [members[i] for i in order[per_cell:]]
    # Cells smaller than per_cell are topped up from the rest, to keep the total.
    short = per_cell * len(strata) - len(chosen)
    if short > 0:
        chosen += [spare[i] for i in rng.permutation(len(spare))[:short]]
    return [chosen[i] for i in rng.permutation(len(chosen))]  # blind order


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claim labelling</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --muted:#57534e; --line:#d6d3d1; --card:#fff; --accent:#1d4ed8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --line:#44403c; --card:#292524; --accent:#93c5fd; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:16px; }}
.item {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin:16px 0; }}
.imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.imgs figure {{ margin:0; }} .imgs img {{ width:100%; height:auto; border-radius:4px; }}
figcaption {{ color:var(--muted); font-size:13px; }}
.q {{ margin-top:8px; }} label {{ margin-right:14px; white-space:nowrap; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1; }}
button {{ font:inherit; padding:6px 12px; border-radius:6px; border:1px solid var(--accent); background:transparent; color:var(--accent); cursor:pointer; }}
@media (max-width:640px) {{ .imgs {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<div class="bar"><strong>Claim labelling</strong> — <span id="count"></span>
<button onclick="exportCsv()">Download CSV</button></div>
<p>Each item shows a computer-generated (imagined) view and the real photo taken from the same
camera. For the named object type, answer whether at least one instance is visible in each
image. Count partly visible objects. Use "unsure" when you cannot tell.</p>
{items}
</main>
<script>
const KEYS = {keys};
const store = {{
  get() {{ try {{ return JSON.parse(localStorage.getItem("{store}") || "{{}}"); }} catch (e) {{ return {{}}; }} }},
  set(v) {{ try {{ localStorage.setItem("{store}", JSON.stringify(v)); }} catch (e) {{}} }},
}};
let answers = store.get();
function update() {{
  const done = KEYS.filter(k => answers[k] && answers[k].real && answers[k].imagined).length;
  document.getElementById("count").textContent = done + " of " + KEYS.length + " done";
}}
document.querySelectorAll("input[type=radio]").forEach(el => {{
  const [k, q] = [el.dataset.key, el.dataset.q];
  if (answers[k] && answers[k][q] === el.value) el.checked = true;
  el.addEventListener("change", () => {{ answers[k] = Object.assign(answers[k] || {{}}, {{[q]: el.value}}); store.set(answers); update(); }});
}});
function exportCsv() {{
  const rows = [["key", "visible_in_real", "visible_in_imagined"]];
  KEYS.forEach(k => rows.push([k, (answers[k] || {{}}).real || "", (answers[k] || {{}}).imagined || ""]));
  const blob = new Blob([rows.map(r => r.map(x => '"' + x + '"').join(",")).join("\\n")], {{type: "text/csv"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "{filename}"; a.click();
}}
update();
</script></body></html>
"""


def radios(key: str, q: str) -> str:
    name = html.escape(f"{key}|{q}")
    return "".join(
        f'<label><input type="radio" name="{name}" value="{v}" data-key="{html.escape(key)}" '
        f'data-q="{q}"> {v}</label>'
        for v in ("yes", "no", "unsure")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()
    sample_path = Path(args.sample)
    if not sample_path.exists():
        sample = draw_sample(Path(args.records), args.per_cell, args.seed)
        save_split(
            sample_path,
            {"claims": sample},
            {"seed": args.seed, "per_cell": args.per_cell, "source": args.records,
             "strata": "SAM 3 truth (invented/real) x view coverage (<0.2, 0.2-0.6, >=0.6)"},
        )  # fmt: skip
    sample = json.loads(sample_path.read_text())["split"]["claims"]
    cache: dict[str, dict] = {}
    items = []
    for i, c in enumerate(sample, 1):
        scene, target, label = c["key"].split("|")
        if scene not in cache:
            cache.clear()
            cache[scene] = dict(np.load(Path(args.views) / scene / "views.npz"))
        v, t = cache[scene], int(target)
        items.append(
            f'<div class="item"><div><strong>{i}.</strong> Object: '
            f"<strong>{html.escape(label)}</strong></div>"
            f'<div class="imgs"><figure><img alt="imagined view" '
            f'src="data:image/jpeg;base64,{jpeg_b64(v["imagined"][t])}"><figcaption>Imagined view'
            f'</figcaption></figure><figure><img alt="real photo" '
            f'src="data:image/jpeg;base64,{jpeg_b64(v["real"][t])}"><figcaption>Real photo, same '
            f"camera</figcaption></figure></div>"
            f'<div class="q">Visible in the <strong>real</strong> photo? {radios(c["key"], "real")}</div>'
            f'<div class="q">Visible in the <strong>imagined</strong> view? '
            f'{radios(c["key"], "imagined")}</div></div>'
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        PAGE.format(
            items="\n".join(items),
            keys=json.dumps([c["key"] for c in sample]),
            store="claim-labels",
            filename="human_labels.csv",
        )
    )
    print(f"{len(sample)} claims -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
