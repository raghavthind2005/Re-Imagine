"""Gradio demo on cached examples: which imagined objects does the map of real frames confirm?

    pip install -r demo/requirements.txt && python demo/app.py

Runs from a clean clone, without a GPU: the examples in `demo/examples/` are cached dev views
from ARKitScenes (see docs/LICENSES.md). Each example is a held-out camera: the view Stable
Virtual Camera imagined from 8 observed frames, the real frame from that camera (used only
as ground truth), and the map of the observed frames rendered from the same camera.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

EXAMPLES = Path(__file__).parent / "examples"
INDEX = json.loads((EXAMPLES / "index.json").read_text())
CHOICES = [f"{v['id']} · map covers {100 * v['map_coverage']:.0f}% of the view" for v in INDEX]
HEADERS = ["Object", "Map check (ours)", "Map evidence", "Region covered", "Human label"]


def show(choice: str):
    v = INDEX[CHOICES.index(choice)]
    d = EXAMPLES / v["dir"]
    rows = [
        [
            c["label"],
            c["map_check"],
            c["map_verdict"],
            f"{100 * c['region_covered']:.0f}%",
            c["human"],
        ]
        for c in v["claims"]
    ]
    return str(d / "imagined.jpg"), str(d / "real.jpg"), str(d / "geo.jpg"), rows


with gr.Blocks(title="Grounded hallucination check") as app:
    gr.Markdown(
        "## Which imagined objects are really there?\n"
        "A world model imagined each view from 8 real frames of the room. The map check keeps "
        "an imagined object only if a labelled 3D map built from those real frames shows it "
        "from this camera. Objects in regions the real frames never saw are flagged. Cached "
        "examples; nothing is rendered live. Human labels: *hallucination* = drawn in the "
        "imagined view but absent from the real frame; *detector error* = not actually drawn."
    )
    pick = gr.Dropdown(CHOICES, value=CHOICES[0], label="Example")
    with gr.Row():
        imagined = gr.Image(label="Imagined view (Stable Virtual Camera)")
        real = gr.Image(label="Real frame, same camera (ground truth only)")
        geo = gr.Image(label="Map of the observed frames, same camera")
    table = gr.Dataframe(headers=HEADERS, interactive=False)
    pick.change(show, pick, [imagined, real, geo, table])
    app.load(show, pick, [imagined, real, geo, table])

if __name__ == "__main__":
    app.launch()
