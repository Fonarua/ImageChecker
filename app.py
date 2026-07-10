"""Local web UI for the AI-watermark detector.

Run:  .venv/bin/python app.py   then open http://127.0.0.1:8000
"""

import uuid
from pathlib import Path

from flask import Flask, request, render_template_string, send_from_directory
from PIL import Image, ImageDraw

from detector import check_metadata, scan, to_gray8, HIGH_DEPTH_MODES

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
RESULTS = BASE / "results"
UPLOADS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# star_score tiers from the geometry verification stage
CONFIRMED = 2.0
LIKELY = 1.0
SHEET_COLS = 10
SHEET_CELL = 72

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Image Checker</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #14161a;
         color: #e6e6e6; max-width: 980px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
  .drop { border: 2px dashed #4a5160; border-radius: 12px; padding: 3rem 1rem;
          text-align: center; cursor: pointer; transition: background .15s; }
  .drop.hover { background: #1e2330; border-color: #7aa2ff; }
  .drop input { display: none; }
  .muted { color: #9aa3b2; font-size: .9rem; }
  .verdict { padding: 1rem 1.2rem; border-radius: 10px; margin: 1.2rem 0; font-size: 1.05rem; }
  .v-red { background: #3a1f24; border: 1px solid #a94452; }
  .v-yellow { background: #3a331f; border: 1px solid #a98f44; }
  .v-green { background: #1f3a26; border: 1px solid #44a95e; }
  .cards { display: flex; flex-wrap: wrap; gap: 1rem; }
  .card { background: #1c2028; border: 1px solid #2c323e; border-radius: 10px;
          padding: .8rem; width: 180px; }
  .card img { width: 100%; image-rendering: pixelated; border-radius: 6px; }
  .card .s { font-size: .85rem; color: #9aa3b2; margin-top: .4rem; }
  .main-img { max-width: 100%; border-radius: 10px; border: 1px solid #2c323e; }
  ul { line-height: 1.6; }
  a { color: #7aa2ff; }
  .spin { display: none; margin: 1rem 0; color: #9aa3b2; }
  .spin .bar { height: 8px; background: #232838; border-radius: 4px;
               overflow: hidden; margin-top: .5rem; }
  .spin .fill { height: 100%; width: 35%; background: #7aa2ff; border-radius: 4px;
                animation: slide 1.2s ease-in-out infinite; }
  @keyframes slide { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }
</style>
</head>
<body>
<h1>🔎 Image Checker <span class="muted">— AI watermark &amp; provenance scan</span></h1>

<form id="f" method="post" action="/check" enctype="multipart/form-data">
  <label class="drop" id="drop">
    <input type="file" name="image" accept="image/*,.tif,.tiff" id="file">
    <div><strong>Drop an image here</strong> or click to choose a file</div>
    <div class="muted">JPEG / PNG / WebP / TIFF, up to 50 MB. Everything stays on your machine.</div>
  </label>
</form>
<div class="spin" id="spin">Analyzing… (10–30 seconds depending on image size)
  <div class="bar"><div class="fill"></div></div>
</div>

{% if result %}
  <div class="verdict {{ result.verdict_class }}">{{ result.verdict }}</div>

  <h2>Provenance metadata</h2>
  {% if result.meta %}
    <ul>{% for m in result.meta %}<li>{{ m }}</li>{% endfor %}</ul>
  {% else %}
    <p class="muted">None — EXIF/XMP/C2PA fully stripped. This proves nothing by itself,
    but means the file's history can't be traced.</p>
  {% endif %}

  <h2>Scanned image ({{ result.w }}×{{ result.h }})</h2>
  <p class="muted">Red boxes = strong sparkle candidates, yellow = weak. Verify with the crops below.</p>
  <img class="main-img" src="/results/{{ result.annotated }}">

  {% if result.confirmed %}
    <h2>Confirmed star marks ({{ result.confirmed|length }})</h2>
    <p class="muted">Clean four-pointed star geometry. If it sits on a flat area or asset corner
    (not inside artwork like a night sky), it is almost certainly an AI watermark.</p>
    <div class="cards">
    {% for c in result.confirmed %}
      <div class="card"><img src="/results/{{ c.crop }}">
        <div class="s">({{ c.x }}, {{ c.y }}) · star {{ "%.1f"|format(c.star) }} · ncc {{ "%.2f"|format(c.ncc) }}</div></div>
    {% endfor %}
    </div>
  {% endif %}

  {% if result.likely %}
    <h2>Likely star shapes ({{ result.likely|length }})</h2>
    <p class="muted">Star-like geometry — may be watermarks, decorative sparkles, or star glints
    inside artwork. Judge by what surrounds them.</p>
    <div class="cards">
    {% for c in result.likely %}
      <div class="card"><img src="/results/{{ c.crop }}">
        <div class="s">({{ c.x }}, {{ c.y }}) · star {{ "%.1f"|format(c.star) }} · ncc {{ "%.2f"|format(c.ncc) }}</div></div>
    {% endfor %}
    </div>
  {% endif %}

  {% if result.sheet %}
    <h2>All remaining candidates ({{ result.rest_count }})</h2>
    <details>
      <summary class="muted" style="cursor:pointer">Show contact sheet — ranked left-to-right, top-to-bottom.
      Mostly noise, but very faint watermarks can end up here; scan it if the image is suspicious.</summary>
      <img class="main-img" style="margin-top:1rem" src="/results/{{ result.sheet }}">
    </details>
  {% endif %}
{% endif %}

<script>
const drop = document.getElementById('drop'), file = document.getElementById('file'),
      form = document.getElementById('f'), spin = document.getElementById('spin');
const submit = () => { spin.style.display = 'block'; form.submit(); };
file.addEventListener('change', () => file.files.length && submit());
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('hover'); }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('hover'); }));
drop.addEventListener('drop', ev => { file.files = ev.dataTransfer.files; submit(); });
</script>
</body>
</html>"""


def analyze(path: Path):
    img = Image.open(path)
    img = (to_gray8(img) if img.mode in HIGH_DEPTH_MODES else img).convert("RGB")
    w, h = img.size
    meta = check_metadata(str(path))

    hits = scan(str(path))  # (star_score, ncc, cx, cy, size), best first
    confirmed = [c for c in hits if c[0] >= CONFIRMED]
    likely = [c for c in hits if LIKELY <= c[0] < CONFIRMED]
    rest = [c for c in hits if c[0] < LIKELY]

    rid = uuid.uuid4().hex[:12]
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    lw = max(2, w // 500)

    def crop_box(cx, cy, size, pad=1.6):
        half = max(int(size * pad), 40)
        return (max(0, cx - half), max(0, cy - half),
                min(w, cx + half), min(h, cy + half))

    def save_crop(c, i, tag):
        _, _, cx, cy, size = c
        name = f"{rid}_{tag}{i}.png"
        img.crop(crop_box(cx, cy, size)).resize((160, 160), Image.NEAREST).save(RESULTS / name)
        return name

    def to_dict(c, i, tag, color):
        star, ncc, cx, cy, size = c
        draw.rectangle((cx - size, cy - size, cx + size, cy + size),
                       outline=color, width=lw)
        return {"star": star, "ncc": ncc, "x": cx, "y": cy, "size": size,
                "crop": save_crop(c, i, tag)}

    confirmed_d = [to_dict(c, i, "c", "#ff4455") for i, c in enumerate(confirmed)]
    likely_d = [to_dict(c, i, "l", "#e6c34a") for i, c in enumerate(likely)]

    ann_name = f"{rid}_annotated.jpg"
    annotated.save(RESULTS / ann_name, quality=88)

    # contact sheet with every remaining candidate, ranked
    sheet_name = None
    if rest:
        cols = SHEET_COLS
        rows = (len(rest) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * SHEET_CELL, rows * SHEET_CELL), "#14161a")
        for i, c in enumerate(rest):
            _, _, cx, cy, size = c
            cell = img.crop(crop_box(cx, cy, size)).resize(
                (SHEET_CELL - 4, SHEET_CELL - 4), Image.NEAREST)
            sheet.paste(cell, ((i % cols) * SHEET_CELL + 2, (i // cols) * SHEET_CELL + 2))
        sheet_name = f"{rid}_sheet.jpg"
        sheet.save(RESULTS / sheet_name, quality=85)

    declared_ai = any("AI" in m or "C2PA" in m or "Gemini" in m for m in meta)
    if declared_ai:
        verdict, cls = "🚨 File metadata declares AI generation or carries content credentials.", "v-red"
    elif confirmed_d:
        verdict, cls = (f"🚨 {len(confirmed_d)} four-pointed star mark(s) confirmed — "
                        "inspect the red crops. An isolated star on a flat area is an AI watermark.", "v-red")
    elif likely_d:
        verdict, cls = (f"⚠️ {len(likely_d)} star-like shape(s) found — check the crops to judge "
                        "whether they are watermarks or just artwork.", "v-yellow")
    else:
        verdict, cls = ("✅ No star watermark found. Note: absence of evidence is not proof — "
                        "watermarks can be cropped, very faint marks can hide in the contact sheet, "
                        "and this tool has no statistical detector yet.", "v-green")

    return {"verdict": verdict, "verdict_class": cls, "meta": meta,
            "w": w, "h": h, "annotated": ann_name,
            "confirmed": confirmed_d, "likely": likely_d,
            "sheet": sheet_name, "rest_count": len(rest)}


@app.route("/")
def index():
    return render_template_string(PAGE, result=None)


@app.route("/check", methods=["POST"])
def check():
    f = request.files.get("image")
    if not f or not f.filename:
        return render_template_string(PAGE, result=None)
    for d in (UPLOADS, RESULTS):
        for old in d.iterdir():
            if old.is_file():
                old.unlink()
    ext = Path(f.filename).suffix.lower() or ".jpg"
    path = UPLOADS / f"{uuid.uuid4().hex[:12]}{ext}"
    f.save(path)
    try:
        result = analyze(path)
    except Exception as e:
        return f"<p>Could not analyze this file: {e}</p><p><a href='/'>back</a></p>", 400
    return render_template_string(PAGE, result=result)


@app.route("/results/<name>")
def results(name):
    return send_from_directory(RESULTS, name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
