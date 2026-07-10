# ImageChecker

Detects visible AI watermarks (the Gemini four-pointed "sparkle" and similar
marks) in images, plus a provenance metadata scan (EXIF / XMP / C2PA).

## How it works

1. **Provenance metadata** — reports EXIF/XMP/C2PA blocks and AI-declaration
   tags. Stripped metadata proves nothing, but declared AI is a sure hit.
2. **Candidate search** — multi-scale normalized cross-correlation with two
   template shapes (thin four-pointed star and fat diamond) on both the raw
   grayscale image and a blurred copy (reveals faint marks under noisy
   textures such as camo patterns).
3. **Geometry verification** — every candidate is checked for real star
   structure: all four arms must contrast with pixels beside them, and the
   arms must *end* (rejects cross-shaped texture seams). Candidates are
   tiered: confirmed / likely / weak.

Results are evidence, not verdicts: confirmed/likely tiers come with zoomed
crops so a human makes the final call. Star glints inside artwork look the
same as watermarks — position (asset corner, flat area) is what tells them
apart.

## Local web UI

```bash
python3 -m venv .venv
.venv/bin/pip install pillow numpy flask
.venv/bin/python app.py     # open http://127.0.0.1:8000
```

Supports JPEG / PNG / WebP / TIFF (incl. 16-bit). Each new upload clears the
previous run's files from `uploads/` and `results/`.

## CLI

```bash
.venv/bin/python detector.py image.jpg texture.tiff ...
```

## Browser version (GitHub Pages)

`docs/index.html` is a self-contained static page that runs the same detector
in the browser via [Pyodide](https://pyodide.org) — no server, images never
leave the machine. Serve the `docs/` folder from GitHub Pages or any static
host.

## Limitations

- Detects **visible** watermarks. A cropped-out watermark passes; catching
  those needs a statistical detector (e.g. a trained classifier or a
  commercial API) — planned as a future layer.
- Invisible watermarks (Google SynthID) require Google's own detector.
- Asking a chatbot "did you generate this image?" is meaningless — models
  have no registry of their outputs. Check SynthID or the marks instead.
