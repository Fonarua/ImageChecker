"""Prototype AI-watermark detector.

Layer 1: metadata / provenance check (EXIF, XMP, C2PA markers).
Layer 2: visible-watermark template matching (Gemini sparkle glyph)
         via FFT-based normalized cross-correlation at multiple scales.
"""

import sys
import numpy as np
from PIL import Image, ImageFilter


# ---------- Layer 1: metadata ----------

def check_metadata(path):
    findings = []
    data = open(path, "rb").read()
    if data.find(b"http://ns.adobe.com/xap/1.0/") != -1:
        findings.append("XMP block present")
    if data.find(b"Exif\x00\x00") != -1:
        findings.append("EXIF block present")
    # EXIF tags via Pillow — works for JPEG and TIFF alike (TIFF carries
    # its tags natively, without the Exif marker searched for above)
    try:
        exif = Image.open(path).getexif()
        for tag, label in ((305, "software"), (271, "camera make"),
                           (272, "camera model"), (315, "artist"),
                           (306, "modified")):
            if tag in exif:
                findings.append(f"{label}: {str(exif[tag]).strip()}")
    except Exception:
        pass
    for marker, label in [
        (b"c2pa", "C2PA content credentials"),
        (b"GContainer", "Google GContainer (Gemini/Pixel)"),
        (b"trainedAlgorithmicMedia", "IPTC trainedAlgorithmicMedia (declared AI)"),
        (b"Midjourney", "Midjourney"),
        (b"DALL-E", "DALL-E"),
    ]:
        if data.find(marker) != -1:
            findings.append(label)
    return findings


# ---------- Layer 2: visible watermark ----------

def make_star_template(size=32, p=0.5):
    """Synthetic 4-point sparkle. p=0.5: thin pinched arms (Gemini star);
    p=1.0: straight-edged fat diamond (how the mark looks enlarged/blurred)."""
    n = size
    y, x = np.mgrid[0:n, 0:n]
    cx = cy = (n - 1) / 2
    dx, dy = np.abs(x - cx) / cx, np.abs(y - cy) / cy
    inside = (dx ** p + dy ** p) <= 1.0
    return inside.astype(np.float64)


def ncc_match(img, tpl):
    """Normalized cross-correlation of img with zero-mean template, via FFT."""
    ih, iw = img.shape
    th, tw = tpl.shape
    tplz = tpl - tpl.mean()
    tnorm = np.sqrt((tplz ** 2).sum())

    fshape = (ih + th - 1, iw + tw - 1)
    F = np.fft.rfft2(img, fshape)
    T = np.fft.rfft2(tplz[::-1, ::-1], fshape)
    corr = np.fft.irfft2(F * T, fshape)[th - 1:ih, tw - 1:iw]

    # local mean/std of image under the template window (integral images)
    ii = np.cumsum(np.cumsum(np.pad(img, ((1, 0), (1, 0))), 0), 1)
    ii2 = np.cumsum(np.cumsum(np.pad(img ** 2, ((1, 0), (1, 0))), 0), 1)
    s = ii[th:, tw:] - ii[:-th, tw:] - ii[th:, :-tw] + ii[:-th, :-tw]
    s2 = ii2[th:, tw:] - ii2[:-th, tw:] - ii2[th:, :-tw] + ii2[:-th, :-tw]
    npix = th * tw
    var = np.maximum(s2 - s ** 2 / npix, 1e-9)
    return corr / (np.sqrt(var) * tnorm)


HIGH_DEPTH_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")


def to_gray8(im):
    """8-bit grayscale; 16-bit/float images (e.g. TIFF) are normalized
    instead of clipped, which Pillow's plain convert("L") would do."""
    if im.mode in HIGH_DEPTH_MODES:
        a = np.asarray(im, dtype=np.float64)
        lo, hi = a.min(), a.max()
        a = (a - lo) / (hi - lo + 1e-9)
        return Image.fromarray((a * 255).astype(np.uint8), "L")
    return im.convert("L")


def load_channels(path):
    """Grayscale image plus a lightly blurred copy (kills dither/halftone
    noise so large faint marks emerge)."""
    im = to_gray8(Image.open(path))
    raw = np.asarray(im, dtype=np.float64) / 255.0
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(2)),
                      dtype=np.float64) / 255.0
    return raw, blur


def find_sparkles(path, threshold=0.55, channels=None):
    raw, blur = channels if channels is not None else load_channels(path)
    # (channel, template exponent, sizes): thin star and fat diamond on the
    # raw image; on the blurred channel only large sizes make sense
    configs = [(raw, 0.5, (16, 24, 32, 48, 64)),
               (raw, 1.0, (16, 24, 32, 48, 64)),
               (blur, 0.5, (16, 24, 32, 48, 64)),
               (blur, 1.0, (12, 16, 24, 32, 48, 64))]
    hits = []
    for img, p, sizes in configs:
        for size in sizes:
            if size >= min(img.shape):
                continue
            score = ncc_match(img, make_star_template(size, p))
            while True:
                ij = np.unravel_index(np.argmax(score), score.shape)
                v = score[ij]
                if v < threshold:
                    break
                cy, cx = ij[0] + size // 2, ij[1] + size // 2
                hits.append((v, cx, cy, size))
                y0, y1 = max(0, ij[0] - size), ij[0] + size
                x0, x1 = max(0, ij[1] - size), ij[1] + size
                score[y0:y1, x0:x1] = 0  # suppress neighborhood
    # merge duplicate detections across scales/shapes/channels
    merged = []
    for v, cx, cy, size in sorted(hits, reverse=True):
        if all((cx - m[1]) ** 2 + (cy - m[2]) ** 2 > (max(size, m[3])) ** 2
               for m in merged):
            merged.append((v, cx, cy, size))
    return merged


# ---------- verification stage: star geometry check ----------

def arm_score(img, cx, cy, R, fat=False):
    """Weakest-arm local contrast: each of the 4 star arms vs pixels beside it.

    Robust to mixed backgrounds (each arm is compared only to its own
    neighborhood) and to polarity (light star on dark or dark on light).
    fat=True samples further out with a wider side offset, matching a
    straight-edged diamond instead of thin pinched arms.
    """
    H, W = img.shape
    M = int(1.55 * R) + 2
    if not (M <= cy < H - M and M <= cx < W - M):
        return None
    p = img[cy - M:cy + M + 1, cx - M:cx + M + 1]
    c = M
    if fat:
        rs = np.arange(max(2, int(R * 0.50)), max(int(R * 0.90), 4))
        o = max(3, int(R * 0.60))
    else:
        rs = np.arange(max(2, int(R * 0.30)), max(int(R * 0.85), 4))
        o = max(3, int(R * 0.45))
    rt = np.arange(int(R * 1.10), max(int(R * 1.45), int(R * 1.10) + 2))
    d = (rs / np.sqrt(2)).astype(int)
    diag = np.concatenate([p[c + d, c + d], p[c + d, c - d],
                           p[c - d, c + d], p[c - d, c - d]])
    # far-diagonal samples: outside both a thin star and a fat diamond,
    # but still inside a round dot of similar radius (dot rejection)
    rd = np.arange(max(2, int(R * 0.80)), max(int(R * 1.05), int(R * 0.80) + 2))
    dd = (rd / np.sqrt(2)).astype(int)
    diag_far = np.concatenate([p[c + dd, c + dd], p[c + dd, c - dd],
                               p[c - dd, c + dd], p[c - dd, c - dd]])
    center = p[c - 1:c + 2, c - 1:c + 2].mean()
    pol = 1.0 if center >= diag.mean() else -1.0
    arms = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        px, py = -dy, dx
        on = p[c + dy * rs, c + dx * rs]
        side = np.concatenate([p[c + dy * rs + py * o, c + dx * rs + px * o],
                               p[c + dy * rs - py * o, c + dx * rs - px * o]])
        norm = side.std() + 0.03
        contrast = pol * (on.mean() - side.mean()) / norm
        # arms must END: past the tip the image returns to background,
        # otherwise this is a cross of long lines (e.g. texture seams).
        # Weighted 1.5x so soft translucent tips aren't over-penalized.
        tip = p[c + dy * rt, c + dx * rt]
        ends = pol * (on.mean() - tip.mean()) / norm
        # arms must beat the far diagonals, otherwise this is a round
        # blob (e.g. halftone dots), not a four-pointed star
        dterm = pol * (on.mean() - diag_far.mean()) / norm
        # arms must be CONTINUOUS: a chain of separate dots (halftone
        # grids form star-like quincunx layouts) has dark gaps along the
        # arm, a real star does not — check the smoothed profile floor
        prof = np.convolve(on, np.ones(3) / 3, mode="valid") if len(on) >= 3 else on
        gap = pol * (prof.min() if pol > 0 else prof.max()) - pol * side.mean()
        gap /= norm
        arms.append(min(contrast, 1.5 * ends, 1.5 * dterm, 2.0 * gap))
    return min(arms)


def upscaled_arm(img, cx, cy, R):
    """arm_score on a 2x-upscaled local patch: small stars (~10-20px) leave
    too few pixels per arm at native resolution for reliable sampling."""
    m = 4 * R + 8
    if not (m <= cy < img.shape[0] - m and m <= cx < img.shape[1] - m):
        return None
    patch = img[cy - m:cy + m, cx - m:cx + m]
    im = Image.fromarray((patch * 255).astype(np.uint8)).resize(
        (4 * m, 4 * m), Image.BICUBIC)
    up = np.asarray(im, dtype=np.float64) / 255.0
    best = None
    for fat in (False, True):
        for dy in (-4, -2, 0, 2, 4):
            for dx in (-4, -2, 0, 2, 4):
                s = arm_score(up, 2 * m + dx, 2 * m + dy, 2 * R, fat)
                if s is not None and (best is None or s > best):
                    best = s
    return best


def periodicity(raw, blur, cx, cy, size):
    """Strongest short-lag autocorrelation peak of the fine-detail band
    around the candidate. High for halftone/dot lattices (4-24px spacing),
    low around isolated star marks. Sub-3px dither is excluded by the lag
    floor (it is harmless: the blur channel removes it)."""
    w = max(24, 3 * size)
    H, W = raw.shape
    x0 = min(max(cx - w, 0), max(W - 2 * w, 0))
    y0 = min(max(cy - w, 0), max(H - 2 * w, 0))
    a = (raw - blur)[y0:y0 + 2 * w, x0:x0 + 2 * w]
    if min(a.shape) < 16:
        return 0.0
    a = a - a.mean()
    P = np.abs(np.fft.rfft2(a)) ** 2
    ac = np.fft.irfft2(P, a.shape)
    ac = ac / (ac[0, 0] + 1e-12)
    ny, nx = a.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    lag2 = np.minimum(yy, ny - yy) ** 2 + np.minimum(xx, nx - xx) ** 2
    mask = (lag2 >= 9) & (lag2 <= 24 ** 2)
    return float(ac[mask].max()) if mask.any() else 0.0


def star_verify(channels, cx0, cy0, size):
    """Best arm_score over channels, arm geometries, nearby centers and
    scales (NCC peaks can be a few pixels off)."""
    best = -9.0
    for ci, img in enumerate(channels):
        for fat in (False, True):
            for f in (0.5, 0.75, 1.0, 1.5, 2.0):
                R = max(8, int(size * f))
                if ci > 0 and R < 24:
                    continue  # blurred channel only for large marks
                step = max(2, R // 6)
                for dy in (-2 * step, -step, 0, step, 2 * step):
                    for dx in (-2 * step, -step, 0, step, 2 * step):
                        s = arm_score(img, cx0 + dx, cy0 + dy, R, fat)
                        if s is not None and s > best:
                            best = s
    if size <= 20 and best < 2.0:  # fine-grained pass for small marks
        up = -9.0
        for img in channels:
            for R in range(max(5, size // 2 - 2), size // 2 + 3):
                s = upscaled_arm(img, cx0, cy0, R)
                if s is not None and s > up:
                    up = s
        if up > best:
            # dot lattices form star-like layouts that fool the upscaled
            # sampling — accept its verdict only on non-periodic ground
            if periodicity(channels[0], channels[1], cx0, cy0, size) >= 0.33:
                up = min(up, 0.9)
            best = max(best, up)
    return best


def scan(path, threshold=0.55, max_candidates=200):
    """Full pipeline: NCC candidates, then geometry verification.

    Returns candidates sorted by verified star score, as
    (star_score, ncc_score, cx, cy, size) tuples.
    """
    channels = load_channels(path)
    hits = sorted(find_sparkles(path, threshold, channels),
                  reverse=True)[:max_candidates]
    scored = [(star_verify(channels, cx, cy, size), ncc, cx, cy, size)
              for ncc, cx, cy, size in hits]
    return sorted(scored, reverse=True)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"=== {path} ===")
        meta = check_metadata(path)
        print("metadata:", ", ".join(meta) if meta else "none (stripped)")
        for star, ncc, cx, cy, size in scan(path):
            if star < 0.5:
                continue
            tier = "CONFIRMED" if star >= 2.0 else ("likely" if star >= 1.0 else "weak")
            print(f"{tier:9s} star at ({cx}, {cy})  scale={size}px  "
                  f"star_score={star:.2f}  ncc={ncc:.2f}")
