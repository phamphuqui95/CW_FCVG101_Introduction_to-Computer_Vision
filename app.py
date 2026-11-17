"""Vision Lab - Task 3 assistant (Streamlit).

Run from the FCVG101_Task1B_Project folder:
    streamlit run app.py

Modes: Calibrate, Filter, Edges and FFT, Features, Classify.
Classify uses the trained Task 2 model (FruitSort_Task2/artifacts/model.pth).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.classical_cv import (add_gaussian_noise, add_salt_pepper_noise, draw_harris, edges_outputs,
                              filtering_outputs, frequency_outputs, harris_points, homography_inliers,
                              mse, psnr, sift_match, to_gray)

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "data" / "demo_scene.png"
SAMPLE_SAME = ROOT / "data" / "same_scene_transformed.png"
SAMPLE_OTHER = ROOT / "data" / "unrelated_scene.png"
TASK2 = Path(os.environ.get("FRUITSORT_DIR", ROOT.parent / "FruitSort_Task2"))
MODEL = TASK2 / "artifacts" / "model.pth"
TYPES = ["jpg", "jpeg", "png"]

st.set_page_config(page_title="Vision Lab", page_icon="🍏", layout="wide", initial_sidebar_state="collapsed")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#0e1116; --mute:#5d6572; --line:#e3e6ec; --paper:#f6f7f9; --card:#fff; --accent:#4a2fd0; --tint:rgba(74,47,208,.08);
        --lift:0 1px 2px rgba(14,17,22,.04), 0 18px 40px -24px rgba(14,17,22,.25); }
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not(code) { font-family:'Inter','Helvetica Neue',Arial,sans-serif; }
.stApp { background:var(--paper); color:var(--ink); -webkit-font-smoothing:antialiased; }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display:none !important; }
.block-container { max-width:1260px; padding:1.4rem 2rem 5rem; }

.brand { display:flex; align-items:center; gap:10px; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.25rem; letter-spacing:-.03em; }
.dot { width:12px; height:12px; border-radius:50%; background:var(--accent); box-shadow:0 0 0 5px var(--tint); }
.hero { padding:2.2rem 0 1.4rem; animation:rise .7s cubic-bezier(.16,1,.3,1) both; }
.headline { font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:-.045em; line-height:1; font-size:clamp(2.4rem,5vw,4rem); }
.lead { margin-top:.9rem; color:var(--mute); font-size:1.08rem; line-height:1.55; max-width:38em; }
@keyframes rise { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:none; } }

.st-key-nav { margin:.4rem 0 1.6rem; }
.st-key-controls, .st-key-results { background:var(--card); border:1px solid var(--line); border-radius:22px; box-shadow:var(--lift); padding:1.7rem; }
.st-key-results { padding:2rem 2.2rem; }
.hint { color:var(--mute); font-size:.95rem; line-height:1.5; margin-bottom:1rem; }
.rtitle { font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:-.035em; font-size:1.9rem; line-height:1.05; }
.rdesc { margin-top:.5rem; color:var(--mute); line-height:1.55; max-width:40em; }
.rhead { padding-bottom:1.3rem; margin-bottom:1.5rem; border-bottom:1px solid var(--line); }
.sub { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.1rem; letter-spacing:-.02em; margin:1.6rem 0 .7rem; }
.note { color:var(--mute); font-size:.88rem; margin:.2rem 0 1rem; }

label p, [data-testid="stWidgetLabel"] p { color:var(--mute) !important; font-size:.9rem !important; }
[data-testid="stFileUploaderDropzone"] { background:var(--paper); border:1.5px dashed #c5cad4; border-radius:16px; transition:border-color .3s, background .3s; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--accent); background:var(--tint); }
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input { border-radius:12px !important; }
.stButton > button, [data-testid^="stBaseButton"] { border-radius:12px; padding:.65rem 1.4rem; font-weight:500; border:1px solid var(--line); transition:background .3s, transform .35s cubic-bezier(.16,1,.3,1), border-color .3s; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] { background:var(--ink); border-color:var(--ink); color:#fff; }
.stButton > button[kind="primary"]:hover:not(:disabled), [data-testid="stBaseButton-primary"]:hover:not(:disabled) { background:var(--accent); border-color:var(--accent); transform:translateY(-1px); color:#fff; }
.stButton > button:disabled { opacity:.35; }

[data-testid="stImage"] img { border-radius:14px; border:1px solid var(--line); transition:transform .7s cubic-bezier(.16,1,.3,1), box-shadow .7s; }
[data-testid="stImage"] img:hover { transform:scale(1.015); box-shadow:0 24px 50px -28px rgba(74,47,208,.55); }
[data-testid="stImageCaption"], figcaption { color:var(--mute) !important; font-size:.88rem !important; font-weight:500; }
[data-testid="stMetric"] { background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:1rem 1.2rem; }
[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:-.03em; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; background:var(--card); }

.tbl { width:100%; border-collapse:collapse; margin:.2rem 0 1rem; }
.tbl th { text-align:left; font-weight:500; font-size:.88rem; color:var(--mute); padding:.7rem .9rem .7rem 0; border-bottom:1px solid var(--line); }
.tbl td { padding:.7rem .9rem .7rem 0; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }
.badge { display:inline-block; margin:.2rem 0 1rem; padding:.7rem 1.1rem; border-radius:12px; background:var(--tint); color:var(--accent); font-weight:600; }
.empty { padding:3.5rem 1.5rem; border:1.5px dashed #c5cad4; border-radius:18px; text-align:center; color:var(--mute); background:var(--paper); }
.pred { margin-bottom:1.4rem; }
.pred .l { font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:-.05em; line-height:1; font-size:clamp(2.6rem,6vw,4.6rem); text-transform:capitalize; }
.bars { max-width:460px; margin:1rem 0 1.4rem; }
.bar { display:grid; grid-template-columns:110px 1fr 56px; align-items:center; gap:12px; margin:.5rem 0; font-size:.92rem; color:var(--mute); }
.bar .track { height:7px; border-radius:99px; background:var(--line); overflow:hidden; }
.bar .fill { height:100%; border-radius:99px; background:#c9c2f3; transition:width .8s cubic-bezier(.16,1,.3,1); }
.bar.top .fill { background:var(--accent); } .bar.top { color:var(--ink); font-weight:600; }
@media (prefers-reduced-motion:reduce) { * { animation-duration:.01ms !important; transition-duration:.01ms !important; } }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

MODES = {
    "Calibrate": ("Camera calibration",
                  "Recover the camera matrix and lens distortion from checkerboard photos, then straighten the image.",
                  "Upload 10 to 20 checkerboard photos taken from different angles."),
    "Filter": ("Noise filtering",
               "Add controlled noise to a photo and measure how well each filter brings it back.",
               "Add a photo, pick a noise type and compare the filters."),
    "Edges and FFT": ("Edges and frequency",
                      "See gradients and edges, then work in the Fourier domain with low-pass and high-pass filters.",
                      "Add a photo and tune the edge thresholds and the frequency cutoff."),
    "Features": ("Feature matching",
                 "Find Harris corners, match SIFT keypoints between two views and verify them with RANSAC.",
                 "Add image A, then choose image B to switch on matching."),
    "Classify": ("Fruit classification",
                 "Run the trained Task 2 model on a photo and read the prediction with its confidence.",
                 "Upload a fruit photo and the model answers right away."),
}


# ---------- small UI helpers ----------
def html(s):
    st.markdown(s, unsafe_allow_html=True)


def sub(text):
    html(f'<div class="sub">{escape(text)}</div>')


def empty(text):
    html(f'<div class="empty">{escape(text)}</div>')


def table(headers, rows):
    head = "".join(f"<th>{escape(h)}</th>" for h in headers) if headers else ""
    body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    html(f'<table class="tbl">{"<thead><tr>" + head + "</tr></thead>" if headers else ""}<tbody>{body}</tbody></table>')


def stats(items):
    for col, (k, v) in zip(st.columns(len(items)), items):
        col.metric(k, v)


def show(img, caption=""):
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (255 * (a - a.min()) / (np.ptp(a) + 1e-9)).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    st.image(a, caption=caption, width="stretch")


def grid(items, n=3):
    for i in range(0, len(items), n):
        for col, (title, im) in zip(st.columns(n), items[i:i + n]):
            with col:
                show(im, title)


def decode(data: bytes, max_w=1000):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    s = max_w / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def load_path(path):
    return decode(Path(path).read_bytes())


def get_image(label, key, demo=True):
    """Uploaded photo first, demo scene as a friendly fallback."""
    f = st.file_uploader(label, type=TYPES, key=key)
    if f is not None:
        img = decode(f.getvalue())
        if img is None:
            st.error("Could not read this file as an image.")
        else:
            return img
    if demo and DEMO.exists():
        st.caption("Using the demo scene until you add a photo.")
        return load_path(DEMO)
    return None


# ---------- compute (cached so sliders feel instant) ----------
@st.cache_data(show_spinner=False)
def compute_filter(img, noise, amount, kernel, sigma):
    clean = to_gray(img)
    noisy = add_gaussian_noise(clean, amount) if noise == "Gaussian" else add_salt_pepper_noise(clean, amount)
    items = [("Original", clean), ("Noisy", noisy), *filtering_outputs(noisy, kernel, sigma).items()]
    rows = [(t, mse(clean, im), psnr(clean, im)) for t, im in items[1:]]
    return items, rows


@st.cache_data(show_spinner=False)
def compute_edges(img, low, high, cutoff):
    g = to_gray(img)
    e, f = edges_outputs(g, low=low, high=high), frequency_outputs(g, cutoff=cutoff)
    spatial = [("Original", g), ("Gradient magnitude", e["magnitude"]), ("Sobel Gx", e["gx"]), ("Canny edges", e["canny"])]
    freq = [("Log spectrum", f["spectrum_log"]), ("Ideal low pass", f["ideal_low_img"]),
            ("Ideal high pass", f["ideal_high_img"]), ("Gaussian low pass", f["gaussian_low_img"])]
    return spatial, freq


@st.cache_data(show_spinner=False)
def compute_features(a, b, ratio, npts):
    resp, pts = harris_points(to_gray(a), max_points=npts)
    out = {"harris": [(f"Harris corners, {len(pts)} found", draw_harris(a, pts)), ("Harris response", np.log1p(1000 * np.maximum(resp, 0) / (resp.max() + 1e-9)))],
           "sift": None}
    if b is not None:
        k1, k2, m = sift_match(a, b, ratio)
        try:
            _, inl = homography_inliers(k1, k2, m)
        except cv2.error:
            inl = 0
        vis = cv2.drawMatches(a, k1, b, k2, sorted(m, key=lambda x: x.distance)[:40], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        out["sift"] = {"stats": [("Keypoints in A", len(k1)), ("Keypoints in B", len(k2)),
                                 ("Good matches", len(m)), ("RANSAC inliers", inl)], "image": vis}
    return out


def find_corners(gray, pattern):
    ok, c = cv2.findChessboardCornersSB(gray, pattern)
    if ok:
        return True, c.astype(np.float32)
    ok, c = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return False, None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, .001)
    return True, cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)


@st.cache_data(show_spinner=False)
def compute_calibration(blobs, names, cols, rows, sq_mm):
    pattern = (cols, rows)
    world = np.zeros((cols * rows, 3), np.float32)
    world[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * sq_mm / 1000
    objp, imgp, shown, used, size, first = [], [], [], [], None, None
    skipped = 0
    for data, name in zip(blobs, names):
        img = decode(data, 1600)
        if img is None:
            skipped += 1
            continue
        g = to_gray(img)
        if size is None:
            size, first = g.shape[::-1], img
        elif g.shape[::-1] != size:
            skipped += 1
            continue
        ok, c = find_corners(g, pattern)
        if ok:
            objp.append(world.copy())
            imgp.append(c)
            used.append(name)
            shown.append(cv2.drawChessboardCorners(img.copy(), pattern, c, ok))
    if len(objp) < 3:
        return {"error": "Fewer than 3 boards were detected. Check the corner counts, sharpness and lighting."}
    rms, K, dist, rv, tv = cv2.calibrateCamera(objp, imgp, size, None, None)
    if not (np.isfinite(K).all() and np.isfinite(dist).all()):
        return {"error": "Calibration produced invalid values. Inspect the photos and try again."}
    errs = []
    for name, ob, im, r, t in zip(used, objp, imgp, rv, tv):
        proj, _ = cv2.projectPoints(ob, r, t, K, dist)
        d = np.linalg.norm(im.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        errs.append((name, float(d.mean()), float(np.sqrt(np.mean(d ** 2)))))
    return {"found": len(objp), "total": len(blobs), "skipped": skipped, "rms": rms, "K": K, "dist": dist.ravel(),
            "errs": errs, "original": first, "undistorted": cv2.undistort(first, K, dist), "corners": shown[:6]}


# ---------- Task 2 model ----------
def _load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, TASK2 / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner=False)
def load_fruit_model():
    """Load FruitCNN in this process. Raises ImportError when torch is not installed here."""
    import torch
    net = _load_module("fruit_model", "src/model.py")
    prep = _load_module("fruit_data_utils", "src/data_utils.py")
    try:
        saved = torch.load(MODEL, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(MODEL, map_location="cpu")
    classes = {i: n for n, i in saved["class_to_idx"].items()}
    model = net.FruitCNN(n_classes=len(classes))
    model.load_state_dict(saved["model_state"])
    model.eval()
    return torch, model, classes, prep.make_transform(False, saved["image_size"], saved["mean"], saved["std"])


SUBPROCESS_CODE = """
import json, sys
from PIL import Image
from src.inference import load_model, predict_pil
model, idx_to_class, transform = load_model("artifacts/model.pth")
label, score = predict_pil(model, idx_to_class, transform, Image.open(sys.argv[1]))
print(json.dumps({"label": label, "score": score}))
"""


def task2_python():
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        if (TASK2 / rel).exists():
            return str(TASK2 / rel)
    return sys.executable


def classify(img):
    if not MODEL.exists():
        return {"error": f"No trained model at {MODEL}. Run train.py in FruitSort_Task2 first."}
    try:
        from PIL import Image
        torch, model, classes, transform = load_fruit_model()
        x = transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)[0].tolist()
        scores = sorted(((classes[i], s) for i, s in enumerate(p)), key=lambda t: -t[1])
        return {"label": scores[0][0], "score": scores[0][1], "scores": scores}
    except ImportError:
        pass  # torch is not in this environment: use the Task 2 environment instead
    except Exception as err:
        return {"error": f"The model could not run: {err}"}
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    try:
        cv2.imwrite(path, img)
        proc = subprocess.run([task2_python(), "-c", SUBPROCESS_CODE, path], cwd=str(TASK2),
                              capture_output=True, text=True, timeout=180)
    finally:
        Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        last = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        return {"error": f"The model could not run: {last}"}
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    r["scores"] = [(r["label"], r["score"])]
    return r


# ---------- page ----------
html('<div class="brand"><span class="dot"></span>Vision Lab</div>'
     '<div class="hero"><div class="headline">Teach a camera<br>to see fruit.</div>'
     '<div class="lead">Pick a tool below, add a photo on the left and read the result on the right.</div></div>')

with st.container(key="nav"):
    mode = st.segmented_control("Tool", list(MODES), default="Calibrate", label_visibility="collapsed", key="mode") or "Calibrate"

title, desc, hint = MODES[mode]
left, right = st.columns([1.15, 2.5], gap="large")
controls, results = left.container(key="controls"), right.container(key="results")

with controls:
    html(f'<div class="hint">{escape(hint)}</div>')
    if mode == "Calibrate":
        files = st.file_uploader("Checkerboard photos", type=TYPES, accept_multiple_files=True)
        a, b = st.columns(2)
        cols = a.number_input("Corners, columns", 3, 20, 9)
        rows = b.number_input("Corners, rows", 3, 20, 6)
        sq = st.number_input("Square size (mm)", 1.0, 100.0, 18.0, step=0.5)
        go = st.button("Run calibration", type="primary", disabled=len(files) < 3, width="stretch")
    elif mode == "Filter":
        img = get_image("Photo", "f")
        noise = st.segmented_control("Noise type", ["Gaussian", "Salt and pepper"], default="Gaussian", key="noise") or "Gaussian"
        amt = st.slider("Noise sigma", 5.0, 60.0, 20.0) if noise == "Gaussian" else st.slider("Noise ratio", 0.01, 0.20, 0.05)
        k = st.slider("Kernel size", 3, 15, 5, step=2)
        sg = st.slider("Gaussian sigma", 0.5, 5.0, 1.2)
    elif mode == "Edges and FFT":
        img = get_image("Photo", "e")
        lo, hi = st.slider("Canny thresholds", 0, 255, (80, 160))
        cut = st.slider("FFT cutoff radius", 5, 150, 26)
    elif mode == "Features":
        a_img = get_image("Image A", "a")
        src = st.segmented_control("Image B", ["Sample", "Unrelated", "Upload"], default="Sample", key="bsrc") or "Sample"
        b_img = None
        if src == "Upload":
            fb = st.file_uploader("Image B, same scene", type=TYPES, key="b")
            b_img = decode(fb.getvalue()) if fb is not None else None
        else:
            p = SAMPLE_SAME if src == "Sample" else SAMPLE_OTHER
            b_img = load_path(p) if p.exists() else None
        ratio = st.slider("Ratio test threshold", 0.5, 0.95, 0.75)
        npts = st.slider("Max Harris points", 20, 300, 80, step=10)
    else:
        img = get_image("Fruit photo", "c", demo=False)

with results:
    html(f'<div class="rhead"><div class="rtitle">{escape(title)}</div><div class="rdesc">{escape(desc)}</div></div>')

    if mode == "Calibrate":
        if len(files) < 3:
            empty("Upload at least 3 checkerboard photos to begin.")
        else:
            sig = (tuple(f.name + str(f.size) for f in files), cols, rows, sq)
            if go:
                with st.spinner("Calibrating"):
                    st.session_state["cal"] = (sig, compute_calibration(tuple(f.getvalue() for f in files),
                                                                       tuple(f.name for f in files), cols, rows, sq))
            saved = st.session_state.get("cal")
            if not saved or saved[0] != sig:
                empty("Photos are ready. Press Run calibration.")
            elif "error" in saved[1]:
                st.error(saved[1]["error"])
            else:
                r = saved[1]
                stats([("Boards detected", f"{r['found']} of {r['total']}"), ("RMS error (px)", f"{r['rms']:.4f}"),
                       ("Focal length (px)", f"{r['K'][0, 0]:.0f}")])
                if r["skipped"]:
                    st.caption(f"{r['skipped']} file(s) skipped because of an unreadable file or a different resolution.")
                sub("Camera matrix K")
                table(None, [[f"{v:.3f}" for v in row] for row in r["K"]])
                sub("Distortion coefficients")
                table(["k1", "k2", "p1", "p2", "k3"], [[f"{v:.5f}" for v in r["dist"][:5]]])
                sub("Before and after")
                grid([("Original", r["original"]), ("Undistorted", r["undistorted"])], 2)
                sub("Reprojection error per view")
                table(["View", "Mean (px)", "RMS (px)"], [(n, f"{a:.4f}", f"{b:.4f}") for n, a, b in r["errs"]])
                with st.expander("Detected corners"):
                    grid([("View", s) for s in r["corners"]], 2)

    elif mode == "Filter":
        if img is None:
            empty("Add a photo to begin.")
        else:
            items, rows_ = compute_filter(img, noise, amt, k, sg)
            grid(items, 3)
            sub("Scores against the clean image")
            table(["Method", "MSE", "PSNR (dB)"], [(t, f"{a:.3f}", f"{b:.3f}") for t, a, b in rows_])
            best = max(rows_[1:], key=lambda x: x[2])
            html(f'<div class="badge">Best filter: {escape(best[0])} at {best[2]:.2f} dB</div>')

    elif mode == "Edges and FFT":
        if img is None:
            empty("Add a photo to begin.")
        else:
            spatial, freq = compute_edges(img, lo, hi, cut)
            t1, t2 = st.tabs(["Gradients and edges", "Frequency domain"])
            with t1:
                grid(spatial, 2)
            with t2:
                grid(freq, 2)

    elif mode == "Features":
        if a_img is None:
            empty("Add image A to begin.")
        else:
            r = compute_features(a_img, b_img, ratio, npts)
            sub("Harris corners")
            grid(r["harris"], 2)
            sub("SIFT matching")
            if r["sift"] is None:
                empty("Choose image B to run SIFT matching.")
            else:
                stats(r["sift"]["stats"])
                show(r["sift"]["image"], "Top 40 SIFT matches after the ratio test")

    else:
        if img is None:
            empty("Upload a fruit photo to classify it.")
        else:
            with st.spinner("Thinking"):
                r = classify(img)
            if "error" in r:
                show(img, "Input")
                st.error(r["error"])
            else:
                label = str(r["label"]).replace("_", " ")
                html(f'<div class="pred"><div class="l">{escape(label)}</div>'
                     f'<div class="rdesc">Model score {r["score"] * 100:.1f} percent</div></div>')
                bars = "".join(f'<div class="bar{" top" if i == 0 else ""}"><span>{escape(str(n).replace("_", " "))}</span>'
                               f'<div class="track"><div class="fill" style="width:{s * 100:.1f}%"></div></div>'
                               f'<span>{s * 100:.1f}%</span></div>' for i, (n, s) in enumerate(r["scores"]))
                html(f'<div class="bars">{bars}</div>')
                show(img, "Input")
                html('<div class="note">Scores come from softmax and are not calibrated probabilities.</div>')
