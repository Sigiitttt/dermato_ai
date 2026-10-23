import base64
import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torchvision.transforms as T
import timm

st.set_page_config(
    page_title="DermatoAI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

@property --ang { syntax: '<angle>'; initial-value: 0deg; inherits: false; }

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(900px 520px at 6% -12%, rgba(56,189,248,0.16), transparent 60%),
        radial-gradient(800px 520px at 102% 0%, rgba(168,85,247,0.14), transparent 60%),
        #080D1B;
    color: #E2E8F0;
}
.stApp::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
}
.ambient {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: radial-gradient(650px 420px at 78% 38%, var(--amb, transparent), transparent 70%);
    animation: breathe 5s ease-in-out infinite;
}
@keyframes breathe { 50% { opacity: .55; } }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1340px; position: relative; z-index: 1; }

/* Topbar */
.topbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 22px; margin-bottom: 26px;
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.09); border-radius: 18px;
    backdrop-filter: blur(14px); box-shadow: 0 10px 40px -20px rgba(0,0,0,0.7);
}
.brand { display: flex; align-items: center; gap: 14px; }
.brand-logo {
    width: 46px; height: 46px; border-radius: 14px; font-size: 1.4rem;
    background: linear-gradient(135deg, #38BDF8, #A855F7);
    display: flex; align-items: center; justify-content: center;
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { box-shadow: 0 0 18px rgba(99,102,241,0.45);} 50% { box-shadow: 0 0 36px rgba(56,189,248,0.75);} }
.brand-name { font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #F8FAFC, #94A3B8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.brand-sub { font-size: 0.78rem; color: #94A3B8; font-weight: 500; }
.chip {
    display: inline-flex; align-items: center; gap: 7px; padding: 6px 14px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; background: rgba(56,189,248,0.12); color: #7DD3FC;
    border: 1px solid rgba(56,189,248,0.3);
}
.chip.green { background: rgba(52,211,153,0.12); color: #6EE7B7; border-color: rgba(52,211,153,0.3); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: #34D399; animation: blink 1.6s infinite; }
@keyframes blink { 50% { opacity: .25; } }

.sec { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: #64748B; margin: 4px 0 12px 0; display: flex; align-items: center; gap: 10px; }
.sec span { color: #38BDF8; font-family: 'JetBrains Mono', monospace; }
.sec::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(255,255,255,0.12), transparent); }

/* Scanner preview */
.scan {
    position: relative; border-radius: 18px; overflow: hidden; margin-top: 4px;
    border: 1px solid rgba(56,189,248,0.35); box-shadow: 0 0 40px -14px rgba(56,189,248,0.6);
}
.scan img { display: block; width: 100%; }
.scan::before {
    content: ""; position: absolute; left: 0; right: 0; height: 70px; z-index: 2;
    background: linear-gradient(180deg, transparent, rgba(56,189,248,0.38), transparent);
    border-bottom: 2px solid rgba(125,211,252,0.9);
    animation: sweep 2.8s ease-in-out infinite;
}
@keyframes sweep { 0% { top: -70px; } 100% { top: 100%; } }
.scan .cn { position: absolute; width: 22px; height: 22px; z-index: 3; border: 2px solid #7DD3FC; }
.scan .tl { top: 10px; left: 10px; border-right: none; border-bottom: none; border-radius: 6px 0 0 0; }
.scan .tr { top: 10px; right: 10px; border-left: none; border-bottom: none; border-radius: 0 6px 0 0; }
.scan .bl { bottom: 10px; left: 10px; border-right: none; border-top: none; border-radius: 0 0 0 6px; }
.scan .br { bottom: 10px; right: 10px; border-left: none; border-top: none; border-radius: 0 0 6px 0; }
.scan .tagline { position: absolute; left: 14px; bottom: 14px; z-index: 3; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: #7DD3FC; background: rgba(8,13,27,0.7); padding: 3px 9px; border-radius: 6px; letter-spacing: 0.08em; }

/* Result card - rotating conic border */
.result-wrap {
    padding: 2px; border-radius: 24px; animation: rise .6s ease, spin 6s linear infinite;
    background: conic-gradient(from var(--ang), var(--c), transparent 25%, var(--c2) 50%, transparent 75%, var(--c));
    box-shadow: 0 0 60px -16px var(--c);
}
@keyframes spin { to { --ang: 360deg; } }
@keyframes rise { from { opacity: 0; transform: translateY(12px);} to { opacity: 1; transform: none; } }
.result {
    display: flex; align-items: center; gap: 30px; padding: 28px 32px; border-radius: 22px;
    background: radial-gradient(520px 240px at 100% 0%, var(--cg), transparent 70%), #0C1224;
}
.gauge { position: relative; width: 150px; height: 150px; flex-shrink: 0; }
.gauge svg { transform: rotate(-90deg); filter: drop-shadow(0 0 10px var(--c)); }
.gauge .bg { stroke: rgba(255,255,255,0.08); }
.gauge .fg { stroke: url(#gg); stroke-linecap: round; stroke-dasharray: var(--d) 339.3; animation: fill 1.3s cubic-bezier(.2,.8,.2,1); }
@keyframes fill { from { stroke-dasharray: 0 339.3; } }
.gauge .num { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.gauge .num b { font-size: 2rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.04em; line-height: 1; }
.gauge .num i { font-style: normal; font-size: 0.66rem; color: #94A3B8; letter-spacing: 0.14em; text-transform: uppercase; margin-top: 5px; }
.rinfo .lbl { font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase; color: #94A3B8; font-weight: 700; }
.rinfo .cls { font-size: 2.2rem; font-weight: 800; line-height: 1.12; margin: 6px 0 8px 0; letter-spacing: -0.03em;
    background: linear-gradient(90deg, var(--c), var(--c2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;
    background: rgba(255,255,255,0.05); color: var(--c); border: 1px solid var(--c); }
.rinfo .act { margin-top: 14px; font-size: 0.88rem; color: #CBD5E1; font-weight: 500; }

/* Risk spectrum */
.risk { margin-top: 20px; padding: 16px 20px; border-radius: 16px; background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08); }
.risk .hd { display: flex; justify-content: space-between; font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase; color: #64748B; font-weight: 700; }
.risk .hd b { color: #F1F5F9; font-family: 'JetBrains Mono', monospace; letter-spacing: 0; }
.spec { position: relative; height: 12px; border-radius: 999px; margin: 14px 0 8px 0;
    background: linear-gradient(90deg, #34D399, #FACC15 50%, #F87171); }
.spec .mk { position: absolute; top: -5px; width: 4px; height: 22px; border-radius: 3px; background: #fff; left: var(--x);
    box-shadow: 0 0 12px #fff; animation: slide 1.2s cubic-bezier(.2,.8,.2,1); transform: translateX(-50%); }
@keyframes slide { from { left: 0; } }
.risk .lb { display: flex; justify-content: space-between; font-size: 0.72rem; color: #94A3B8; font-weight: 600; }

/* Bars */
.prow { display: flex; align-items: center; gap: 14px; margin: 13px 0; }
.pname { width: 150px; font-size: 0.82rem; font-weight: 600; color: #CBD5E1; }
.ptrack { flex: 1; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.07); overflow: hidden; }
.pfill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--c), var(--c2)); animation: grow 1s ease; box-shadow: 0 0 14px var(--c); }
@keyframes grow { from { width: 0; } }
.pval { width: 58px; text-align: right; font-size: 0.85rem; font-weight: 700; color: #F1F5F9; font-family: 'JetBrains Mono', monospace; }

/* Stat cards */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 18px; }
.stat { padding: 13px 15px; border-radius: 14px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); transition: all .2s; }
.stat:hover { border-color: rgba(56,189,248,0.5); transform: translateY(-2px); }
.stat .k { font-size: 0.64rem; color: #64748B; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 700; }
.stat .v { font-size: 1.05rem; font-weight: 700; color: #F1F5F9; margin-top: 3px; font-family: 'JetBrains Mono', monospace; }

/* Pipeline */
.pipe { display: flex; align-items: stretch; gap: 0; margin-top: 4px; }
.node { flex: 1; min-width: 0; position: relative; animation: rise .6s ease both; }
.node:nth-child(1) { animation-delay: .00s; } .node:nth-child(3) { animation-delay: .10s; }
.node:nth-child(5) { animation-delay: .20s; } .node:nth-child(7) { animation-delay: .30s; }
.node:nth-child(9) { animation-delay: .40s; } .node:nth-child(11) { animation-delay: .50s; }
.card { padding: 8px; border-radius: 16px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
    transition: all .25s; }
.card:hover { transform: translateY(-5px) scale(1.03); border-color: #38BDF8; box-shadow: 0 18px 36px -14px rgba(56,189,248,0.55); }
.card img { width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 10px; display: block; }
.card.hot { border-color: rgba(251,146,60,0.55); box-shadow: 0 0 28px -10px rgba(251,146,60,0.7); }
.card.hot:hover { border-color: #FB923C; box-shadow: 0 18px 36px -14px rgba(251,146,60,0.7); }
.cap { text-align: center; font-size: 0.74rem; font-weight: 600; color: #94A3B8; padding: 8px 0 3px 0; }
.cap b { display: inline-flex; width: 18px; height: 18px; align-items: center; justify-content: center; margin-right: 6px;
    border-radius: 50%; background: rgba(56,189,248,0.2); color: #7DD3FC; font-size: 0.66rem; }
.cap.hot b { background: rgba(251,146,60,0.22); color: #FDBA74; }
.arrow { width: 26px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; color: #475569; font-size: 1.1rem; padding-bottom: 26px; }

/* Widgets */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,0.04); padding: 4px; border-radius: 12px; }
.stTabs [data-baseweb="tab"] { border-radius: 9px; color: #94A3B8; font-weight: 600; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: rgba(56,189,248,0.16); color: #7DD3FC !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,0.03); border: 1.5px dashed rgba(148,163,184,0.35); border-radius: 14px; transition: all .2s; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #38BDF8; background: rgba(56,189,248,0.06); }
[data-testid="stFileUploaderDropzone"] * { color: #CBD5E1 !important; }
.stButton > button { border-radius: 12px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.05); color: #E2E8F0; font-weight: 600; transition: all .2s; }
.stButton > button:hover { border-color: #38BDF8; color: #7DD3FC; transform: translateY(-2px); background: rgba(56,189,248,0.1); }
[data-testid="stExpander"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }
[data-testid="stExpander"] * { color: #CBD5E1; }
label, .stMarkdown p, .stSlider * { color: #CBD5E1; }
[data-testid="stSpinner"] * { color: #7DD3FC !important; }

.empty { min-height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; text-align: center;
    border: 1.5px dashed rgba(148,163,184,0.25); border-radius: 24px; color: #64748B;
    background: radial-gradient(320px 220px at 50% 40%, rgba(56,189,248,0.08), transparent 70%); }
.empty .ico { font-size: 3.2rem; animation: float 3s ease-in-out infinite; }
@keyframes float { 50% { transform: translateY(-9px); } }

.foot { margin-top: 36px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.07); text-align: center; font-size: 0.74rem; color: #64748B; }
</style>
""", unsafe_allow_html=True)

_this_file = Path(__file__).resolve()
if len(_this_file.parents) >= 3 and _this_file.parents[1].name == "kode":
    REPO_ROOT = _this_file.parents[2]
else:
    REPO_ROOT = _this_file.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "best_convnext_tiny_irisan_8W12P_r1.pt"
DEFAULT_MASK_PATH = REPO_ROOT / "models" / "best_mask.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ACC = "83.87%"
MODELS_RELEASE_URL = "https://github.com/Sigiitttt/dermato_ai/releases/download/v1.0/models.zip"

CLASS_NAMES = ["melanoma", "nevus", "seborrheic_keratosis"]
CLASS_LABELS = {
    "melanoma": ("Melanoma", "Ganas · Malignant", "#F87171", "#FB923C", "rgba(248,113,113,0.22)", "Prioritas: rujuk segera untuk biopsi."),
    "nevus": ("Nevus", "Jinak · Benign", "#34D399", "#2DD4BF", "rgba(52,211,153,0.20)", "Tahi lalat jinak, pantau berkala (ABCDE)."),
    "seborrheic_keratosis": ("Keratosis Seboroik", "Jinak · Benign", "#38BDF8", "#818CF8", "rgba(56,189,248,0.20)", "Tumor jinak non-kanker."),
}

REF_SIZE = 600
TARGET_SIZE = 224


def remove_hair(img_rgb):
    h, w = img_rgb.shape[:2]
    k = max(9, int(round(9 * max(h, w) / REF_SIZE)) | 1)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(
        hair_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    return cv2.inpaint(img_rgb, hair_mask, 3, cv2.INPAINT_TELEA), hair_mask


def auto_segment_lesion(img_rgb):
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c_max = max(contours, key=cv2.contourArea)
        mask_final = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_final, [c_max], -1, 255, thickness=cv2.FILLED)
        return mask_final
    return mask_clean


def crop_lesion_roi(img_rgb, mask=None, padding_ratio=0.15):
    h, w = img_rgb.shape[:2]
    if mask is None:
        mask = auto_segment_lesion(img_rgb)
    elif mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_bin = (mask > 127).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(mask_clean > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_rgb, mask_clean

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    pad_x = int((x2 - x1 + 1) * padding_ratio)
    pad_y = int((y2 - y1 + 1) * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x + 1)
    y2 = min(h, y2 + pad_y + 1)
    return img_rgb[y1:y2, x1:x2], mask_clean


def apply_enhancement(img_rgb):
    lab = cv2.cvtColor(img_rgb.astype("uint8"), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)


def pad_and_resize(img, target_size=224):
    h, w = img.shape[:2]
    max_dim = max(h, w)
    top = (max_dim - h) // 2
    bottom = (max_dim - h) - top
    left = (max_dim - w) // 2
    right = (max_dim - w) - left
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    return cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)


def ensure_models_downloaded():
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    m_convnext = models_dir / "best_convnext_tiny_irisan_8W12P_r1.pt"
    m_mask = models_dir / "best_mask.pth"

    if not m_convnext.exists() or not m_mask.exists():
        placeholder = st.empty()
        with placeholder.container():
            st.info("Mengunduh bobot model resmi dari GitHub Release (v1.0)... Mohon tunggu sebentar.")
            pbar = st.progress(0.0)
            zip_path = models_dir / "models.zip"
            try:
                import urllib.request
                import zipfile

                req = urllib.request.Request(MODELS_RELEASE_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_file:
                    total_sz = int(resp.headers.get("Content-Length", 194463692))
                    downloaded = 0
                    chunk_sz = 1024 * 512
                    while True:
                        chunk = resp.read(chunk_sz)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        pbar.progress(min(1.0, downloaded / total_sz))

                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(models_dir)

                if zip_path.exists():
                    zip_path.unlink()
                placeholder.success("Bobot model resmi berhasil diunduh dan siap digunakan!")
                time.sleep(1)
                placeholder.empty()
            except Exception as exc:
                placeholder.error(f"Gagal mengunduh bobot model: {exc}")


@st.cache_resource
def load_unet_model(ckpt_path=None):
    ensure_models_downloaded()
    target = Path(ckpt_path) if ckpt_path else DEFAULT_MASK_PATH
    if not target.exists():
        target = DEFAULT_MASK_PATH
    if not target.exists():
        return None
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        model.load_state_dict(torch.load(target, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception:
        return None


def predict_unet_mask(model, img_rgb):
    h_orig, w_orig = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_resized / 255.0 - mean) / std
    t = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(t)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask_256 = (probs > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_256 = cv2.morphologyEx(mask_256, cv2.MORPH_CLOSE, kernel)
    return cv2.resize(mask_256, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)


@st.cache_resource
def load_convnext_model(ckpt_path):
    ensure_models_downloaded()
    target = Path(ckpt_path) if ckpt_path else DEFAULT_MODEL_PATH
    if not target.exists():
        target = DEFAULT_MODEL_PATH

    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=3)
    if target.exists():
        model.load_state_dict(torch.load(target, map_location=DEVICE))
    else:
        st.error(f"Berkas bobot model tidak ditemukan: {target}")
    model.to(DEVICE)
    model.eval()
    return model


NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def to_tensor(img_224):
    return NORM(T.ToTensor()(Image.fromarray(img_224))).unsqueeze(0).to(DEVICE)


def infer_image(model, img_224):
    with torch.no_grad():
        return torch.softmax(model(to_tensor(img_224)), dim=1).cpu().numpy()[0]


def grad_cam(model, img_224, class_idx):
    feats, grads = {}, {}
    layer = model.stages[-1]

    def fwd(_, __, out):
        feats["a"] = out
        out.register_hook(lambda g: grads.__setitem__("a", g))

    handle = layer.register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            model.zero_grad()
            logits = model(to_tensor(img_224))
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    weights = grads["a"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats["a"]).sum(dim=1))[0].detach().cpu().numpy()
    cam = cv2.resize(cam, (img_224.shape[1], img_224.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_224, 0.55, heat, 0.45, 0)


def to_b64(img, max_w=520):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def demo_sample(mode="melanoma"):
    sample_path = Path(__file__).resolve().parent / "samples" / f"{mode}.jpg"
    if sample_path.exists():
        return np.array(Image.open(sample_path).convert("RGB"))
    return np.full((360, 480, 3), 200, dtype=np.uint8)


st.markdown(f"""
<div class="topbar">
<div class="brand">
<div class="brand-logo">🔬</div>
<div><div class="brand-name">DermatoAI</div><div class="brand-sub">Deteksi Lesi Kulit · 3 Kelas</div></div>
</div>
<div style="display:flex; gap:8px; align-items:center;">
<span class="chip">ConvNeXt-Tiny (83,87%)</span>
<span class="chip green"><span class="dot"></span>U-Net Mask (ResNet-34)</span>
</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.3], gap="large")

if "active_img" not in st.session_state:
    st.session_state["active_img"] = None

with col_left:
    st.markdown('<div class="sec"><span>01</span>Input Citra</div>', unsafe_allow_html=True)
    tab_up, tab_quick = st.tabs(["Unggah", "Sampel Demo"])

    with tab_up:
        up_file = st.file_uploader("Foto dermatoskopi", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if up_file is not None:
            active_img = np.array(Image.open(up_file).convert("RGB"))
        else:
            active_img = None

    with tab_quick:
        # Langsung muat citra paten Melanoma asli otomatis
        st.markdown('<div style="font-size:0.85rem; color:#94A3B8; margin-bottom:8px;">Citra paten dermatoskopi Melanoma (Ganas) aktif otomatis.</div>', unsafe_allow_html=True)
        if active_img is None:
            active_img = demo_sample("melanoma")

    with st.expander("Pengaturan Model & Prapemrosesan"):
        pad_val = st.slider("Margin kulit sehat (%)", 5, 30, 15, 5) / 100.0
        show_cam = st.checkbox("Tampilkan Grad-CAM", value=True)
        custom_ckpt = st.text_input("Path model klasifikasi", value=str(DEFAULT_MODEL_PATH))
        custom_mask_ckpt = st.text_input("Path model segmentasi U-Net", value=str(DEFAULT_MASK_PATH))

    if active_img is not None:
        h0, w0 = active_img.shape[:2]
        st.markdown('<div class="sec" style="margin-top:18px;"><span>●</span>Citra Terpilih</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="scan">
<img src="{to_b64(active_img, 700)}"/>
<div class="cn tl"></div><div class="cn tr"></div><div class="cn bl"></div><div class="cn br"></div>
<div class="tagline">SCAN · {w0}×{h0}px</div>
</div>
""", unsafe_allow_html=True)

if active_img is not None:
    with st.spinner("Menganalisis citra..."):
        t0 = time.time()
        clean_img, hair_mask = remove_hair(active_img)

        unet_m = load_unet_model(custom_mask_ckpt)
        if unet_m is not None:
            mask_label = "U-Net ResNet-34"
            lesion_mask_raw = predict_unet_mask(unet_m, clean_img)
        else:
            lesion_mask_raw = auto_segment_lesion(clean_img)
            mask_label = "Otsu Heuristik"

        roi_img, lesion_mask = crop_lesion_roi(clean_img, mask=lesion_mask_raw, padding_ratio=pad_val)
        enh_img = apply_enhancement(roi_img)
        final_img = pad_and_resize(enh_img, target_size=TARGET_SIZE)
        t_prep = (time.time() - t0) * 1000

        t1 = time.time()
        model = load_convnext_model(custom_ckpt)
        probs = infer_image(model, final_img)
        t_inf = (time.time() - t1) * 1000

    pred_idx = int(np.argmax(probs))
    key = CLASS_NAMES[pred_idx]
    title, status, c1_, c2_, cglow, action = CLASS_LABELS[key]
    conf = probs[pred_idx] * 100
    dash = conf / 100 * 339.3
    mel_p = probs[CLASS_NAMES.index("melanoma")] * 100

    st.markdown(f'<div class="ambient" style="--amb:{cglow};"></div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="sec"><span>02</span>Hasil Diagnosis</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="result-wrap" style="--c:{c1_}; --c2:{c2_};">
<div class="result" style="--c:{c1_}; --c2:{c2_}; --cg:{cglow}; --d:{dash:.1f};">
<div class="gauge">
<svg width="150" height="150" viewBox="0 0 120 120">
<defs><linearGradient id="gg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{c1_}"/><stop offset="100%" stop-color="{c2_}"/></linearGradient></defs>
<circle class="bg" cx="60" cy="60" r="54" fill="none" stroke-width="9"/>
<circle class="fg" cx="60" cy="60" r="54" fill="none" stroke-width="9"/>
</svg>
<div class="num"><b>{conf:.1f}%</b><i>keyakinan</i></div>
</div>
<div class="rinfo">
<div class="lbl">Diagnosis Terdeteksi</div>
<div class="cls">{title}</div>
<span class="badge">{status}</span>
<div class="act">{action}</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="risk">
<div class="hd"><span>Spektrum Probabilitas Melanoma</span><b>{mel_p:.1f}%</b></div>
<div class="spec"><div class="mk" style="--x:{mel_p:.1f}%;"></div></div>
<div class="lb"><span>Rendah</span><span>Tinggi</span></div>
</div>
""", unsafe_allow_html=True)

        bars = ""
        for i, k in enumerate(CLASS_NAMES):
            n, _, a, b, _, _ = CLASS_LABELS[k]
            p = probs[i] * 100
            bars += f"""
<div class="prow" style="--c:{a}; --c2:{b};">
<div class="pname">{n}</div>
<div class="ptrack"><div class="pfill" style="width:{p:.1f}%;"></div></div>
<div class="pval">{p:.1f}%</div>
</div>"""
        st.markdown(f'<div style="margin-top:18px;">{bars}</div>', unsafe_allow_html=True)

        st.markdown(f"""
<div class="stats">
<div class="stat"><div class="k">Prep</div><div class="v">{t_prep:.0f} ms</div></div>
<div class="stat"><div class="k">Inferensi</div><div class="v">{t_inf:.0f} ms</div></div>
<div class="stat"><div class="k">Masker</div><div class="v">{mask_label.split()[0]}</div></div>
<div class="stat"><div class="k">Device</div><div class="v">{DEVICE.type.upper()}</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sec" style="margin-top:38px;"><span>03</span>Alur Prapemrosesan & Fokus Model</div>', unsafe_allow_html=True)
    steps = [
        (active_img, "Citra Asli", False),
        (clean_img, "Dull Razor", False),
        (lesion_mask, f"Masker {mask_label}", False),
        (roi_img, f"ROI +{int(pad_val * 100)}%", False),
        (final_img, "Input CLAHE", False),
    ]
    if show_cam:
        steps.append((grad_cam(model, final_img, pred_idx), "Grad-CAM", True))

    nodes = []
    for n, (img, cap, hot) in enumerate(steps, 1):
        h = " hot" if hot else ""
        nodes.append(
            f'<div class="node"><div class="card{h}"><img src="{to_b64(img, 360)}"/></div>'
            f'<div class="cap{h}"><b>{n}</b>{cap}</div></div>'
        )
    pipe = '<div class="arrow">›</div>'.join(nodes)
    st.markdown(f'<div class="pipe">{pipe}</div>', unsafe_allow_html=True)

    st.markdown('<div class="foot">Alat bantu riset akademik. Bukan pengganti diagnosis dokter.</div>', unsafe_allow_html=True)
else:
    with col_right:
        st.markdown('<div class="sec"><span>02</span>Hasil Diagnosis</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="empty">
<div class="ico">🩺</div>
<div style="font-weight:700; color:#94A3B8; font-size:1.05rem;">Belum ada citra</div>
<div style="font-size:0.85rem;">Unggah foto atau pilih sampel demo</div>
</div>
""", unsafe_allow_html=True)
