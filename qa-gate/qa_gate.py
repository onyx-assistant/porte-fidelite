#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Porte qualité Onyx — contrôle automatique d'une image produite vs la référence canonique.

Trois contrôles locaux (aucun service externe, aucune dépendance payante) :

1. PALETTE (déterministe) — couleurs dominantes du personnage comparées aux hex
   de la charte graphique (§3) en ΔE CIEDE2000 (espace Lab, illuminant D65).
2. PROPORTIONS (heuristique) — largeur de tête estimée / hauteur de silhouette,
   comparée aux cadrages canoniques (buste 1,67 T → ≈0,70 ; corps entier
   2,5 T → ≈0,47). Voir README.md pour les limites honnêtes de cette mesure.
3. SIMILARITÉ (embedding) — cosinus DINOv2-small (ONNX, CPU) entre l'image
   recadrée sur le personnage et la référence canonique. Pré-filtre seulement :
   un embedding global ne voit pas les petits écarts d'identité (rapport du
   22 nov. 2026, §5) — c'est la palette et les proportions qui font foi pour
   les écarts fins, et l'humain qui tranche au final.

Verdicts : PASS / À SURVEILLER / REJET, par contrôle et global.
Sortie : résumé texte (stdout) + rapport JSON (--out, défaut qa-report.json).
Code de sortie : 0 = tout PASS, 1 = au moins un À SURVEILLER, 2 = au moins un REJET.

Usage :
    python3 qa_gate.py IMAGE [IMAGE ...] [--ref REF] [--out RAPPORT.json]
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constantes — charte graphique Onyx v0.4 (§3 et §7)
# ---------------------------------------------------------------------------

CHARTE_ORANGES = {
    "orange_lumiere": "#FC8B32",
    "orange_median": "#C95D1D",
    "orange_ombre": "#8D2E0F",
}
CHARTE_AUTRES = {
    "noir_chaud": "#1A1512",   # yeux, sourcils, broderies (~)
    "blanc": "#FFFFFF",
}

# Proportions canoniques (charte §7) : T = hauteur de tête (crâne → menton).
# Tête : largeur max 1,17 T. Buste de référence : hauteur visible 1,67 T.
# Corps entier : 2,5 T + oreilles 0,09 T → silhouette 2,51 T (oreilles incluses
# dans le masque, donc c'est 2,51 qu'on compare).
RATIO_BUSTE = 1.17 / 1.67        # ≈ 0,701
RATIO_CORPS = 1.17 / 2.51        # ≈ 0,466

# Seuils palette (ΔE2000), CALIBRÉS sur la référence canonique le 22 nov. 2026.
# Point d'honnêteté : les 3 hex de la charte sont des échantillons ponctuels
# (lumière / médian / ombre) d'un dégradé continu de pelage. Mesuré sur la
# référence elle-même : ΔE médian au ton le plus proche = 4,83, couverture
# conforme (ΔE ≤ 10) = 96 %. Le plancher intrinsèque de la métrique est donc
# ~4,8 — les seuils PASS ci-dessous gardent une marge de ×1,35 au-dessus de la
# référence, et la couverture fait le gros du travail de rejet (une teinte
# décalée fait chuter la couverture vers 0). Voir tests/RESULTATS.md.
PALETTE_PASS_DE = 6.5            # ΔE médian orange → PASS
PALETTE_WARN_DE = 10.0           # → À SURVEILLER ; au-delà → REJET
PALETTE_PASS_COV = 0.85          # part des pixels « orange conformes » (ΔE ≤ 10)
PALETTE_WARN_COV = 0.65

# Seuils proportions (écart relatif au cadrage canonique le plus proche)
PROP_PASS_DEV = 0.15
PROP_WARN_DEV = 0.40             # au-delà des deux cadrages → REJET

# Seuils embedding (cosinus DINOv2) — calibrés le 22 nov. 2026 sur la paire
# référence / composite reprojeté (voir tests/RESULTATS.md). Pré-filtre.
EMB_PASS = 0.80
EMB_WARN = 0.60

WORK_SIZE = 640                  # côté max pour masque / palette / proportions
MIN_MASK_PIXELS = 500

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "dinov2-small.onnx"
REF_CACHE = BASE_DIR / "models" / "reference_embedding.json"
DEFAULT_REF = BASE_DIR.parent.parent / "references" / "onyx-canonique.webp"

VERDICTS = ["PASS", "À SURVEILLER", "REJET", "NON MESURABLE"]
_RANK = {"PASS": 0, "À SURVEILLER": 1, "NON MESURABLE": 1, "REJET": 2}


def worst(*verdicts):
    return max(verdicts, key=lambda v: _RANK[v])


# ---------------------------------------------------------------------------
# Science des couleurs : sRGB → Lab (D65) et ΔE CIEDE2000 (numpy, vectorisé)
# ---------------------------------------------------------------------------

def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def rgb_to_lab(rgb):
    """rgb : (..., 3) valeurs 0–255 → Lab (..., 3), illuminant D65."""
    rgb = rgb.astype(np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # sRGB linéaire → XYZ (D65)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz / white
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e2000(lab1, lab2):
    """ΔE CIEDE2000 entre deux jeux de couleurs Lab (formulation de Sharma)."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0
    Cbar7 = Cbar ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    h1p = np.where(C1p == 0, 0.0, h1p)
    h2p = np.where(C2p == 0, 0.0, h2p)
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dhp = np.where(C1p * C2p == 0, 0.0,
                   np.where(np.abs(dh) <= 180, dh,
                            np.where(dh > 180, dh - 360.0, dh + 360.0)))
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hbarp = np.where(C1p * C2p == 0, hsum,
                     np.where(np.abs(h1p - h2p) <= 180, hsum / 2.0,
                              np.where(hsum < 360, (hsum + 360.0) / 2.0,
                                       (hsum - 360.0) / 2.0)))
    T = (1.0 - 0.17 * np.cos(np.radians(hbarp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbarp))
         + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0)))
    dtheta = 30.0 * np.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp ** 7
    Rc = 2.0 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7))
    Sl = 1.0 + (0.015 * (Lbarp - 50.0) ** 2) / np.sqrt(20.0 + (Lbarp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc
    tL = dLp / Sl
    tC = dCp / Sc
    tH = dHp / Sh
    return np.sqrt(tL ** 2 + tC ** 2 + tH ** 2 + Rt * tC * tH)


CHARTE_LAB = {nom: rgb_to_lab(hex_to_rgb(hx)) for nom, hx in
              {**CHARTE_ORANGES, **CHARTE_AUTRES}.items()}
ORANGE_LABS = np.stack([CHARTE_LAB[n] for n in CHARTE_ORANGES])
ALL_CHARTE_LABS = np.stack(list(CHARTE_LAB.values()))
ALL_CHARTE_NAMES = list(CHARTE_LAB.keys())


# ---------------------------------------------------------------------------
# Chargement d'image, aplatissement, masque du personnage
# ---------------------------------------------------------------------------

def load_image(path):
    img = Image.open(path)
    return img.convert("RGBA")


def flatten_on_white(rgba):
    """Composite RGBA sur fond blanc → RGB (le fond d'identité de la charte)."""
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def character_mask(rgba_small):
    """
    Masque booléen du personnage sur l'image réduite.
    - Si l'alpha est informatif → seuil alpha.
    - Sinon → soustraction du fond : couleur de fond = médiane du cadre
      extérieur ; premier plan = distance Lab au fond > 12. On garde ensuite
      la plus grande composante connexe (le personnage).
    Retourne (masque, méthode).
    """
    alpha = np.asarray(rgba_small)[..., 3]
    if (alpha < 250).mean() > 0.005:
        return alpha >= 128, "alpha"
    rgb = np.asarray(rgba_small.convert("RGB")).astype(np.float64)
    h, w = rgb.shape[:2]
    frame = np.concatenate([
        rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3),
        rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3)])
    bg_lab = rgb_to_lab(np.median(frame, axis=0))
    lab = rgb_to_lab(rgb)
    dist = np.sqrt(((lab - bg_lab) ** 2).sum(axis=-1))
    mask = dist > 12.0
    mask = _largest_component(mask)
    return mask, "soustraction du fond (médiane du cadre, ΔE76 > 12)"


def _largest_component(mask):
    """Plus grande composante connexe (4-connexité), BFS sur grille réduite."""
    from collections import deque
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    best_label, best_size, cur = 0, 0, 0
    for y in range(h):
        row = mask[y]
        for x in np.flatnonzero(row & (labels[y] == 0)):
            cur += 1
            q = deque([(y, int(x))])
            labels[y, x] = cur
            size = 0
            while q:
                cy, cx = q.popleft()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = cur
                        q.append((ny, nx))
            if size > best_size:
                best_label, best_size = cur, size
    if best_label == 0:
        return mask
    return labels == best_label


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------------------------------------------------------------------------
# Contrôle 1 — palette
# ---------------------------------------------------------------------------

def check_palette(rgb_small, mask):
    pixels = np.asarray(rgb_small)[mask]
    n = int(pixels.shape[0])
    if n < MIN_MASK_PIXELS:
        return {"verdict": "NON MESURABLE",
                "raison": f"masque trop petit ({n} px < {MIN_MASK_PIXELS})"}
    rng = np.random.default_rng(42)
    sample = pixels if n <= 60000 else pixels[rng.choice(n, 60000, replace=False)]
    sample_lab = rgb_to_lab(sample.astype(np.float64))

    # Distance de chaque pixel à l'orange charte le plus proche + couleur
    # charte la plus proche toutes catégories (pour savoir qui est « orange »).
    d_orange = np.stack([delta_e2000(sample_lab, ol) for ol in ORANGE_LABS], axis=-1)
    d_orange_min = d_orange.min(axis=1)
    d_all = np.stack([delta_e2000(sample_lab, cl) for cl in ALL_CHARTE_LABS], axis=-1)
    nearest_charte = np.array(ALL_CHARTE_NAMES)[d_all.argmin(axis=1)]
    is_orange = np.isin(nearest_charte, list(CHARTE_ORANGES.keys()))

    orange_pixels_de = d_orange_min[is_orange]
    coverage = float((d_orange_min <= 10.0).mean())
    if len(orange_pixels_de) >= 50:
        median_de = float(np.median(orange_pixels_de))
        p90_de = float(np.percentile(orange_pixels_de, 90))
    else:
        median_de, p90_de = None, None

    # Couleurs dominantes (quantification médiane, 6 grappes) pour le rapport.
    side = int(np.ceil(np.sqrt(len(sample))))
    strip = Image.new("RGB", (side, side))
    padded = np.zeros((side * side, 3), dtype=np.uint8)
    padded[:len(sample)] = sample.astype(np.uint8)
    strip.putdata([tuple(px) for px in padded])
    quant = strip.quantize(colors=6, method=Image.MEDIANCUT)
    counts = sorted(quant.getcolors(side * side), reverse=True)
    pal = quant.getpalette()
    clusters = []
    for count, idx in counts:
        if idx is None:
            continue
        rgb_c = np.array(pal[idx * 3:idx * 3 + 3], dtype=np.float64)
        lab_c = rgb_to_lab(rgb_c)
        d_o = [float(delta_e2000(lab_c, ol)) for ol in ORANGE_LABS]
        nearest = ALL_CHARTE_NAMES[int(np.argmin(
            [float(delta_e2000(lab_c, cl)) for cl in ALL_CHARTE_LABS]))]
        clusters.append({
            "rgb": [int(v) for v in rgb_c],
            "hex": "#%02X%02X%02X" % tuple(int(v) for v in rgb_c),
            "part_masque": round(count / max(1, side * side), 4),
            "charte_proche": nearest,
            "delta_e_orange_min": round(min(d_o), 2),
        })

    if median_de is None:
        verdict = "REJET"
        raison = "aucune grappe de pixels ne se rattache aux oranges de la charte"
    elif median_de <= PALETTE_PASS_DE and coverage >= PALETTE_PASS_COV:
        verdict, raison = "PASS", ""
    elif median_de <= PALETTE_WARN_DE and coverage >= PALETTE_WARN_COV:
        verdict, raison = "À SURVEILLER", ""
    else:
        verdict = "REJET"
        raison = (f"ΔE médian orange {median_de} et/ou couverture conforme "
                  f"{coverage:.0%} hors tolérances")
    return {
        "verdict": verdict,
        "raison": raison,
        "delta_e_median_orange": None if median_de is None else round(median_de, 2),
        "delta_e_p90_orange": None if p90_de is None else round(p90_de, 2),
        "couverture_conforme_de10": round(coverage, 3),
        "part_pixels_orange": round(float(is_orange.mean()), 3),
        "grappes_dominantes": clusters,
        "seuils": {"pass_de": PALETTE_PASS_DE, "warn_de": PALETTE_WARN_DE,
                   "pass_couverture": PALETTE_PASS_COV, "warn_couverture": PALETTE_WARN_COV},
    }


# ---------------------------------------------------------------------------
# Contrôle 2 — proportions (heuristique, voir README)
# ---------------------------------------------------------------------------

def check_proportions(mask, img_shape):
    h_img, w_img = img_shape
    frac = float(mask.mean())
    bbox = mask_bbox(mask)
    if bbox is None or frac < 0.01 or frac > 0.98:
        return {"verdict": "NON MESURABLE",
                "raison": f"masque non fiable (part de l'image : {frac:.1%})"}
    x0, y0, x1, y1 = bbox
    H = y1 - y0 + 1
    W = x1 - x0 + 1
    if (x0 == 0 and y0 == 0 and x1 == w_img - 1 and y1 == h_img - 1):
        return {"verdict": "NON MESURABLE",
                "raison": "la silhouette touche les quatre bords : cadrage indéterminable"}
    # Largeur de la silhouette par rangée (étendue min–max, robuste aux trous).
    spans = np.zeros(H, dtype=np.int32)
    for i, y in enumerate(range(y0, y1 + 1)):
        xs = np.flatnonzero(mask[y])
        if len(xs):
            spans[i] = xs.max() - xs.min() + 1
    # Zone « tête » : rangées entre 2 % et 40 % de la hauteur de silhouette.
    zone = spans[int(0.02 * H):max(int(0.02 * H) + 1, int(0.40 * H))]
    head_w = int(zone.max()) if len(zone) else 0
    ratio = head_w / H
    dev_buste = abs(ratio - RATIO_BUSTE) / RATIO_BUSTE
    dev_corps = abs(ratio - RATIO_CORPS) / RATIO_CORPS
    if dev_buste <= dev_corps:
        cadrage, dev, attendu = "buste (référence)", dev_buste, RATIO_BUSTE
    else:
        cadrage, dev, attendu = "corps entier", dev_corps, RATIO_CORPS
    if dev <= PROP_PASS_DEV:
        verdict = "PASS"
    elif dev <= PROP_WARN_DEV:
        verdict = "À SURVEILLER"
    else:
        verdict = "REJET"
    return {
        "verdict": verdict,
        "ratio_largeur_tete_sur_hauteur": round(ratio, 3),
        "cadrage_le_plus_proche": cadrage,
        "ratio_attendu": round(attendu, 3),
        "ecart_relatif": round(dev, 3),
        "silhouette_px": {"largeur": int(W), "hauteur": int(H)},
        "seuils": {"pass_ecart_relatif": PROP_PASS_DEV, "warn_ecart_relatif": PROP_WARN_DEV},
        "note": "heuristique : largeur max de la silhouette entre 2 % et 40 % de sa hauteur ; "
                "ne vaut que pour une vue de face ou quasi de face, personnage entier dans le cadre",
    }


# ---------------------------------------------------------------------------
# Contrôle 3 — similarité d'embedding (DINOv2-small, ONNX, CPU)
# ---------------------------------------------------------------------------

class DinoEmbedder:
    def __init__(self, model_path=MODEL_PATH):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0  # tous les cœurs disponibles
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, rgb_img, bbox=None):
        """Embedding L2-normalisé (vecteur CLS) d'une image RGB PIL."""
        img = rgb_img
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            mx = int(0.05 * (x1 - x0 + 1))
            my = int(0.05 * (y1 - y0 + 1))
            img = img.crop((max(0, x0 - mx), max(0, y0 - my),
                            min(img.width, x1 + mx + 1), min(img.height, y1 + my + 1)))
        # Prétraitement du preprocessor_config.json DINOv2 : resize bord
        # court = 256, crop central 224, normalisation ImageNet.
        w, h = img.size
        scale = 256 / min(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.BILINEAR)
        w, h = img.size
        left, top = (w - 224) // 2, (h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
        arr = np.asarray(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.session.run(None, {self.input_name: arr})[0]
        vec = out[0, 0] if out.ndim == 3 else out[0]
        vec = np.asarray(vec, dtype=np.float64)
        return vec / np.linalg.norm(vec)


def reference_embedding(embedder, ref_path):
    """Embedding de la référence, mis en cache (invalidé par le SHA-256)."""
    sha = hashlib.sha256(Path(ref_path).read_bytes()).hexdigest()
    if REF_CACHE.exists():
        try:
            cache = json.loads(REF_CACHE.read_text())
            if cache.get("ref_sha256") == sha:
                return np.array(cache["embedding"]), True
        except Exception:
            pass
    rgba = load_image(ref_path)
    small = rgba.copy()
    small.thumbnail((WORK_SIZE, WORK_SIZE), Image.LANCZOS)
    mask, _ = character_mask(small)
    rgb = flatten_on_white(rgba)
    # Recadrage aux proportions de la petite image → pleine résolution.
    bbox = mask_bbox(mask)
    if bbox is not None:
        sx = rgba.width / small.width
        sy = rgba.height / small.height
        bbox = tuple(int(round(v * s)) for v, s in
                     zip(bbox, (sx, sy, sx, sy)))
    vec = embedder.embed(rgb, bbox)
    REF_CACHE.write_text(json.dumps(
        {"ref": str(ref_path), "ref_sha256": sha, "embedding": vec.tolist()}))
    return vec, False


def check_embedding(embedder, ref_vec, rgb_full, mask_small, small_size, full_size):
    bbox = mask_bbox(mask_small)
    if bbox is not None:
        sx = full_size[0] / small_size[0]
        sy = full_size[1] / small_size[1]
        bbox = tuple(int(round(v * s)) for v, s in zip(bbox, (sx, sy, sx, sy)))
    t0 = time.perf_counter()
    vec = embedder.embed(rgb_full, bbox)
    dt = time.perf_counter() - t0
    sim = float(np.dot(vec, ref_vec))
    if sim >= EMB_PASS:
        verdict = "PASS"
    elif sim >= EMB_WARN:
        verdict = "À SURVEILLER"
    else:
        verdict = "REJET"
    return {"verdict": verdict, "cosinus": round(sim, 4),
            "temps_calcul_s": round(dt, 2),
            "seuils": {"pass": EMB_PASS, "warn": EMB_WARN},
            "note": "pré-filtre global : ne détecte pas les petits écarts "
                    "d'identité (détails du visage, coutures)"}


# ---------------------------------------------------------------------------
# Pipeline complet pour une image
# ---------------------------------------------------------------------------

def analyse(path, embedder, ref_vec):
    t0 = time.perf_counter()
    rgba = load_image(path)
    full_rgb = flatten_on_white(rgba)
    small = rgba.copy()
    small.thumbnail((WORK_SIZE, WORK_SIZE), Image.LANCZOS)
    small_rgb = flatten_on_white(small)
    mask, mask_method = character_mask(small)

    palette = check_palette(small_rgb, mask)
    proportions = check_proportions(mask, (small.height, small.width))
    embedding = check_embedding(embedder, ref_vec, full_rgb, mask,
                                (small.width, small.height),
                                (rgba.width, rgba.height))
    verdict = worst(palette["verdict"], proportions["verdict"], embedding["verdict"])
    return {
        "fichier": str(path),
        "taille_px": [rgba.width, rgba.height],
        "masque": {"methode": mask_method,
                   "part_image": round(float(mask.mean()), 4)},
        "palette": palette,
        "proportions": proportions,
        "embedding": embedding,
        "verdict": verdict,
        "temps_total_s": round(time.perf_counter() - t0, 2),
    }


def print_summary(res):
    print(f"\n=== {res['fichier']}")
    print(f"    taille {res['taille_px'][0]}×{res['taille_px'][1]} · "
          f"masque : {res['masque']['methode']} ({res['masque']['part_image']:.1%} de l'image)")
    p = res["palette"]
    if p["verdict"] == "NON MESURABLE":
        print(f"    palette     : {p['verdict']} ({p.get('raison', '')})")
    elif p["delta_e_median_orange"] is None:
        print(f"    palette     : {p['verdict']} — aucun pixel rattaché aux oranges "
              f"de la charte, couverture conforme {p['couverture_conforme_de10']:.0%}")
    else:
        print(f"    palette     : {p['verdict']} — ΔE médian orange {p['delta_e_median_orange']}, "
              f"p90 {p['delta_e_p90_orange']}, couverture conforme {p['couverture_conforme_de10']:.0%}")
    pr = res["proportions"]
    if pr["verdict"] == "NON MESURABLE":
        print(f"    proportions : {pr['verdict']} ({pr.get('raison', '')})")
    else:
        print(f"    proportions : {pr['verdict']} — ratio tête/hauteur {pr['ratio_largeur_tete_sur_hauteur']} "
              f"(cadrage proche : {pr['cadrage_le_plus_proche']}, attendu {pr['ratio_attendu']}, "
              f"écart {pr['ecart_relatif']:.0%})")
    e = res["embedding"]
    print(f"    embedding   : {e['verdict']} — cosinus DINOv2 {e['cosinus']} "
          f"({e['temps_calcul_s']} s CPU)")
    print(f"    => VERDICT : {res['verdict']}  (analyse en {res['temps_total_s']} s)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Porte qualité Onyx : palette, proportions, similarité vs référence canonique.")
    ap.add_argument("images", nargs="+", help="image(s) à contrôler")
    ap.add_argument("--ref", default=str(DEFAULT_REF),
                    help=f"référence canonique (défaut : {DEFAULT_REF})")
    ap.add_argument("--out", default="qa-report.json",
                    help="rapport JSON de sortie (défaut : qa-report.json ; '' pour aucun)")
    args = ap.parse_args(argv)

    if not MODEL_PATH.exists():
        print(f"Modèle DINOv2 introuvable : {MODEL_PATH}", file=sys.stderr)
        return 2
    embedder = DinoEmbedder()
    ref_vec, cached = reference_embedding(embedder, Path(args.ref))
    etat_cache = "en cache" if cached else "calculé"
    print(f"Référence : {args.ref} (embedding {etat_cache})")

    results = []
    for img_path in args.images:
        try:
            res = analyse(Path(img_path), embedder, ref_vec)
        except Exception as exc:  # image illisible, etc.
            res = {"fichier": img_path, "verdict": "REJET",
                   "erreur": f"{type(exc).__name__} : {exc}"}
            print(f"\n=== {img_path}\n    => VERDICT : REJET (erreur : {exc})")
        results.append(res)
        if "erreur" not in res:
            print_summary(res)

    if args.out:
        report = {
            "outil": "qa-gate Onyx v1.0",
            "reference": str(args.ref),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "resultats": results,
        }
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nRapport JSON : {args.out}")
    ranks = [_RANK[r["verdict"]] for r in results]
    return max(ranks) if ranks else 0


if __name__ == "__main__":
    sys.exit(main())
