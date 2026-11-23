#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génère l'image négative de test : la référence canonique avec une rotation
de teinte de +70° (l'orange devient verdâtre) — mêmes formes, couleurs fausses.
La porte qualité doit la SIGNALER (rejet palette attendu)."""
from pathlib import Path
import colorsys

from PIL import Image

BASE = Path(__file__).resolve().parent
REF = BASE.parent.parent.parent / "references" / "onyx-canonique.webp"
OUT = BASE / "negative-teinte-decalee.png"

img = Image.open(REF).convert("RGB")
hsv = img.convert("HSV")
import numpy as np
arr = np.asarray(hsv).astype(np.int16)
arr[..., 0] = (arr[..., 0] + int(70 / 360 * 255)) % 256  # +70° de teinte
Image.fromarray(arr.astype(np.uint8), "HSV").convert("RGB").save(OUT)
print(f"Image négative écrite : {OUT}")
