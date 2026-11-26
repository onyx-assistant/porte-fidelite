---
language:
- fr
license: mit
---

# porte-fidelite — la porte de fidélité d'Onyx

Avant qu'une image d'Onyx soit montrée — et à plus forte raison publiée —
elle passe une **porte** : un ensemble de contrôles automatiques, entièrement
locaux, qui la comparent à la référence officielle du personnage. Si ça ne
ressemble pas à Onyx, ça ne sort pas.

Onyx est une peluche virtuelle fabriquée dans un atelier à Montréal.
Son apparence est **verrouillée** : mêmes couleurs, mêmes proportions, mêmes
traits, partout et toujours. Ce dépôt contient les deux instruments qui
gardent cette constance — un pour les images, un pour la voix.

## 1. `qa-gate/` — la porte image

Trois contrôles, aucun service externe, tout tourne hors ligne après
l'installation :

1. **Palette (déterministe)** — les couleurs dominantes du personnage sont
   comparées aux couleurs de la charte graphique en **ΔE CIEDE2000**
   (espace Lab, D65). Verdict PASS si ΔE médian ≤ 6,5 **et** couverture
   conforme ≥ 85 %.
2. **Proportions (heuristique)** — largeur de tête estimée / hauteur de
   silhouette, comparée aux cadrages canoniques de la charte (buste ≈ 0,70 ;
   corps entier ≈ 0,47).
3. **Similarité (pré-filtre)** — cosinus **DINOv2-small** (ONNX, CPU) entre
   l'image recadrée et la référence. C'est un pré-filtre seulement : la
   palette et les proportions font foi pour les écarts fins, et l'humain
   tranche au final.

Verdicts par image : **PASS / À SURVEILLER / REJET**, en résumé texte et en
rapport JSON complet. Codes de sortie pensés pour les scripts : `0` tout
PASS, `1` au moins un À SURVEILLER, `2` au moins un REJET.

```bash
cd qa-gate
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python qa_gate.py image1.png image2.jpg --out rapport.json
```

Le détail complet — calibration des seuils, limites honnêtes de chaque
mesure, résultats des tests négatifs (une teinte décalée de +18° fait tomber
la couverture à 7 % : REJET) — est dans [`qa-gate/README.md`](qa-gate/README.md).

**Ce qui n'est pas ici, volontairement** : la référence canonique du
personnage (le fichier maître ne quitte jamais l'atelier), le modèle ONNX
DINOv2 (tiers, retéléchargeable depuis `onnx-community/dinov2-small`), et les
images de test du personnage. Fournissez votre propre référence avec `--ref`.

## 2. `verificateur-voix/` — la porte voix

La voix d'Onyx est, elle aussi, verrouillée (la voix « nº 1 », dite *la
master*). `verifier-voix.py` mesure la **fréquence fondamentale médiane**
d'un enregistrement et vérifie qu'elle reste dans la bande de référence de
la voix master (**123–151 Hz**, référence 137 Hz). Hors bande : la prise
n'est pas la bonne voix, elle ne part pas au montage.

```bash
python3 verificateur-voix/verifier-voix.py prise.wav
```

## Place dans la chaîne de l'atelier

La porte est le **dernier maillon avant l'humain** : génération (ComfyUI,
LTX, Blender…) → porte de fidélité → œil de l'atelier → publication. Elle
transforme « générer 50 images, en regarder 50 » en « générer 50 images,
regarder les quelques-unes qui passent ». Les gabarits 3D du dépôt
`gabarits-porte-3d` appliquent le même principe au modèle 3D, avec leurs
propres seuils (proportions ≤ 3 %, traits ≤ 2 %, couleurs de la charte).

## Licence

Code sous licence MIT (voir `LICENCE.md`). Le personnage Onyx et sa
référence canonique ne font pas partie de ce dépôt et restent la propriété
de l'atelier.
