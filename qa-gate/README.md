# Porte qualité Onyx (`qa-gate`)

_Construite le 22 novembre 2026 par l'atelier, à partir d'une recherche interne sur les outils de contrôle visuel._

## À quoi ça sert

Dans le pipeline de production, chaque image nouvelle d'Onyx (génération Qwen/FLUX, édition, rendu) passe par cette porte **avant** d'être montrée à l'atelier. La porte compare l'image à la référence canonique (`references/onyx-canonique.webp`) et rend un verdict par image :

- **PASS** — rien à signaler, l'image peut être montrée ;
- **À SURVEILLER** — un contrôle est en zone limite ou non mesurable : œil humain requis ;
- **REJET** — un contrôle échoue franchement : l'image ne doit pas être montrée comme un résultat valide.

Objectif (rapport §5) : transformer « générer 50 images, en regarder 50 » en « générer 50 images, regarder les quelques-unes qui passent ».

## Installation

Tout est local, gratuit, sans service externe. Le modèle DINOv2 (88 Mo) est déjà dans `models/` (source unique : `onnx-community/dinov2-small` sur Hugging Face, téléchargé une fois le 22 nov. 2026) ; après l'installation, l'outil fonctionne **hors ligne**.

```bash
cd <racine-du-projet>/tools/qa-gate
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # numpy, Pillow, onnxruntime
```

## Usage

```bash
# Une ou plusieurs images ; rapport JSON par défaut dans qa-report.json
.venv/bin/python qa_gate.py image1.png image2.jpg --out rapport.json

# Référence différente (rare — la canonique est le défaut)
.venv/bin/python qa_gate.py image.png --ref autre-reference.png
```

Sorties : un résumé texte par image sur stdout, et le détail complet (toutes les mesures, grappes de couleurs, seuils) en JSON.

**Codes de sortie** (pour l'enchaînement dans des scripts) : `0` = tout PASS, `1` = au moins un À SURVEILLER, `2` = au moins un REJET ou une erreur.

## Les trois contrôles

### 1. Palette — déterministe, ΔE CIEDE2000

Le personnage est isolé (canal alpha si présent, sinon soustraction du fond par la couleur médiane du cadre), aplati sur fond blanc (le fond d'identité de la charte), puis :

- chaque pixel du masque est rattaché à la couleur charte la plus proche (oranges `#FC8B32` / `#C95D1D` / `#8D2E0F`, noir chaud `#1A1512`, blanc — charte §3) ;
- métriques : **ΔE médian** des pixels rattachés aux oranges (distance au ton charte le plus proche) et **couverture conforme** = part des pixels du masque à ΔE ≤ 10 d'un orange charte ;
- le rapport JSON liste aussi les 6 couleurs dominantes du masque (grappes) avec leur ΔE.

**Seuils** (dans `qa_gate.py`) : PASS si ΔE médian ≤ 6,5 **et** couverture ≥ 85 % ; À SURVEILLER si ΔE ≤ 10 et couverture ≥ 65 % ; sinon REJET.

**Calibration — à savoir avant de resserrer les seuils.** Les trois hex de la charte sont des échantillons ponctuels (lumière/médian/ombre) d'un dégradé continu de pelage. Mesuré sur la référence canonique elle-même : ΔE médian 4,83, couverture 96 %. Le plancher intrinsèque de la métrique est donc ≈ 4,8 : les seuils gardent une marge de ×1,35 au-dessus de la référence, et c'est la **couverture** qui fait le gros du rejet (une teinte décalée la fait tomber vers 0 % — voir les tests : +18° de teinte → couverture 7 %, REJET). Si la charte passe un jour à une rampe de tons échantillonnée plus finement, on pourra resserrer.

L'implémentation ΔE2000 a été validée contre la librairie `colour-science` (écart max 0,0 sur 2 000 paires aléatoires ; `colour-science` n'est pas une dépendance de l'outil).

### 2. Proportions — heuristique, à lire honnêtement

Ce qui est **réellement mesuré** : à partir du masque, la hauteur de silhouette H et la plus grande largeur de silhouette dans la zone 2 %–40 % de H (là où se trouve la tête sur une vue de face), d'où un ratio largeur/hauteur comparé aux cadrages canoniques de la charte (§7) :

- buste (cadrage de la référence) : 1,17 T / 1,67 T ≈ **0,701** ;
- corps entier : 1,17 T / 2,51 T ≈ **0,466**.

PASS si l'écart relatif au cadrage le plus proche ≤ 15 %, À SURVEILLER ≤ 40 %, REJET au-delà des deux. Sur la référence elle-même, le ratio mesuré est 0,696 — à 1 % de la théorie.

Ce qui **n'est pas** mesuré, et qu'il ne faut pas prétendre :

- ce n'est **pas** une mesure du vrai ratio « hauteur totale ÷ hauteur de tête (crâne→menton) » : sans repères anatomiques fiables (menton, sommet du crâne hors oreilles), aucun script simple ne le mesure sur une peluche. Le ratio livré est un **indice de silhouette**, pas la mesure T de la charte ;
- il ne vaut que pour une **vue de face ou quasi de face, personnage entier dans le cadre, fond uni ou alpha**. Vue ¾, personnage assis, cadrage coupé, pose dynamique → l'indice peut dévier sans que le personnage soit faux : d'où le verdict plafonné par la zone À SURVEILLER tant qu'on reste dans 40 % ;
- si le masque est non fiable (silhouette touchant les quatre bords, masque < 1 % ou > 98 % de l'image), le contrôle rend **NON MESURABLE** au lieu d'inventer un chiffre — le verdict global tombe alors au mieux à À SURVEILLER.

Le contrôle « dur » des 2,5 T reste donc : cet indice + l'œil humain (et, plus tard, le masque SAM 3 + repères, si le besoin se confirme).

### 3. Similarité — embedding DINOv2 (pré-filtre)

- Modèle : **DINOv2-small** (ViT-S/14) en ONNX, exécuté sur **CPU** via onnxruntime — choisi plutôt que SigLIP/CLIP parce qu'il est purement visuel (pas de tour texte inutile), petit (88 Mo), et ses traits sont une référence standard pour la similarité d'images. Temps mesuré : **0,7–0,85 s par image** sur cette machine ; analyse complète ≈ 2–3 s par image.
- L'image est recadrée sur la boîte du personnage (marge 5 %), prétraitée selon le `preprocessor_config.json` officiel (resize 256, crop central 224, normalisation ImageNet) ; on compare le cosinus du vecteur CLS à celui de la référence (mis en cache dans `models/reference_embedding.json`, invalidé par SHA-256 si la référence change).
- Seuils : PASS ≥ 0,80 ; À SURVEILLER ≥ 0,60 ; REJET en dessous.

**Limite assumée (rapport §5)** : un embedding global est presque insensible à la couleur et aux petits détails d'identité. Preuve par les tests : la référence à teinte décalée de +70° obtient un cosinus de **0,9923** — c'est la palette qui la rejette. L'embedding sert à écarter vite un personnage différent, déformé ou hors sujet ; il ne certifie ni les couleurs, ni les sourcils, ni les coutures.

## Portée et hors-portée

- **Portée** : images du personnage seul — rendus et générations sur fond uni, images avec canal alpha, vues de face ou quasi. C'est le flux de production réel (génération → détourage → cette porte → présentation).
- **Hors portée** : les compositions finies (planches de carrousel, publications avec texte et mise en page, scènes avec décor). Le masque y capture des éléments graphiques et les verdicts n'ont pas de sens — démontré par le test bonus sur la planche 1 du carrousel (voir `tests/RESULTATS.md`). Pour ces livrables, le contrôle reste l'œil humain sur le personnage détouré **avant** composition.
- **Ce que la porte ne fait pas** : le « juge VLM » du rapport §5 (relecture par un modèle de vision d'une liste de contrôle : symétrie des traits, coutures, détails interdits de la charte §1) n'est pas dans cet outil — aucun service externe n'est autorisé ici. Dans le flux réel, cette relecture est faite par l'assistant sur les images qui passent la porte. La porte ne vérifie pas non plus les détails fins (broderies, couture centrale, reflets interdits dans les yeux).

## Tests

```bash
.venv/bin/python tests/make_negative.py   # régénère le négatif +70°
.venv/bin/python qa_gate.py \
  <racine-du-projet>/references/onyx-canonique.webp \
  <racine-du-projet>/layers/reprojection-v1/composite-reprojete-rgba.png \
  tests/negative-teinte-decalee.png --out tests/qa-report-tests.json
```

Résultats réels du 22 novembre 2026 (3 tests obligatoires + 2 bonus) : **`tests/RESULTATS.md`** et `tests/qa-report-tests.json`.

## Fichiers

| Fichier | Rôle |
|---|---|
| `qa_gate.py` | l'outil (un seul script, seuils en constantes en tête) |
| `models/dinov2-small.onnx` | modèle d'embedding (fp32, 88 Mo) |
| `models/dinov2-preprocessor.json` | paramètres de prétraitement officiels |
| `models/reference_embedding.json` | cache de l'embedding de la référence (régénéré si la référence change) |
| `tests/make_negative.py` | génère l'image négative de test (+70° de teinte) |
| `tests/negative-*.png` | images négatives de test (+70°, +18°) |
| `tests/qa-report-tests.json` | rapport JSON des tests |
| `tests/RESULTATS.md` | résultats des tests et leur lecture |
