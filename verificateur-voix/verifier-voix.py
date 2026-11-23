# -*- coding: utf-8 -*-
"""
Vérificateur de voix — contrôle de la fondamentale (F0) des vidéos d'Onyx.

Généralisation de l'outil de mesure du POC B1
(`tests/poc-b1-voix-postsynchro/mesurer_f0.py`) : les fonctions de mesure
ci-dessous (`f0_autocorr`, `analyser`) en sont reprises à l'identique — même
méthode que celle qui a donné ≈ 137 Hz sur la refonte validée
« Demain, on me branche » et ≈ 246 Hz sur la version fautive
(`instagram-branche-2026-11-23/retake-voix-master/README.md`).

Méthode : extraction audio (ffmpeg → WAV 16 kHz mono), autocorrélation par
segments de 0,5 s ; un segment est « voisé » si son énergie dépasse un seuil
relatif ET si le pic d'autocorrélation est net ; la F0 retenue est la médiane
des segments voisés.

Bande de référence de la voix master nº 1 : **123–151 Hz** (référence
137 Hz). Verdict : DANS LA BANDE / HORS BANDE. Règle du projet
(`docs/NORMES-AV-V1.md` § 2.8) : la mesure ne remplace pas l'écoute —
contrôle à l'écoute ET par mesure, avant toute publication.

Usage : python3 verifier-voix.py <fichier-audio-ou-video> [--json] [--segments]
Codes de sortie : 0 = dans la bande ; 1 = hors bande ; 2 = aucune voix
mesurable ou erreur d'entrée.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

FMIN, FMAX = 60.0, 400.0  # plage de recherche de la fondamentale (Hz)
SEG_S = 0.5               # segments de 0,5 s (méthode du retake)

BANDE_MIN, BANDE_MAX = 123.0, 151.0  # bande de la voix master (Hz)
REFERENCE = 137.0                    # F0 de référence de la voix master (Hz)


# --- Mesure : reprise à l'identique de mesurer_f0.py (POC B1) --------------

def f0_autocorr(segment, sr):
    """F0 par autocorrélation normalisée sur un segment ; None si pas de pic net."""
    x = segment - segment.mean()
    if np.sqrt(np.mean(x ** 2)) < 1e-4:
        return None, 0.0
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    if corr[0] <= 0:
        return None, 0.0
    corr = corr / corr[0]
    lag_min = int(sr / FMAX)
    lag_max = int(sr / FMIN)
    zone = corr[lag_min:lag_max]
    if len(zone) == 0:
        return None, 0.0
    pic = int(np.argmax(zone)) + lag_min
    force = float(corr[pic])
    if force < 0.35:  # pic pas assez net -> segment non voisé / bruit
        return None, force
    # interpolation parabolique autour du pic
    if 0 < pic < len(corr) - 1:
        a, b, c = corr[pic - 1], corr[pic], corr[pic + 1]
        denom = a - 2 * b + c
        if denom != 0:
            pic = pic + 0.5 * (a - c) / denom
    return sr / pic, force


def analyser(chemin):
    sr, data = wavfile.read(chemin)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.ndim and data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    else:
        x = data.astype(np.float64)
    n_seg = int(len(x) / sr / SEG_S)
    segments = []
    rms_global = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    seuil_rms = max(0.15 * rms_global, 0.01)
    for i in range(n_seg):
        seg = x[int(i * SEG_S * sr): int((i + 1) * SEG_S * sr)]
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        f0, force = f0_autocorr(seg, sr)
        voise = f0 is not None and rms >= seuil_rms
        segments.append({
            "debut_s": round(i * SEG_S, 2),
            "fin_s": round((i + 1) * SEG_S, 2),
            "rms": round(rms, 4),
            "f0_hz": round(f0, 1) if (f0 and voise) else None,
            "force_pic": round(force, 3),
            "voise": bool(voise),
        })
    f0s = [s["f0_hz"] for s in segments if s["voise"]]
    # régions voisées contiguës (fusion des segments voisés séparés par <= 0,5 s)
    regions = []
    for s in segments:
        if s["voise"]:
            if regions and s["debut_s"] - regions[-1]["fin_s"] <= SEG_S + 1e-9:
                regions[-1]["fin_s"] = s["fin_s"]
            else:
                regions.append({"debut_s": s["debut_s"], "fin_s": s["fin_s"]})
    return {
        "fichier": chemin,
        "sr": sr,
        "duree_s": round(len(x) / sr, 3),
        "nb_segments": n_seg,
        "nb_voises": len(f0s),
        "f0_mediane_hz": round(float(np.median(f0s)), 1) if f0s else None,
        "f0_min_hz": round(min(f0s), 1) if f0s else None,
        "f0_max_hz": round(max(f0s), 1) if f0s else None,
        "regions_voisees": regions,
        "segments": segments,
    }


# --- Extraction audio --------------------------------------------------------

def extraire_audio(chemin, dossier_tmp):
    """Extrait la piste audio en WAV 16 kHz mono via ffmpeg (même préparation
    que les mesures de référence du projet). Retourne le chemin du WAV."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg introuvable dans le PATH")
    wav = os.path.join(dossier_tmp, "audio-16k.wav")
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-i", chemin,
         "-vn", "-ac", "1", "-ar", "16000", wav],
        capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(wav):
        raise RuntimeError(
            "extraction audio impossible (ffmpeg) : " + (proc.stderr.strip()[:300] or "échec inconnu"))
    return wav


# --- Verdict -----------------------------------------------------------------

def verdict(res):
    """Retourne (libellé, code de sortie) à partir du résultat d'analyse."""
    med = res["f0_mediane_hz"]
    if med is None:
        return "AUCUNE VOIX MESURABLE", 2
    if BANDE_MIN <= med <= BANDE_MAX:
        return "DANS LA BANDE", 0
    return "HORS BANDE", 1


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[-3])
        print("Usage : python3 verifier-voix.py <fichier-audio-ou-video> [--json] [--segments]")
        return 2
    chemin = args[0]
    if not os.path.isfile(chemin):
        print(f"ERREUR : fichier introuvable : {chemin}")
        return 2
    try:
        with tempfile.TemporaryDirectory(prefix="verif-voix-") as tmp:
            wav = extraire_audio(chemin, tmp)
            res = analyser(wav)
    except RuntimeError as e:
        print(f"ERREUR : {e}")
        return 2
    libelle, code = verdict(res)
    part = (100.0 * res["nb_voises"] / res["nb_segments"]) if res["nb_segments"] else 0.0
    sortie = {
        "fichier": chemin,
        "duree_s": res["duree_s"],
        "f0_mediane_hz": res["f0_mediane_hz"],
        "f0_min_hz": res["f0_min_hz"],
        "f0_max_hz": res["f0_max_hz"],
        "segments_voises": res["nb_voises"],
        "segments_total": res["nb_segments"],
        "part_voisee_pct": round(part, 1),
        "bande_hz": [BANDE_MIN, BANDE_MAX],
        "reference_hz": REFERENCE,
        "verdict": libelle,
    }
    if "--json" in argv:
        if "--segments" in argv:
            sortie["segments"] = res["segments"]
            sortie["regions_voisees"] = res["regions_voisees"]
        print(json.dumps(sortie, ensure_ascii=False, indent=1))
    else:
        print(f"Fichier          : {chemin}")
        print(f"Durée            : {res['duree_s']:.1f} s")
        if res["f0_mediane_hz"] is not None:
            print(f"F0 médiane       : {res['f0_mediane_hz']:.1f} Hz "
                  f"(référence master : {REFERENCE:.0f} Hz)")
            print(f"Segments voisés  : {res['nb_voises']}/{res['nb_segments']} ({part:.0f} %)")
            print(f"Étendue voisée   : {res['f0_min_hz']:.1f}–{res['f0_max_hz']:.1f} Hz")
        else:
            print("F0 médiane       : — (aucun segment voisé détecté)")
        print(f"Bande master     : {BANDE_MIN:.0f}–{BANDE_MAX:.0f} Hz")
        print(f"VERDICT          : {libelle}")
        if "--segments" in argv:
            for s in res["segments"]:
                f0 = f"{s['f0_hz']:.1f} Hz" if s["f0_hz"] else "—"
                print(f"  {s['debut_s']:>5.1f}–{s['fin_s']:<5.1f} s  {f0:>10}  "
                      f"{'voisé' if s['voise'] else ''}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
