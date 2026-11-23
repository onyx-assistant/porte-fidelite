# Vérificateur de voix — `tools/verificateur-voix/`

Contrôle de la fondamentale (F0) de la voix dans les fichiers audio/vidéo
d'Onyx, avant publication (règle `docs/NORMES-AV-V1.md` § 2.8).

## Usage

```bash
python3 verifier-voix.py <fichier-audio-ou-video>          # verdict lisible
python3 verifier-voix.py <fichier> --json                  # sortie machine
python3 verifier-voix.py <fichier> --segments              # détail par segment
```

Codes de sortie : `0` = DANS LA BANDE ; `1` = HORS BANDE ; `2` = aucune voix
mesurable ou erreur d'entrée.

## Méthode

- Extraction audio par **ffmpeg** (résolu par le `PATH` ; dans l'environnement
  de l'atelier : `/usr/bin/ffmpeg`, version 8.1.2) → WAV 16 kHz mono.
- Mesure **identique** à celle du POC B1
  (`tests/poc-b1-voix-postsynchro/mesurer_f0.py`, fonctions reprises à
  l'identique) : autocorrélation par segments de 0,5 s, segment « voisé » si
  énergie suffisante et pic d'autocorrélation net ; F0 retenue = **médiane
  des segments voisés**.
- **Bande de référence de la voix master nº 1 : 123–151 Hz** (référence
  137 Hz). Verdict : DANS LA BANDE / HORS BANDE, avec part de segments
  voisés et étendue min–max.

## Rappel § 2.8 — la mesure ne suffit pas

Le contrôle obligatoire avant publication est **à l'écoute ET par mesure** :
le générateur re-synthétise la voix, donc le contrôle porte sur le fichier
produit. Un verdict DANS LA BANDE ne dispense jamais de l'écoute ; un verdict
HORS BANDE bloque la publication.

## Test sur vérité connue (22 novembre 2026)

| Fichier | Attendu | Mesuré (médiane) | Segments voisés | Verdict |
|---|---|---|---|---|
| `instagram-branche-2026-11-23/retake-voix-master/reel-demain-on-me-branche-VOIX-MASTER-LIVRABLE.mp4` (livrable validé) | ≈ 137 Hz, dans la bande | **144,2 Hz** (étendue 120,9–211,0 Hz) | 16/20 (80 %) | ✅ **DANS LA BANDE** |
| `instagram-branche-2026-11-23/rates/tentative-2-generateur.mp4` (version fautive publiée puis retirée, voix féminine) | ≈ 246 Hz, hors bande | **232,1 Hz** (étendue 100,8–400,3 Hz) | 19/20 (95 %) | ✅ **HORS BANDE** |
| `instagram-branche-2026-11-23/reel-demain-on-me-branche.mp4` (version livrée de la tentative 2, contrôle additionnel) | hors bande | **233,9 Hz** (étendue 100,9–400,3 Hz) | 18/20 (90 %) | ✅ **HORS BANDE** |

**Écart honnête sur les valeurs absolues.** Les médianes mesurées ici
(144,2 Hz / 232,1 Hz) diffèrent des chiffres consignés à l'époque dans
`retake-voix-master/README.md` (≈ 137 Hz / ≈ 246 Hz, étendue 113–190 Hz) :
ces chiffres-là venaient d'un passage de mesure antérieur. La preuve
d'identité de la méthode a été faite : le script d'origine du POC B1
(`mesurer_f0.py`), appliqué au propre fichier `audio-16k.wav` du retake,
donne **exactement 144,2 Hz** (étendue 120,9–211,7 Hz, 16/20 segments) —
au dixième près la même sortie que le présent outil sur le même fichier,
et le même résultat sur le livrable et sur la brute v1. Les deux classes
restent séparées par ~90 Hz, loin des bords de la bande : les verdicts,
qui sont l'objet de l'outil, sont inchangés.
