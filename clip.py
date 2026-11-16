#!/usr/bin/env python3
"""Clippeur maison : un lien YouTube -> des shorts verticaux sous-titres.

Pipeline : yt-dlp (telechargement) -> Whisper Groq (transcription mot a mot)
-> LLM Groq (choix des meilleurs moments + titre + score) -> ffmpeg
(decoupe 9:16 + sous-titres incrustes).

Usage :
    python clip.py <url_ou_fichier> [--clips 4] [--out sorties]
    python clip.py --selfcheck
"""
import argparse
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
import unicodedata
import urllib.request
import urllib.error

GROQ_BASE = "https://api.groq.com/openai/v1"
WHISPER_MODEL = "whisper-large-v3"
CHAT_MODEL = "openai/gpt-oss-120b"
AUDIO_CHUNK_BYTES = 24 * 1024 * 1024  # marge sous la limite 25 Mo de Groq
# UA generique : sans ca, Cloudflare bloque urllib (403 code 1010)
USER_AGENT = "Mozilla/5.0 (clippeur-maison)"

# format -> (largeur, hauteur, taille des sous-titres)
FORMATS = {"9:16": (1080, 1920, 96), "4:5": (1080, 1350, 84),
           "1:1": (1080, 1080, 76), "16:9": (1920, 1080, 72)}
# type de video -> consigne ajoutee au choix des moments
CATS = {
    "auto": "",
    "chat": "C'est un live de discussion face camera (Just Chatting) : privilegie les "
            "reactions fortes, les punchlines, les histoires racontees et les moments droles.",
    "irl": "C'est un live IRL dans la rue : privilegie les interactions avec des inconnus, "
           "les situations imprevues et les grosses reactions.",
    "gaming": "C'est un live de jeu video : privilegie la rage, les victoires, les moments "
              "droles et les reactions du streamer (tu ne vois que ce qui est dit).",
    "podcast": "C'est un podcast ou une interview : privilegie les idees fortes, les "
               "anecdotes et les avis tranches qui se comprennent seuls.",
}
MAX_RANGE = 3600  # une heure max par passage (quota audio Groq gratuit)
LAYOUTS = ("auto", "full", "split")
ACCENT = "#FF4322"  # orange maison, couleur par defaut du titre
HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
MAX_KEYS = 40
DUR_MIN, DUR_MAX = 40, 90  # bornes du reglage "duree des shorts", en secondes
SPLIT_TOP = 0.4  # ecran partage : part de la hauteur donnee a la facecam
EDGE = 5.0  # marge gardee autour de chaque extrait brut, pour l'allonger a l'edition


# ----------------------------------------------------------------------------- utils

def run(cmd, **kw):
    kw.setdefault("check", True)
    kw.setdefault("stdout", subprocess.DEVNULL)
    kw.setdefault("stderr", subprocess.PIPE)
    try:
        return subprocess.run(cmd, **kw)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", "ignore")[-800:]
        sys.exit(f"[erreur] {cmd[0]} a echoue :\n{err}")


def load_groq_key():
    k = os.environ.get("GROQ_API_KEY")
    if k:
        return k.strip()
    env = pathlib.Path.home() / ".config" / "watch" / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("GROQ_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    sys.exit("[erreur] Pas de cle Groq. Mets GROQ_API_KEY dans l'env ou ~/.config/watch/.env")


def slugify(text, maxlen=40):
    # ASCII seulement : GitHub retire les accents des noms de fichiers publies, et le
    # site ne retrouvait plus "jachète-..." devenu "jachete-..."
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return (s or "short")[:maxlen]


def ass_time(seconds):
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ).stdout.decode().strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def video_dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ).stdout.decode().strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def parse_opts(raw):
    """Reglages venus du site (JSON). Tout ce qui est invalide retombe sur le defaut."""
    try:
        d = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        d = {}
    if not isinstance(d, dict):
        d = {}
    o = {
        "format": d.get("format") if d.get("format") in FORMATS else "9:16",
        "cat": d.get("cat") if d.get("cat") in CATS else "auto",
        "cut": d.get("cut", True) is not False,
        "title": d.get("title", True) is not False,
        "layout": d.get("layout") if d.get("layout") in LAYOUTS else "auto",
        "find": re.sub(r"\s+", " ", str(d.get("find") or ""))[:200].strip(),
        "start": None, "end": None,
        "dmin": DUR_MIN, "dmax": DUR_MAX, "loop": d.get("loop", True) is not False,
        "theme": d.get("theme") if d.get("theme") in THEMES else "none",
    }
    try:
        a, b = int(float(d["dmin"])), int(float(d["dmax"]))
        a, b = max(DUR_MIN, min(a, DUR_MAX)), max(DUR_MIN, min(b, DUR_MAX))
        if b - a >= 5:
            o["dmin"], o["dmax"] = a, b
    except (KeyError, TypeError, ValueError):
        pass
    try:
        a, b = float(d["start"]), float(d["end"])
        if 0 <= a < b <= 86400 and b - a <= MAX_RANGE:
            o["start"], o["end"] = a, b
    except (KeyError, TypeError, ValueError):
        pass
    return o


def keep_segments(words, start, end, gap=0.6, pad=0.15):
    """Morceaux parles d'un extrait : les blancs de plus de `gap` s sont retires."""
    ws = [w for w in words if float(w["start"]) >= start and float(w["end"]) <= end]
    if not ws:
        return [(start, end)]
    segs = []
    a, last = float(ws[0]["start"]), float(ws[0]["end"])
    for w in ws[1:]:
        ws_, we = float(w["start"]), float(w["end"])
        if ws_ - last > gap:
            segs.append((max(start, a - pad), min(end, last + pad)))
            a = ws_
        last = max(last, we)
    segs.append((max(start, a - pad), min(end, last + pad)))
    return [(round(x, 3), round(y, 3)) for x, y in segs if y - x > 0.05]


def remap(t, segs):
    """Temps de la video source -> temps dans le clip une fois les blancs retires."""
    acc = 0.0
    for a, b in segs:
        if t < a:
            return acc
        if t <= b:
            return acc + t - a
        acc += b - a
    return acc


def compute_crop(iw, ih, cx=0.5, cy=0.5, ratio=9 / 16):
    """Fenetre au ratio voulu centree sur (cx, cy) normalises, clampee dans l'image."""
    if iw / ih >= ratio:  # source trop large -> on rogne en largeur
        croph, cropw = ih, min(iw, max(2, round(ih * ratio)))
        x = int(min(max(cx * iw - cropw / 2, 0), iw - cropw))
        y = 0
    else:  # source trop haute -> on rogne en hauteur
        cropw, croph = iw, min(ih, max(2, round(iw / ratio)))
        x = 0
        y = int(min(max(cy * ih - croph / 2, 0), ih - croph))
    cropw -= cropw % 2
    croph -= croph % 2  # dimensions paires pour yuv420
    return cropw, croph, x, y


_FACE_CASCADE = None


def _face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        import cv2
        # cascade embarquee (certaines installs cv2 n'ont pas le fichier data)
        bundled = pathlib.Path(__file__).parent / "assets" / "haarcascade_frontalface_default.xml"
        path = str(bundled) if bundled.exists() else \
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(path)
        if _FACE_CASCADE.empty():
            raise RuntimeError("cascade de visage introuvable")
    return _FACE_CASCADE


YUNET = pathlib.Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"


def _face_finder():
    """Fonction image BGR -> liste de boites (x, y, l, h). YuNet (reseau d'OpenCV, voit
    les petites facecams, les casques, les trois-quarts) ; Haar en secours."""
    import cv2
    if hasattr(cv2, "FaceDetectorYN") and YUNET.exists():
        det = {}

        def yunet(img):
            size = (img.shape[1], img.shape[0])
            if det.get("size") != size:
                det["d"] = cv2.FaceDetectorYN.create(str(YUNET), "", size, 0.6)
                det["size"] = size
            _, faces = det["d"].detect(img)
            return [] if faces is None else [tuple(f[:4]) for f in faces]
        return yunet
    cascade = _face_cascade()
    return lambda img: list(cascade.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(20, 20)))


def face_samples(video, start, end, step=0.5):
    """Visage dominant toutes les `step` s : liste de (t, cx, cy, largeur) normalises,
    (t, None, None, None) quand rien n'est vu. Liste vide si OpenCV est indisponible."""
    try:
        import cv2
        find = _face_finder()
    except Exception as e:  # cv2 absent ou KO -> recadrage centre
        print(f"[info] detection visage indisponible ({e}), recadrage centre.")
        return []
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, round(fps * step))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    out = []
    # lecture sequentielle : beaucoup plus rapide que de repositionner la video a chaque fois
    for i in range(int((end - start) * fps)):
        if i % every:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        w = frame.shape[1]
        img = cv2.resize(frame, None, fx=960 / w, fy=960 / w) if w > 960 else frame
        faces = find(img)
        t = start + i / fps
        if len(faces) == 0:
            out.append((t, None, None, None))
            continue
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        gh, gw = img.shape[:2]
        out.append((t, (fx + fw / 2) / gw, (fy + fh / 2) / gh, fw / gw))
    cap.release()
    return out


def smooth_track(samples, axis=1, dead=0.06, pan=0.4):
    """Trajectoire de camera calme : trous combles, filtre median sur 5 mesures, puis la
    camera ne bouge que si le visage s'eloigne de plus de `dead` (fraction de l'image).
    Renvoie [(t, position)] a interpoler lineairement, ou [] si aucun visage."""
    vals = [smp[axis] for smp in samples]
    if not any(v is not None for v in vals):
        return []
    first = next(v for v in vals if v is not None)
    filled, last = [], first
    for v in vals:
        last = v if v is not None else last
        filled.append(last)
    med = [statistics.median(filled[max(0, i - 2):i + 3]) for i in range(len(filled))]
    pts = [(samples[0][0], med[0])]
    cam = med[0]
    for (t, *_), v in zip(samples, med):
        if abs(v - cam) > dead:
            if t - pan > pts[-1][0]:
                pts.append((t - pan, cam))  # tient la position, puis glisse en `pan` s
            pts.append((t, v))
            cam = v
    return pts


def piecewise_expr(points):
    """[(t, valeur)] -> expression ffmpeg interpolant lineairement en fonction de t."""
    if len(points) == 1:
        return str(points[0][1])
    expr = str(points[-1][1])
    for (t0, v0), (t1, v1) in reversed(list(zip(points, points[1:]))):
        expr = f"if(lt(t,{t1:.3f}),{v0}+({v1 - v0})*(t-{t0:.3f})/{t1 - t0:.3f},{expr})"
    return expr


def plan_layout(samples, opts, iw, ih):
    """'split' (facecam en haut, jeu en bas) ou 'full' (plein cadre qui suit le visage)."""
    faces = [smp for smp in samples if smp[1] is not None]
    vertical = opts["format"] in ("9:16", "4:5") and iw > ih
    if not faces or not vertical or opts["layout"] == "full":
        return "full"
    if opts["layout"] == "split":
        return "split"
    cx = statistics.median(f[1] for f in faces)
    cy = statistics.median(f[2] for f in faces)
    fw = statistics.median(f[3] for f in faces)
    # fixe = presque toutes les mesures pres de la mediane (un faux visage du jeu ne compte pas)
    near = sum(abs(f[1] - cx) < 0.05 and abs(f[2] - cy) < 0.05 for f in faces)
    still = near >= 0.8 * len(faces)
    # facecam typique : petit visage fixe, dans un coin, visible la plupart du temps
    # ponytail: heuristique simple ; un detecteur de cadre de facecam ferait mieux
    if (fw < 0.13 and still and len(faces) >= 0.4 * len(samples) and opts["cat"] != "irl"
            and (abs(cx - 0.5) > 0.2 or abs(cy - 0.5) > 0.2)):
        return "split"
    return "full"


def facecam_crop(samples, iw, ih, ratio):
    """Cadre autour de la facecam (visage median elargi), au ratio du panneau du haut."""
    faces = [smp for smp in samples if smp[1] is not None]
    cx = statistics.median(f[1] for f in faces) * iw
    cy = statistics.median(f[2] for f in faces) * ih
    fwp = statistics.median(f[3] for f in faces) * iw
    bw = min(iw, fwp * 5)  # la boite YuNet serre le visage : x5 = tete et epaules
    bh = bw / ratio
    if bh > ih:
        bh, bw = ih, ih * ratio
    bw, bh = int(bw) // 2 * 2, int(bh) // 2 * 2
    x = int(min(max(cx - bw / 2, 0), iw - bw))
    y = int(min(max(cy + 0.25 * fwp - bh / 2, 0), ih - bh))
    return bw, bh, x, y


# ----------------------------------------------------------------------------- Groq HTTP

def groq_post_multipart(path_endpoint, fields, file_field, filename, data, key):
    boundary = "----clip" + os.urandom(12).hex()
    body = bytearray()
    for name, value in fields:  # liste de tuples -> cles repetables
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
             f"filename=\"{filename}\"\r\nContent-Type: audio/mpeg\r\n\r\n").encode()
    body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        GROQ_BASE + path_endpoint, data=bytes(body), method="POST",
        headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def groq_chat(messages, key, json_mode=True):
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.4}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        GROQ_BASE + "/chat/completions", data=json.dumps(payload).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT,
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ----------------------------------------------------------------------------- etapes

def download(url, workdir):
    out = workdir / "source.mp4"
    print("[1/4] telechargement...")
    cmd = ["yt-dlp", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4",
           "--no-playlist", "-o", str(out), url]
    # sur un serveur, YouTube exige des cookies d'un compte connecte
    cookies = os.environ.get("YT_COOKIES_FILE")
    if cookies and pathlib.Path(cookies).exists():
        cmd[1:1] = ["--cookies", cookies]
    run(cmd)
    if not out.exists():  # yt-dlp peut suffixer l'extension
        cands = list(workdir.glob("source.*"))
        if cands:
            return cands[0]
        sys.exit("[erreur] telechargement : aucun fichier produit")
    return out


def extract_audio(video, workdir):
    audio = workdir / "audio.m4a"
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "64k", str(audio)])
    return audio


def _transcribe_one(data, key):
    fields = [("model", WHISPER_MODEL), ("response_format", "verbose_json"),
              ("timestamp_granularities[]", "word")]
    return groq_post_multipart("/audio/transcriptions", fields, "file", "audio.m4a", data, key)


def transcribe(video, workdir, key):
    print("[2/4] transcription (Whisper Groq)...")
    audio = extract_audio(video, workdir)
    size = audio.stat().st_size
    words = []
    if size <= AUDIO_CHUNK_BYTES:
        resp = _transcribe_one(audio.read_bytes(), key)
        words = resp.get("words") or []
    else:
        dur = ffprobe_duration(audio)
        nchunks = math.ceil(size / AUDIO_CHUNK_BYTES)
        step = dur / nchunks
        for i in range(nchunks):
            start = i * step
            part = workdir / f"audio_{i}.m4a"
            run(["ffmpeg", "-y", "-ss", str(start), "-t", str(step), "-i", str(audio),
                 "-ac", "1", "-ar", "16000", "-b:a", "64k", str(part)])
            resp = _transcribe_one(part.read_bytes(), key)
            for w in resp.get("words") or []:
                w["start"] = float(w.get("start", 0)) + start
                w["end"] = float(w.get("end", 0)) + start
                words.append(w)
    words = [w for w in words if w.get("word") and w.get("end") is not None]
    if not words:
        sys.exit("[erreur] transcription vide (pas de paroles detectees ?)")
    return words


def transcript_lines(words, group=10):
    """Transcript compact et horodate pour le LLM."""
    lines = []
    for i in range(0, len(words), group):
        chunk = words[i:i + group]
        t = int(float(chunk[0]["start"]))
        txt = " ".join(w["word"].strip() for w in chunk)
        lines.append(f"[{t//60:02d}:{t%60:02d}] ({t}s) {txt}")
    return "\n".join(lines)


# Offre gratuite Groq : 8000 jetons par minute, question et reponse comprises. Une demi-heure
# de parole depasse ca (erreur 413) : on envoie la transcription par tranches.
CHUNK_CHARS = 11000  # ~3500 jetons de transcription par appel
# Choix de Jacques (2026-11-15) : tout ce qui est publie est en anglais, quelle que soit la
# langue de la video. Consigne en anglais : en francais, le modele repondait parfois en francais.
ENGLISH = ("IMPORTANT: write every title, caption and hashtag in ENGLISH, even if the video "
           "is in French or any other language. Never answer in French.")


def chunk_lines(text, size=CHUNK_CHARS):
    chunks, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) > size:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    return chunks + [cur] if cur else chunks


def chat_retry(messages, key, tries=4):
    """groq_chat qui attend la fin de la minute quand le quota par minute est atteint."""
    for i in range(tries):
        try:
            return groq_chat(messages, key)
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            wait = min(float(e.headers.get("retry-after") or 30), 90) + 1
            print(f"[info] quota Groq par minute atteint, pause de {wait:.0f} s")
            time.sleep(wait)


def best_distinct(clips, n):
    """Meilleurs extraits sans chevauchement."""
    out = []
    for c in sorted(clips, key=lambda c: c["score"], reverse=True):
        if all(c["end"] <= o["start"] or c["start"] >= o["end"] for o in out):
            out.append(c)
    return out[:n]


def kept(words, a, b, cut):
    """Duree finale d'un extrait, blancs retires si demande."""
    return sum(y - x for x, y in keep_segments(words, a, b)) if cut else b - a


def norm_word(w):
    return re.sub(r"[^\w']", "", str(w).lower())


def fit_clip(clip, words, opts):
    """Cale un extrait sur les mots : debut sur un mot, fin sur la fin d'un mot, duree
    finale dans [dmin, dmax]. En boucle, la fin prefere le moment ou la phrase continue
    (pas de blanc) et, mieux, celui ou le mot suivant est le premier mot du short :
    la derniere image s'enchaine sur la premiere."""
    dmin, dmax, cut = opts["dmin"], opts["dmax"], opts["cut"]
    ws = [w for w in words if float(w["end"]) > clip["start"] - .3]
    if not ws:
        return clip
    i0 = 0
    start = max(0.0, float(ws[i0]["start"]) - .05)
    first = norm_word(ws[i0]["word"])
    ws = words = [w for w in ws if float(w["start"]) < start + dmax + 5]  # fenetre utile seulement
    best, best_sc = None, None
    for i, w in enumerate(ws[i0:], i0):
        e = float(w["end"])
        d = kept(words, start, e + .08, cut)
        if d > dmax:
            break
        if d < dmin:
            continue
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        gap = float(nxt["start"]) - e if nxt else 9
        sc = -abs(e - clip["end"]) / 8  # reste pres de la fin choisie par l'IA
        if opts["loop"]:
            sc += 1.0 if gap < .25 else -1.0 if gap > .6 else 0
            if nxt and norm_word(nxt["word"]) == first and first:
                sc += 3
        else:
            sc += 1.0 if gap > .5 else 0
        if best_sc is None or sc > best_sc:
            best, best_sc = (e, nxt), sc
    if best is None:  # pas assez de parole : on garde ce qui rentre, au plus pres de dmax
        last = [w for w in ws if kept(words, start, float(w["end"]) + .08, cut) <= dmax]
        best = (float((last or ws)[-1]["end"]), None)
    e, nxt = best
    end = e + .08 if not nxt else min(e + .08, float(nxt["start"]))
    return {**clip, "start": round(start, 2), "end": round(end, 2)}


def select_clips(words, key, n, cat="auto", find="", opts=None):
    opts = opts or parse_opts("{}")
    print("[3/4] choix des meilleurs moments (LLM Groq)...")
    total = int(float(words[-1]["end"]))
    chunks = chunk_lines(transcript_lines(words))
    found = []
    for i, part in enumerate(chunks):
        if len(chunks) > 1:
            print(f"    tranche {i + 1}/{len(chunks)}")
        try:
            found += select_in(part, total, key, n, cat, find, len(chunks) > 1, opts)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError) as e:
            print(f"[info] LLM indisponible sur cette tranche ({e})")
    if not found:
        print("[info] aucun extrait propose par le LLM, decoupage regulier en secours.")
        found = fallback_clips(total, n, opts["dmax"])
    return best_distinct([fit_clip(c, words, opts) for c in found], n)


def select_in(transcript, total, key, n, cat, find, partial, opts):
    sys_msg = ("Tu es un monteur expert en shorts viraux (TikTok, Reels, YouTube Shorts). "
               "A partir d'une transcription horodatee, tu reperes les meilleurs extraits "
               "autonomes a transformer en shorts verticaux.")
    what = (f"un passage de la transcription d'une video de {total}s" if partial
            else f"la transcription d'une video de {total}s")
    user_msg = (
        f"Voici {what}. Chaque ligne commence par son "
        f"temps de debut en secondes.\n\n{transcript}\n\n"
        f"Choisis les {n} meilleurs extraits. Reponds STRICTEMENT en JSON :\n"
        '{\"clips\":[{\"start\":<debut_en_secondes>,\"end\":<fin_en_secondes>,'
        '\"title\":\"<titre accrocheur, max 60 caracteres>\",'
        '\"score\":<viralite 0-100>,\"reason\":\"<pourquoi ca marche, 1 phrase>\",'
        '\"caption\":\"<legende de post, 1-2 phrases accrocheuses>\",'
        '\"hashtags\":[\"<4 a 6 hashtags pertinents>\"]}]}\n'
        f"Regles : chaque extrait dure entre {opts['dmin']} et {opts['dmax']} secondes, start et end sont des "
        "nombres en secondes pris dans la video, les extraits ne se chevauchent pas, "
        "tries du meilleur score au moins bon. Sans aucun emoji. Rien d'autre que le JSON.\n"
        + ENGLISH)
    if opts["loop"]:
        user_msg += ("\n\nLe short sera lu en boucle : choisis une fin qui s'enchaine "
                     "naturellement avec la premiere phrase de l'extrait, en plein elan, "
                     "sans conclusion ni au revoir, pour que le spectateur ne voie pas la coupure.")
    if CATS.get(cat):
        user_msg += "\n\nContexte : " + CATS[cat]
    if find:
        user_msg += ("\n\nLe spectateur cherche un moment precis : \u00ab " + find + " \u00bb. "
                     "Ne retiens que des extraits qui y correspondent, du plus probable au "
                     "moins probable (le score dit a quel point ca correspond). S'il n'y a "
                     "pas de correspondance exacte, donne les passages les plus proches.\n"
                     + ENGLISH)
    raw = chat_retry([{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}], key)
    clips = parse_clips(raw, total)
    if not clips:
        print("[info] reponse LLM inexploitable sur cette tranche")
    return clips


def parse_clips(raw, total):
    """Extrait et valide la liste de clips depuis la reponse LLM."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    items = data.get("clips") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out = []
    for c in items:
        try:
            start = max(0.0, float(c["start"]))
            end = min(float(total), float(c["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not (2 <= end - start <= 120):  # garde-fou duree
            continue
        tags = c.get("hashtags")
        if isinstance(tags, str):
            tags = tags.split()
        if isinstance(tags, list):  # "#" garanti, espaces internes retires : pret a publier
            tags = " ".join("#" + re.sub(r"[\s#]+", "", str(t)) for t in tags if str(t).strip("# "))
        out.append({
            "start": start, "end": end,
            "title": str(c.get("title", "Short"))[:80],
            "score": int(c.get("score", 0) or 0),
            "reason": str(c.get("reason", ""))[:200],
            "caption": str(c.get("caption", ""))[:400],
            "hashtags": str(tags or "")[:300],
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def fallback_clips(total, n, dur=30.0):
    """Secours : n fenetres reparties si le LLM echoue."""
    dur = min(float(dur), max(15.0, total / max(n, 1)))
    base = {"score": 50, "caption": "", "hashtags": ""}
    if total <= dur:
        return [{"start": 0, "end": total, "title": "Short 1",
                 "reason": "video entiere", **base}]
    step = (total - dur) / max(n - 1, 1)
    clips = []
    for i in range(n):
        s = round(i * step, 1)
        clips.append({"start": s, "end": round(s + dur, 1), "title": f"Short {i+1}",
                      "reason": "decoupage regulier", **base})
    return clips


def ass_text(s):
    """Neutralise les codes ASS ({...}, \\N) dans un texte."""
    return (s.replace("\\", "/").replace("{", "(").replace("}", ")")
            .replace("\n", " ").strip())


def ass_color(hexc):
    """#RRGGBB -> &H00BBGGRR (ordre ASS)."""
    return f"&H00{hexc[5:7]}{hexc[3:5]}{hexc[1:3]}".upper()


def ink_on(hexc):
    """Texte noir sur fond clair, blanc sur fond fonce."""
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return "#0A0A0B" if .299 * r + .587 * g + .114 * b > 140 else "#FFFFFF"


def build_ass(words, segs, out_path, fmt="9:16", title="", group=3, margin_v=None,
              subs=True, color=ACCENT, subc="#FFFFFF", font="Arial Black", accent=None):
    """Sous-titres ASS : groupes de mots gros et centres, titre en haut si demande.
    `segs` = morceaux gardes de la source ; les temps sont recales dessus.
    `accent` : le mot le plus long de chaque groupe prend cette couleur (l'oeil s'y accroche)."""
    W, H, fs = FORMATS[fmt]
    big = font == "Anton"  # police etroite : plus grande a lisibilite egale
    header = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
        f"Style: Pop,{font},{round(fs * 1.12) if big else fs},{ass_color(subc)},&H00000000,&H00000000,"
        f"{0 if big else -1},1,{7 if big else 6},2,2,80,80,{round(H * .135) if margin_v is None else margin_v}\n"
        # titre : texte sur bandeau de couleur (BorderStyle 3 = boite opaque)
        f"Style: Title,Arial Black,{round(fs * .62)},{ass_color(ink_on(color))},{ass_color(color)},&H00000000,"
        f"-1,3,14,0,8,90,90,{round(H * .07)}\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lo, hi = segs[0][0], segs[-1][1]
    total = sum(b - a for a, b in segs)
    win = [w for w in words if float(w["end"]) > lo and float(w["start"]) < hi] if subs else []
    events = []
    if title:
        events.append(f"Dialogue: 1,{ass_time(0)},{ass_time(total)},Title,,0,0,0,,{ass_text(title)}")
    for i in range(0, len(win), group):
        chunk = win[i:i + group]
        start = remap(float(chunk[0]["start"]), segs)
        end = max(start + 0.3, remap(float(chunk[-1]["end"]), segs))
        ws = [ass_text(w["word"].strip().upper()) for w in chunk]
        if accent:
            i = max(range(len(ws)), key=lambda j: len(re.sub(r"\W", "", ws[j])))
            if len(re.sub(r"\W", "", ws[i])) >= 4:
                ws[i] = f"{{\\c{ass_color(accent)}}}{ws[i]}{{\\c{ass_color(subc)}}}"
        text = " ".join(ws)
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(min(end, total))},Pop,,0,0,0,,{text}")
    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_path


def track_crop(samples, segs, iw, ih, cw, ch):
    """Position du cadre dans le temps (temps du clip), en pixels : (x, y) en expressions."""
    if cw < iw:  # cadre plus etroit que la source : il glisse a l'horizontale
        axis, span, size = 1, iw, cw
    else:
        axis, span, size = 2, ih, ch
    pts = []
    for t, v in smooth_track(samples, axis):
        tt = remap(t, segs)
        px = int(min(max(v * span - size / 2, 0), span - size))
        if pts and tt <= pts[-1][0] + 1e-3:
            pts[-1] = (pts[-1][0], px)
        else:
            pts.append((tt, px))
    if not pts:
        pts = [(0.0, (span - size) // 2)]
    expr = piecewise_expr(pts)
    other = str((ih - ch) // 2 if axis == 1 else (iw - cw) // 2)
    return (expr, other) if axis == 1 else (other, expr)


def split_heights(H):
    ht = int(H * SPLIT_TOP) // 2 * 2
    return ht, H - ht


def norm_rect(px, iw, ih):
    w, h, x, y = px
    return {"x": round(x / iw, 4), "y": round(y / ih, 4),
            "w": round(w / iw, 4), "h": round(h / ih, 4)}


def rect_px(r, iw, ih, ratio):
    """Rectangle normalise (editeur) -> cadre en pixels au ratio exact, garde dans l'image."""
    cx, cy = (r["x"] + r["w"] / 2) * iw, (r["y"] + r["h"] / 2) * ih
    w = max(32.0, r["w"] * iw)
    h = w / ratio
    if h > ih:
        h, w = ih, ih * ratio
    if w > iw:
        w, h = iw, iw / ratio
    w, h = int(w) // 2 * 2, int(h) // 2 * 2
    x = int(min(max(cx - w / 2, 0), iw - w))
    y = int(min(max(cy - h / 2, 0), ih - h))
    return w, h, x, y


def parse_spec(d, dur):
    """Reglages d'un short tels que l'editeur les envoie. Invalide -> valeur sure."""
    d = d if isinstance(d, dict) else {}

    def rect(r):
        try:
            r = {k: float(r[k]) for k in "xywh"}
        except (KeyError, TypeError, ValueError):
            return None
        ok = all(0 <= r[k] <= 1 for k in "xy") and all(0.02 <= r[k] <= 1 for k in "wh")
        return r if ok else None
    try:
        a, b = float(d["trim"][0]), float(d["trim"][1])
        if not (0 <= a < b <= dur + 0.05 and b - a >= 1):
            raise ValueError
        b = min(b, dur)
    except (KeyError, TypeError, ValueError, IndexError):
        a, b = 0.0, dur
    return {
        "format": d.get("format") if d.get("format") in FORMATS else "9:16",
        "layout": "split" if d.get("layout") == "split" else "full",
        "crop": rect(d.get("crop")), "cam": rect(d.get("cam")), "game": rect(d.get("game")),
        "track": d.get("track") is True,
        "trim": [round(a, 2), round(b, 2)],
        "cut": d.get("cut", True) is not False,
        "subs": d.get("subs", True) is not False,
        "title": ass_text(str(d.get("title") or ""))[:100],
        "ask": re.sub(r"\s+", " ", str(d.get("ask") or ""))[:300].strip(),
        "color": d["color"].upper() if HEX_RE.match(str(d.get("color") or "")) else ACCENT,
        "subc": d["subc"].upper() if HEX_RE.match(str(d.get("subc") or "")) else "#FFFFFF",
        "keys": parse_keys(d.get("keys"), dur, rect),
        "theme": d.get("theme") if d.get("theme") in THEMES else "none",
        **{k: ass_text(str(d.get(k) or ""))[:n] for k, n in TEXTS.items()},
    }


TEXTS = {"kicker": 40, "lab1": 20, "lab2": 20, "credit": 60, "ticker": 200}  # textes du decor


def parse_keys(keys, dur, rect):
    """Cadrages valides a la main a un moment : chacun vaut jusqu'au suivant."""
    out = {}
    for k in keys if isinstance(keys, list) else []:
        try:
            t = round(float(k["t"]), 2)
        except (KeyError, TypeError, ValueError):
            continue
        r = {n: rect(k.get(n)) for n in ("crop", "cam", "game")}
        if 0 <= t <= dur and any(r.values()):
            out[t] = {"t": t, **{n: v for n, v in r.items() if v}}
    return [out[t] for t in sorted(out)][:MAX_KEYS]


def plans(spec):
    """Decoupe le short en plans [(debut, fin, cadres)] en temps du short (0 = debut de coupe)."""
    s0, s1 = spec["trim"]
    cur = {n: spec[n] for n in ("crop", "cam", "game")}
    out = [[s0, cur]]
    for k in spec.get("keys") or []:
        r = {**out[-1][1], **{n: k[n] for n in ("crop", "cam", "game") if k.get(n)}}
        if k["t"] <= s0 + .05:
            out[-1][1] = r
        elif k["t"] < s1 - .05:
            out.append([k["t"], r])
    return [(round(a - s0, 3), round((out[i + 1][0] if i + 1 < len(out) else s1) - s0, 3), r)
            for i, (a, r) in enumerate(out)]


def auto_spec(samples, opts, iw, ih, s0, s1, title, tags=""):
    """Reglages de depart, habillage compris (theme choisi a la generation)."""
    sp = base_spec(samples, opts, iw, ih, s0, s1, title)
    sp.update(theme=opts.get("theme", "none"), kicker=gta_kicker(), lab1="", lab2="",
              credit="", ticker=" ".join(tags.split()[:8]) or title)
    return sp


def base_spec(samples, opts, iw, ih, s0, s1, title):
    """Reglages de depart d'un short : ce que le moteur choisit seul, modifiable ensuite."""
    W, H, _ = FORMATS[opts["format"]]
    ht, hb = split_heights(H)
    faces = [smp for smp in samples if smp[1] is not None]
    cx = statistics.median(f[1] for f in faces) if faces else 0.5
    cy = statistics.median(f[2] for f in faces) if faces else 0.5
    cam = facecam_crop(samples, iw, ih, W / ht) if faces else \
        compute_crop(iw, ih, 0.12, 0.15, W / ht)
    return {
        "format": opts["format"], "layout": plan_layout(samples, opts, iw, ih),
        "crop": norm_rect(compute_crop(iw, ih, cx, cy, W / H), iw, ih),
        "cam": norm_rect(cam, iw, ih),
        "game": norm_rect(compute_crop(iw, ih, 0.5, 0.5, W / hb), iw, ih),
        "track": bool(faces), "trim": [round(s0, 2), round(s1, 2)],
        "cut": opts["cut"], "subs": True, "title": title, "ask": "",
        "color": ACCENT, "subc": "#FFFFFF", "keys": [],
    }


# ----------------------------------------------------------------------------- themes GTA
# Habillages 9:16 inspires de l'univers GTA (couleurs de Vice City, etoiles de recherche,
# ecrans de chargement). Aucun logo ni visuel de Rockstar : seulement l'esprit.
# Fenetres = (x, y, largeur, hauteur, angle en degres) en pixels 1080x1920. "full" = une
# fenetre (cadre unique), "split" = deux (facecam/plan 1 puis jeu/plan 2).
# Le site garde une copie de ces fenetres (THEMES dans index.html) : les deux doivent suivre.
FONTS = pathlib.Path(__file__).parent / "assets" / "fonts"
PINK, ORANGE, CYAN, GOLD, PLUM = "#FF2E88", "#FF8A3D", "#2EE6F0", "#FFC83D", "#16071F"
THEMES = {
    "neon": {"full": [(60, 600, 960, 720, 0)], "split": [(60, 540, 960, 400, 0), (60, 960, 960, 560, 0)],
             "sub": {"full": 1560, "split": 1700}, "accent": PINK},
    "wanted": {"full": [(0, 0, 1080, 1920, 0)], "split": [(0, 0, 1080, 768, 0), (0, 768, 1080, 1152, 0)],
               "sub": {"full": 1600, "split": 790}, "accent": GOLD},
    "loading": {"full": [(80, 560, 920, 1150, -2.5)],
                "split": [(110, 420, 860, 520, -3), (60, 1010, 960, 680, 2)],
                "sub": {"full": 1815, "split": 1815}, "accent": GOLD},
    "versus": {"full": [(0, 330, 1080, 760, 0), (0, 1110, 1080, 760, 0)],
               "split": [(0, 330, 1080, 760, 0), (0, 1110, 1080, 760, 0)],
               "sub": {"full": 1780, "split": 1780}, "accent": GOLD},
    "news": {"full": [(0, 420, 1080, 608, 0)], "split": [(0, 150, 1080, 432, 0), (0, 582, 1080, 608, 0)],
             "sub": {"full": 1640, "split": 1730}, "accent": "#FFD23F"},
}
GTA6_OUT = (2026, 11, 19)


def gta_kicker(today=None):
    """Pastille de serie : compte a rebours avant la sortie, puis jour N. L'envie de ne
    pas rater l'episode du jour fait revenir les memes spectateurs."""
    import datetime
    d = (datetime.date(*GTA6_OUT) - (today or datetime.date.today())).days
    if d > 1:
        return f"GTA 6 · {d} DAYS LEFT"
    if d == 1:
        return "GTA 6 · OUT TOMORROW"
    if d == 0:
        return "GTA 6 · OUT TODAY"
    return f"GTA 6 · DAY {1 - d}"


def theme_windows(spec):
    """Fenetres du theme pour ce short, ou None si pas de theme (ou format autre que 9:16)."""
    th = THEMES.get(spec.get("theme"))
    if not th or spec["format"] != "9:16":
        return None
    two = spec["theme"] == "versus" or (spec["layout"] == "split" and spec["cam"] and spec["game"])
    return th["split" if two else "full"]


def _font(name, size):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONTS / name), size)


def _rgb(hexc, a=255):
    return tuple(int(hexc[i:i + 2], 16) for i in (1, 3, 5)) + (a,)


def _wrap(text, font, maxw, maxlines):
    lines, cur = [], ""
    for w in text.upper().split():
        t = f"{cur} {w}".strip()
        if cur and font.getlength(t) > maxw:
            lines.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines if len(lines) <= maxlines else None


def _fit(text, name, maxw, maxlines, size, smallest):
    """Plus grosse taille ou le texte tient en `maxlines` lignes."""
    while size > smallest:
        f = _font(name, size)
        lines = _wrap(text, f, maxw, maxlines)
        if lines:
            return f, lines
        size -= 6
    f = _font(name, smallest)
    return f, (_wrap(text, f, maxw, 99) or [""])[:maxlines]


def _grad(w, h, c1, c2, diag=True):
    """Degrade c1 -> c2, en diagonale (haut-gauche -> bas-droite) ou horizontal."""
    from PIL import Image
    a, b = _rgb(c1), _rgb(c2)
    ramp = Image.linear_gradient("L").rotate(90).resize((w, h))  # 0 a gauche -> 255 a droite
    if diag:
        ramp = Image.linear_gradient("L").rotate(45, expand=True).resize((w, h))
        ramp = ramp.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(180)
    return Image.composite(Image.new("RGBA", (w, h), b), Image.new("RGBA", (w, h), a), ramp)


def _rot_rect(x, y, w, h, deg, pad=0):
    """Coins d'un rectangle tourne autour de son centre (sens horaire, comme ffmpeg rotate)."""
    cx, cy, r = x + w / 2, y + h / 2, math.radians(deg)
    pts = []
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        px, py = dx * (w / 2 + pad), dy * (h / 2 + pad)
        pts.append((cx + px * math.cos(r) - py * math.sin(r), cy + px * math.sin(r) + py * math.cos(r)))
    return pts


def _star(cx, cy, ro, ri=None):
    ri = ri or ro * .45
    return [(cx + (ro if i % 2 == 0 else ri) * math.sin(i * math.pi / 5),
             cy - (ro if i % 2 == 0 else ri) * math.cos(i * math.pi / 5)) for i in range(10)]


def _text(img, xy, txt, font, fill, anchor="mm", stroke=0, stroke_fill=(0, 0, 0, 255), shadow=0):
    from PIL import ImageDraw, ImageFilter, Image
    if shadow:  # ombre douce : lisible sur n'importe quel plan du jeu
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((xy[0], xy[1] + shadow // 2), txt, font=font, fill=(0, 0, 0, 170),
                                anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0, 170))
        img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(shadow)))
    ImageDraw.Draw(img).text(xy, txt, font=font, fill=fill, anchor=anchor,
                             stroke_width=stroke, stroke_fill=stroke_fill)


def _grad_text(img, xy, txt, font, c1, c2, anchor="mm", shadow=10):
    """Texte rempli d'un degrade (dernier mot du titre : la ou l'oeil tombe)."""
    from PIL import Image, ImageDraw
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).text(xy, txt, font=font, fill=255, anchor=anchor)
    box = m.getbbox()
    if not box:
        return
    _text(img, xy, txt, font, (0, 0, 0, 0), anchor, shadow=shadow)
    g = Image.new("RGBA", img.size, (0, 0, 0, 0))
    g.paste(_grad(box[2] - box[0], box[3] - box[1], c1, c2, diag=False), box[:2])
    img.paste(g, (0, 0), m)


def _pill(img, cx, cy, txt, font, bg, fg, padx=28, h=None, outline=None):
    from PIL import ImageDraw
    w = font.getlength(txt) + 2 * padx
    h = h or font.size * 1.5
    ImageDraw.Draw(img).rounded_rectangle((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), h / 2,
                                          fill=bg, outline=outline, width=4 if outline else 0)
    _text(img, (cx, cy + font.size * .04), txt, font, fg)
    return w


def theme_layers(spec, wins, dur, workdir):
    """Dessine le decor : fond (sous la video), avant-plan (par-dessus) et elements animes.
    Renvoie (fond.png, avant.png, [(png, filtre, x, y, enable)])."""
    from PIL import Image, ImageDraw, ImageFilter
    W, H = 1080, 1920
    name, two = spec["theme"], len(wins) == 2
    bg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d_bg, d_fg = ImageDraw.Draw(bg), ImageDraw.Draw(fg)
    title, kicker = spec.get("title") or "", spec.get("kicker") or ""
    extras = []
    stem = workdir / f"_{name}_{os.getpid()}"

    def save(img, tag):
        p = pathlib.Path(f"{stem}_{tag}.png")
        img.save(p)
        return p

    def progress(x, y, w, h, c1, c2):
        """Barre qui se remplit avec la video : l'oeil veut la voir arriver au bout."""
        d_fg.rounded_rectangle((x, y, x + w, y + h), h / 2, fill=(255, 255, 255, 46))
        bar = Image.new("RGBA", (2 * w, h), (0, 0, 0, 0))
        g = _grad(w, h, c1, c2, diag=False)
        mk = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mk).rounded_rectangle((0, 0, w - 1, h - 1), h / 2, fill=255)
        bar.paste(g, (0, 0), mk)
        extras.append((save(bar, "bar"), f"crop=w={w}:h={h}:x='{w}-{w}*t/{dur:.3f}':y=0", x, y, ""))

    if name == "neon":
        d_bg.rectangle((0, 0, W, H), fill=_rgb(PLUM, 175))
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for (x, y, w, h, _), (c1, c2) in zip(wins, ((PINK, CYAN), (CYAN, PINK))):
            frame = _grad(w + 40, h + 40, c1, c2)
            m = Image.new("L", (w + 40, h + 40), 0)
            ImageDraw.Draw(m).rectangle((14, 14, w + 26, h + 26), outline=255, width=12)
            glow.paste(frame, (x - 20, y - 20), m)
            m2 = Image.new("L", (w + 40, h + 40), 0)
            ImageDraw.Draw(m2).rectangle((15, 15, w + 25, h + 25), outline=255, width=6)
            fg.paste(frame, (x - 20, y - 20), m2)
        bg.alpha_composite(glow.filter(ImageFilter.GaussianBlur(22)))
        bg.alpha_composite(glow.filter(ImageFilter.GaussianBlur(6)))
        top = wins[0][1]
        if kicker:
            _pill(fg, W / 2, 150, kicker, _font("Anton-Regular.ttf", 44), _rgb(PINK), (255, 255, 255, 255),
                  h=76)
        if title:
            f, lines = _fit(title, "Anton-Regular.ttf", 920, 2, 112, 70)
            lh = f.size * 1.08
            y0 = (205 + top - 30) / 2 - lh * (len(lines) - 1) / 2
            for i, ln in enumerate(lines):
                if i == len(lines) - 1:
                    _grad_text(fg, (W / 2, y0 + i * lh), ln, f, PINK, ORANGE)
                else:
                    _text(fg, (W / 2, y0 + i * lh), ln, f, (255, 255, 255, 255), shadow=10)
        x, y, w, h, _ = wins[-1]
        progress(x, y + h + 26, w, 10, PINK, CYAN)
        if spec.get("credit"):
            _text(fg, (W / 2, y + h + 70), spec["credit"].upper(), _font("BarlowCondensed-Bold.ttf", 32),
                  (255, 255, 255, 150))

    elif name == "wanted":
        shade = Image.new("L", (1, H))  # haut et bas assombris : le texte reste lisible
        for yy in range(H):
            a = max(0.0, 1 - yy / 520) * 190 + max(0.0, (yy - 1380) / 540) * 200
            shade.putpixel((0, yy), int(min(a, 210)))
        fg.paste(Image.new("RGBA", (W, H), (0, 0, 0, 255)), (0, 0), shade.resize((W, H)))
        stars = [(W - 92 - i * 84, 96) for i in range(5)][::-1]
        for cx, cy in stars:
            d_fg.polygon(_star(cx, cy, 36), fill=(0, 0, 0, 120), outline=(255, 255, 255, 230), width=4)
        if kicker:
            _text(fg, (60, 96), kicker, _font("Anton-Regular.ttf", 46), (255, 255, 255, 255), anchor="lm",
                  stroke=5)
        if title:
            f, lines = _fit(title, "Anton-Regular.ttf", 960, 3, 124, 76)
            lh = f.size * 1.1
            for i, ln in enumerate(lines):
                last = i == len(lines) - 1
                _text(fg, (W / 2, 250 + i * lh), ln, f, _rgb(GOLD) if last else (255, 255, 255, 255),
                      stroke=9, shadow=12)
        # etoiles qui s'allument au fil du clip : on reste pour voir la 5e
        for i, (cx, cy) in enumerate(stars):
            s = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
            sd = ImageDraw.Draw(s)
            sd.polygon(_star(48, 48, 40), fill=(0, 0, 0, 255))
            sd.polygon(_star(48, 48, 34), fill=_rgb(GOLD))
            t0 = dur * (i + 1) / 6.5
            extras.append((save(s, f"star{i}"), "", cx - 48, cy - 48, f"gte(t,{t0:.2f})"))
        # recompense de fin : boucle bouclee, envie de relancer
        ban = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
        ImageDraw.Draw(ban).rectangle((0, 90, W, 330), fill=(0, 0, 0, 165))
        _text(ban, (W / 2, 185), "MISSION PASSED", _font("Anton-Regular.ttf", 150), _rgb(GOLD), stroke=8,
              shadow=10)
        _text(ban, (W / 2, 290), "RESPECT +", _font("Anton-Regular.ttf", 58), (255, 255, 255, 255), stroke=5)
        extras.append((save(ban, "end"), f"fade=in:st={max(0, dur - 2.2):.2f}:d=0.25:alpha=1",
                       0, 750, f"gte(t,{max(0, dur - 2.2):.2f})"))

    elif name == "loading":
        # affiche peinte : bandes de coucher de soleil en biais, trame BD sur les jointures
        bands = [(0, "#FF4F9A"), (.33, "#FF8A3D"), (.62, "#FFB547"), (.84, "#1EB7C2")]
        for i, (p, c) in enumerate(bands):
            y0 = p * H
            d_bg.polygon([(0, y0 - 140), (W, y0 + 140), (W, H + 300), (0, H + 300)], fill=_rgb(c))
            for yy in range(int(y0 - 170), int(y0 + 200), 22):  # trame de points d'imprimerie
                for xx in range(-20, W + 20, 22):
                    k = 1 - abs((yy - (y0 - 140 + 280 * xx / W)) / 160)
                    if k > 0:
                        r = 2 + 5 * k
                        d_bg.ellipse((xx - r, yy - r, xx + r, yy + r), fill=(0, 0, 0, 34))
        for x, y, w, h, a in wins:
            d_bg.polygon(_rot_rect(x, y, w, h, a, 26), fill=(0, 0, 0, 255))
            d_bg.polygon(_rot_rect(x, y, w, h, a, 12), fill=(255, 255, 255, 255))
        top = min(_rot_rect(*wins[0])[i][1] for i in range(4))
        if title:
            f, lines = _fit(title, "Anton-Regular.ttf", 880, 2, 104, 66)
            lh = f.size * 1.16
            strip = Image.new("RGBA", (W, int(lh * len(lines) + 70)), (0, 0, 0, 0))
            sd = ImageDraw.Draw(strip)
            for i, ln in enumerate(lines):
                tw = f.getlength(ln) + 56
                cy = 35 + lh * (i + .5)
                sd.rectangle((W / 2 - tw / 2, cy - lh / 2, W / 2 + tw / 2, cy + lh / 2), fill=(10, 10, 12, 255))
                _text(strip, (W / 2, cy + 4), ln, f, (255, 255, 255, 255))
            strip = strip.rotate(2, resample=Image.Resampling.BICUBIC, expand=True)
            fg.alpha_composite(strip, (int(W / 2 - strip.width / 2), int(max(150, top - 40 - strip.height))))
        if kicker:
            tag = Image.new("RGBA", (W, 140), (0, 0, 0, 0))
            _pill(tag, W / 2, 70, kicker, _font("Anton-Regular.ttf", 42), _rgb(GOLD), (10, 10, 12, 255), h=74,
                  outline=(10, 10, 12, 255))
            tag = tag.rotate(-3, resample=Image.Resampling.BICUBIC)
            fg.alpha_composite(tag, (0, 40))

    elif name == "versus":
        d_bg.rectangle((0, 0, W, H), fill=(8, 8, 10, 150))
        (x1, y1, w1, h1, _), (x2, y2, w2, h2, _) = wins
        mid = (y1 + h1 + y2) / 2
        d_fg.rectangle((0, mid - 9, W, mid + 9), fill=_rgb(GOLD))
        d_fg.ellipse((W / 2 - 74, mid - 74, W / 2 + 74, mid + 74), fill=(10, 10, 12, 255), outline=_rgb(GOLD),
                     width=8)
        _text(fg, (W / 2, mid + 4), "VS", _font("Anton-Regular.ttf", 84), _rgb(GOLD))
        lf = _font("Anton-Regular.ttf", 54)
        for (x, y, w, h, _), lab, col in ((wins[0], spec.get("lab1") or "BEFORE", "#FFFFFF"),
                                          (wins[1], spec.get("lab2") or "AFTER", PINK)):
            tw = lf.getlength(lab.upper()) + 48
            d_fg.rectangle((x + 30, y + 30, x + 30 + tw, y + 112), fill=(10, 10, 12, 230))
            d_fg.rectangle((x + 30, y + 30, x + 40, y + 112), fill=_rgb(col))
            _text(fg, (x + 30 + tw / 2 + 6, y + 73), lab.upper(), lf, _rgb(col))
        if title:
            f, lines = _fit(title, "Anton-Regular.ttf", 980, 2, 100, 64)
            lh = f.size * 1.08
            y0 = y1 / 2 - lh * (len(lines) - 1) / 2 + 6
            for i, ln in enumerate(lines):
                _text(fg, (W / 2, y0 + i * lh), ln, f, _rgb(GOLD) if i == len(lines) - 1 else (255, 255, 255, 255),
                      shadow=10)

    elif name == "news":
        d_bg.rectangle((0, 0, W, H), fill=(6, 10, 22, 165))
        x, y, w, h, _ = wins[-1]
        top = wins[0][1]
        red, ink = _rgb("#E5202E"), (12, 12, 14, 255)
        bf = _font("Anton-Regular.ttf", 44)
        d_fg.rectangle((48, top - 96, 48 + bf.getlength("LIVE") + 76, top - 30), fill=red)
        d_fg.ellipse((66, top - 72, 86, top - 52), fill=(255, 255, 255, 255))
        _text(fg, (102, top - 61), "LIVE", bf, (255, 255, 255, 255), anchor="lm")
        if kicker:
            _text(fg, (48 + bf.getlength("LIVE") + 104, top - 61), kicker, bf, (255, 255, 255, 255), anchor="lm")
        yb = y + h
        band = 252  # hauteur fixe (deux lignes) : l'apercu du site tombe juste
        d_fg.rectangle((0, yb + 70, W, yb + 70 + band), fill=(246, 244, 238, 255))
        d_fg.rectangle((0, yb + 70, 16, yb + 70 + band), fill=red)
        if title:
            tf, lines = _fit(title, "Anton-Regular.ttf", 960, 2, 92, 60)
            lh = tf.size * 1.12
            for i, ln in enumerate(lines):
                _text(fg, (54, yb + 70 + band / 2 + lh * (i - (len(lines) - 1) / 2) + 4), ln, tf, ink,
                      anchor="lm")
        tagf = _font("Anton-Regular.ttf", 46)
        d_fg.rectangle((0, yb + 6, tagf.getlength("BREAKING NEWS") + 70, yb + 72), fill=red)
        _text(fg, (36, yb + 40), "BREAKING NEWS", tagf, (255, 255, 255, 255), anchor="lm")
        ty = yb + 70 + band
        d_fg.rectangle((0, ty, W, ty + 64), fill=ink)
        tick = "   •   ".join(t for t in (spec.get("ticker") or title or "GTA 6").split()) + "   •   "
        tkf = _font("BarlowCondensed-Bold.ttf", 40)
        tw = int(tkf.getlength(tick))
        strip = Image.new("RGBA", (tw * 2, 64), (0, 0, 0, 0))
        _text(strip, (0, 33), tick + tick, tkf, _rgb("#FFD23F"), anchor="lm")
        extras.append((save(strip, "tick"), "", f"-mod(t*150,{tw})", ty, ""))
        progress(0, ty + 64, W, 8, "#E5202E", "#FFD23F")
    return save(bg, "bg"), save(fg, "fg"), extras


def theme_assets(outdir):
    """Calques des themes sans titre ni pastille, en 270x480, pour l'apercu de l'editeur du site."""
    from PIL import Image
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, th in THEMES.items():
        for lay in ("full", "split"):
            sp = {"theme": name, "title": "", "kicker": "", "lab1": "", "lab2": "", "credit": "",
                  "ticker": "#GTA6 #GTAVI #ViceCity #Leonida"}
            bg, fg, extras = theme_layers(sp, th[lay], 10, outdir)
            for p, tag in ((bg, "bg"), (fg, "fg")):
                Image.open(p).resize((270, 480), Image.Resampling.LANCZOS).save(outdir / f"{name}-{lay}-{tag}.png",
                                                                                 optimize=True)
            for p in [bg, fg] + [e[0] for e in extras]:
                p.unlink(missing_ok=True)
    print(f"calques des themes dans {outdir}")
    return 0


def themed_vargs(spec, wins, iw, ih, segs, pre, samples, workdir):
    """Graphe ffmpeg d'un short habille : fond flou, decor, video dans les fenetres,
    avant-plan, animations. Renvoie (entrees en plus, filter_complex)."""
    W, H = 1080, 1920
    s0 = spec["trim"][0]
    dur = sum(b - a for a, b in segs)
    names = ["cam", "game"] if len(wins) == 2 else ["crop"]
    if spec["theme"] == "versus" and spec["layout"] != "split":
        names = ["crop", "crop"]  # un seul cadrage : les deux moities montrent le meme plan
    cuts = plans(spec)
    g, k = [], len(wins)

    def region(r, n, j):
        ww, wh = wins[j][2], wins[j][3]
        if r.get(n):
            return rect_px(r[n], iw, ih, ww / wh)
        if n == "crop" and r.get("cam"):
            return rect_px(r["cam"], iw, ih, ww / wh)
        return compute_crop(iw, ih, .5, .5, ww / wh)
    if len(cuts) == 1:
        r = cuts[0][2]
        g.append(f"[0:v]{pre}split={k}" + "".join(f"[a{j}]" for j in range(k)))
        for j in range(k):
            ww, wh = wins[j][2], wins[j][3]
            if k == 1 and spec["track"] and samples:
                cw, ch, _, _ = compute_crop(iw, ih, .5, .5, ww / wh)
                x, y = track_crop(samples, segs, iw, ih, cw, ch)
            else:
                cw, ch, x, y = region(r, names[j], j)
            g.append(f"[a{j}]crop=w={cw}:h={ch}:x='{x}':y='{y}',scale={ww}:{wh},setsar=1[w{j}]")
    else:
        g.append(f"[0:v]split={len(cuts)}" + "".join(f"[s{i}]" for i in range(len(cuts))))
        for i, (a, b, r) in enumerate(cuts):
            g.append(f"[s{i}]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,split={k}"
                     + "".join(f"[c{i}_{j}]" for j in range(k)))
            for j in range(k):
                cw, ch, x, y = region(r, names[j], j)
                g.append(f"[c{i}_{j}]crop={cw}:{ch}:{x}:{y},scale={wins[j][2]}:{wins[j][3]},setsar=1[d{i}_{j}]")
        for j in range(k):
            g.append("".join(f"[d{i}_{j}]" for i in range(len(cuts)))
                     + f"concat=n={len(cuts)}:v=1:a=0,{pre or 'null,'}null[w{j}]")
    bg, fg, extras = theme_layers(spec, wins, dur, workdir)
    inputs = ["-loop", "1", "-i", str(bg.resolve()), "-loop", "1", "-i", str(fg.resolve())]
    m = k - 1  # le plan principal (jeu) sert aussi de fond flou
    g.append(f"[w{m}]split=2[w{m}][bb]")
    g.append(f"[bb]scale=216:384:force_original_aspect_ratio=increase,crop=216:384,boxblur=14:2,"
             f"scale={W}:{H},eq=brightness=-0.06:saturation=1.25[base]")
    g.append("[base][1:v]overlay=0:0:shortest=1[l0]")
    cur = "l0"
    for j, (x, y, w, h, a) in enumerate(wins):
        src = f"w{j}"
        if a:
            g.append(f"[w{j}]format=rgba,rotate={a}*PI/180:c=none:ow=rotw({a}*PI/180):oh=roth({a}*PI/180)[r{j}]")
            src, pos = f"r{j}", f"x='{x + w / 2:.0f}-w/2':y='{y + h / 2:.0f}-h/2'"
        else:
            pos = f"x={x}:y={y}"
        g.append(f"[{cur}][{src}]overlay={pos}:shortest=1[p{j}]")
        cur = f"p{j}"
    g.append(f"[{cur}][2:v]overlay=0:0:shortest=1[f0]")
    cur = "f0"
    for n, (png, filt, x, y, en) in enumerate(extras):
        inputs += ["-loop", "1", "-i", str(png.resolve())]
        src = f"{3 + n}:v"
        if filt:
            g.append(f"[{src}]{filt}[x{n}]")
            src = f"x{n}"
        opt = f":enable='{en}'" if en else ""
        g.append(f"[{cur}][{src}]overlay=x='{x}':y='{y}'{opt}:shortest=1[e{n}]")
        cur = f"e{n}"
    return inputs, g, cur, [bg, fg] + [e[0] for e in extras]


def render_spec(raw, words, spec, out, samples=None):
    """Rend un short a partir de son extrait brut. `words` en temps de l'extrait."""
    iw, ih = video_dims(raw)
    W, H, fs = FORMATS[spec["format"]]
    s0, s1 = spec["trim"]
    segs = keep_segments(words, s0, s1) if spec["cut"] else [(s0, s1)]
    subs = out.with_suffix(".ass")
    pre, af = "", []
    if segs != [(s0, s1)]:  # retrait des blancs : on ne garde que les morceaux parles
        sel = "+".join(f"between(t,{a - s0:.3f},{b - s0:.3f})" for a, b in segs)
        pre = f"select='{sel}',setpts=N/FRAME_RATE/TB,"
        af = ["-af", f"aselect='{sel}',asetpts=N/SR/TB"]
    wins = theme_windows(spec)
    if wins:  # habillage GTA : son propre graphe, le reste de la fonction ne change pas
        th = THEMES[spec["theme"]]
        sub_y = th["sub"]["split" if len(wins) == 2 else "full"]
        build_ass(words, segs, subs, "9:16", "", subs=spec["subs"], subc=spec.get("subc", "#FFFFFF"),
                  margin_v=round(H - sub_y - fs * .55), font="Anton", accent=th["accent"])
        fonts = out.parent / "fonts"
        fonts.mkdir(exist_ok=True)
        for f in FONTS.glob("*.ttf"):
            (fonts / f.name).write_bytes(f.read_bytes())
        inputs, g, cur, tmp = themed_vargs(spec, wins, iw, ih, segs, pre, samples, out.parent)
        g.append(f"[{cur}]subtitles={subs.name}:fontsdir=fonts,format=yuv420p[v]")
        print(f"    habillage : {spec['theme']} ({len(wins)} fenetre(s))")
        run(["ffmpeg", "-y", "-ss", str(s0), "-to", str(s1), "-i", str(raw.resolve()), *inputs,
             "-filter_complex", ";".join(g), "-map", "[v]", "-map", "0:a?", *af,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", "30",
             "-c:a", "aac", "-b:a", "128k", str(out.name)], cwd=str(out.parent.resolve()))
        subs.unlink(missing_ok=True)
        for p in tmp:
            p.unlink(missing_ok=True)
        for f in fonts.glob("*"):
            f.unlink()
        fonts.rmdir()
        return out
    split = spec["layout"] == "split" and spec["cam"] and spec["game"] and spec["format"] in ("9:16", "4:5")
    look = {"color": spec.get("color", ACCENT), "subc": spec.get("subc", "#FFFFFF")}
    cuts = plans(spec)
    if len(cuts) > 1:  # cadrages valides moment par moment : un plan chacun, mis bout a bout
        ht, hb = split_heights(H)
        build_ass(words, segs, subs, spec["format"], spec["title"], subs=spec["subs"], **look,
                  margin_v=hb - round(fs * .5) if split else None)

        def box(r, ratio):
            return rect_px(r, iw, ih, ratio) if r else compute_crop(iw, ih, 0.5, 0.5, ratio)
        graph = [f"[0:v]split={len(cuts)}" + "".join(f"[s{i}]" for i in range(len(cuts)))]
        for i, (a, b, r) in enumerate(cuts):
            head = f"[s{i}]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,"
            if split:
                cw, ch, cx, cy = box(r["cam"], W / ht)
                gw, gh, gx, gy = box(r["game"], W / hb)
                graph.append(f"{head}split=2[a{i}][b{i}];"
                             f"[a{i}]crop={cw}:{ch}:{cx}:{cy},scale={W}:{ht},setsar=1[t{i}];"
                             f"[b{i}]crop={gw}:{gh}:{gx}:{gy},scale={W}:{hb},setsar=1[u{i}];"
                             f"[t{i}][u{i}]vstack[p{i}]")
            else:
                cw, ch, x, y = box(r["crop"], W / H)
                graph.append(f"{head}crop={cw}:{ch}:{x}:{y},scale={W}:{H},setsar=1[p{i}]")
        graph.append("".join(f"[p{i}]" for i in range(len(cuts)))
                     + f"concat=n={len(cuts)}:v=1:a=0,{pre}subtitles={subs.name}[v]")
        vargs = ["-filter_complex", ";".join(graph), "-map", "[v]", "-map", "0:a?"]
        print(f"    {len(cuts)} cadrages choisis a la main")
    elif split:
        ht, hb = split_heights(H)
        # sous-titres a cheval sur la jonction des deux panneaux
        build_ass(words, segs, subs, spec["format"], spec["title"],
                  margin_v=hb - round(fs * .5), subs=spec["subs"], **look)
        fw_, fh_, fx, fy = rect_px(spec["cam"], iw, ih, W / ht)
        gw, gh, gx, gy = rect_px(spec["game"], iw, ih, W / hb)
        graph = (f"[0:v]{pre}split=2[a][b];"
                 f"[a]crop={fw_}:{fh_}:{fx}:{fy},scale={W}:{ht},setsar=1[top];"
                 f"[b]crop={gw}:{gh}:{gx}:{gy},scale={W}:{hb},setsar=1[bot];"
                 f"[top][bot]vstack,subtitles={subs.name}[v]")
        vargs = ["-filter_complex", graph, "-map", "[v]", "-map", "0:a?"]
    else:
        build_ass(words, segs, subs, spec["format"], spec["title"], subs=spec["subs"], **look)
        if spec["track"] and samples:
            cw, ch, _, _ = compute_crop(iw, ih, 0.5, 0.5, W / H)
            x, y = track_crop(samples, segs, iw, ih, cw, ch)
        elif spec["crop"]:  # cadre choisi a la main : fixe, et zoome s'il est plus petit
            cw, ch, x, y = rect_px(spec["crop"], iw, ih, W / H)
        else:
            cw, ch, x, y = compute_crop(iw, ih, 0.5, 0.5, W / H)
        vargs = ["-vf", f"{pre}crop=w={cw}:h={ch}:x='{x}':y='{y}',scale={W}:{H},setsar=1,"
                        f"subtitles={subs.name}"]
    print(f"    mise en page : {'ecran partage' if split else 'plein cadre'}")
    # cwd=dossier de sortie pour eviter l'echappement des chemins Windows dans subtitles
    run(["ffmpeg", "-y", "-ss", str(s0), "-to", str(s1), "-i", str(raw.resolve()),
         *vargs, *af, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-c:a", "aac", "-b:a", "128k", str(out.name)], cwd=str(out.parent.resolve()))
    subs.unlink(missing_ok=True)
    return out


def render_clip(video, words, clip, workdir, index, iw, ih, use_face=True, opts=None,
                total=None):
    """Extrait brut (moment + marge) puis short. Renvoie l'entree de shorts.json."""
    opts = opts or parse_opts("{}")
    name = f"short_{index:02d}_{clip['score']:03d}_{slugify(clip['title'])}"
    total = total or ffprobe_duration(video)
    r0, r1 = max(0.0, clip["start"] - EDGE), min(total, clip["end"] + EDGE)
    raw = workdir / f"raw_{index:02d}.mp4"
    # extrait reencode (coupe exacte) : c'est lui que l'editeur montre et recadre
    run(["ffmpeg", "-y", "-ss", str(r0), "-to", str(r1), "-i", str(video.resolve()),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "160k",
         str(raw)])
    rel = [{"word": w["word"].strip(), "start": round(float(w["start"]) - r0, 2),
            "end": round(float(w["end"]) - r0, 2)}
           for w in words if float(w["end"]) > r0 and float(w["start"]) < r1]
    s0, s1 = clip["start"] - r0, clip["end"] - r0
    samples = face_samples(raw, s0, s1) if use_face else []
    spec = auto_spec(samples, opts, iw, ih, s0, s1, clip["title"] if opts["title"] else "",
                     clip.get("hashtags", ""))
    out = render_spec(raw, rel, spec, workdir / f"{name}.mp4", samples)
    # pack de publication, pret a copier-coller
    tags = clip.get("hashtags", "")
    (workdir / f"{name}.txt").write_text(
        f"{clip['title']}\n\n{clip.get('caption', '')}\n\n{tags}\n", encoding="utf-8")
    return {**clip, "file": out.name, "raw": raw.name, "dur": round(r1 - r0, 2), "spec": spec,
            "words": [[w["word"], w["start"], w["end"]] for w in rel]}


def ask_edit(spec, words, dur, key):
    """Consigne en francais courant -> changements de reglages (trim, titre, format...)."""
    shown = {k: spec[k] for k in ("format", "layout", "trim", "cut", "subs", "title")}
    msg = (
        f"Reglages actuels du short : {json.dumps(shown, ensure_ascii=False)}\n"
        f"L'extrait brut dure {dur:.1f} s ; trim = [debut, fin] en secondes dans cet extrait.\n"
        f"Transcription horodatee de l'extrait :\n{transcript_lines(words, 6)}\n\n"
        f"Demande du monteur : \u00ab {spec['ask']} \u00bb\n\n"
        'Reponds STRICTEMENT en JSON : {"changes": {...}} avec seulement les cles a modifier '
        "parmi trim, title, cut, subs, format (9:16, 4:5, 1:1, 16:9) et layout (full ou split). "
        "Sans emoji.\n" + ENGLISH)
    try:
        raw = groq_chat([{"role": "system", "content": "Tu es monteur de shorts. Tu ne reponds qu'en JSON."},
                         {"role": "user", "content": msg}], key)
        ch = json.loads(raw).get("changes") or {}
    except (urllib.error.URLError, KeyError, ValueError, AttributeError) as e:
        print(f"[info] consigne IA non appliquee ({e})")
        return spec
    allowed = ("trim", "title", "cut", "subs", "format", "layout")
    return {**spec, **{k: v for k, v in ch.items() if k in allowed}, "ask": ""}


def edit_main(args):
    """Refait un short depuis son extrait brut avec les reglages de l'editeur."""
    meta = json.loads(pathlib.Path(args.meta).read_text(encoding="utf-8"))
    raw = pathlib.Path(args.raw)
    entry = next((m for m in meta if m.get("raw") == raw.name), None)
    if not entry:
        sys.exit(f"[erreur] {raw.name} absent de shorts.json")
    words = [{"word": w, "start": a, "end": b} for w, a, b in entry["words"]]
    dur = ffprobe_duration(raw)
    try:
        spec = parse_spec(json.loads(args.spec), dur)
    except json.JSONDecodeError:
        sys.exit("[erreur] reglages illisibles")
    if spec["ask"]:
        spec = parse_spec(ask_edit(spec, words, dur, load_groq_key()), dur)
    samples = None
    if spec["track"] and spec["layout"] == "full" and not spec["keys"]:
        samples = face_samples(raw, *spec["trim"])
    out = pathlib.Path(args.edit_out)
    render_spec(raw, words, spec, out, samples)
    spec["ask"] = ""
    out.with_suffix(".json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    print(f"Edition prete : {out}")
    return 0


# ----------------------------------------------------------------------------- main

def main():
    sys.stdout.reconfigure(errors="replace")  # console Windows cp1252 : pas de crash sur un titre
    ap = argparse.ArgumentParser(description="Clippeur maison : lien -> shorts sous-titres")
    ap.add_argument("source", nargs="?", help="URL YouTube ou chemin d'un fichier video")
    ap.add_argument("--clips", type=int, default=4, help="nombre de shorts (defaut 4)")
    ap.add_argument("--out", default="sorties", help="dossier de sortie (defaut: sorties)")
    ap.add_argument("--no-face", action="store_true",
                    help="desactiver le suivi du visage (recadrage centre)")
    ap.add_argument("--opts", default=os.environ.get("OPTS", "{}"),
                    help='reglages JSON : {"format":"9:16","cat":"auto","cut":true,'
                         '"title":true,"find":"","start":0,"end":600}')
    ap.add_argument("--edit", type=int, help="refaire le short N depuis son extrait brut")
    ap.add_argument("--meta", help="shorts.json du travail (avec --edit)")
    ap.add_argument("--raw", help="extrait brut raw_NN.mp4 (avec --edit)")
    ap.add_argument("--spec", default=os.environ.get("SPEC", "{}"), help="reglages JSON de l'editeur")
    ap.add_argument("--edit-out", help="fichier de sortie de l'edition")
    ap.add_argument("--selfcheck", action="store_true", help="tests internes rapides")
    ap.add_argument("--theme-assets", help="ecrit les calques d'apercu des themes dans ce dossier")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.theme_assets:
        return theme_assets(args.theme_assets)
    if args.edit:
        if not (args.meta and args.raw and args.edit_out):
            ap.error("--edit demande --meta, --raw et --edit-out")
        return edit_main(args)
    if not args.source:
        ap.error("donne une URL ou un fichier (ou --selfcheck)")

    key = load_groq_key()
    base_out = pathlib.Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)
    src = args.source
    is_url = src.startswith("http://") or src.startswith("https://")
    job = base_out / (slugify(src.rsplit("/", 1)[-1]) if not is_url else "job")
    # dossier unique par video
    tag = re.sub(r"\W+", "", src)[-11:] or "job"
    workdir = base_out / f"video_{tag}"
    workdir.mkdir(parents=True, exist_ok=True)

    opts = parse_opts(args.opts)
    video = download(src, workdir) if is_url else pathlib.Path(src)
    if not video.exists():
        sys.exit(f"[erreur] fichier introuvable : {video}")
    if opts["start"] is not None:  # on ne garde que la partie demandee
        part = workdir / "partie.mp4"
        run(["ffmpeg", "-y", "-ss", str(opts["start"]), "-to", str(opts["end"]),
             "-i", str(video), "-c", "copy", str(part)])
        video = part

    words = transcribe(video, workdir, key)
    clips = select_clips(words, key, args.clips, opts["cat"], opts["find"], opts)
    iw, ih = video_dims(video)
    print(f"[4/4] rendu de {len(clips)} short(s)...")

    meta, total = [], ffprobe_duration(video)
    for i, clip in enumerate(clips, 1):
        entry = render_clip(video, words, clip, workdir, i, iw, ih,
                            use_face=not args.no_face, opts=opts, total=total)
        meta.append(entry)
        print(f"  - {entry['file']}  (score {clip['score']}/100)  {clip['title']}")
    (workdir / "partie.mp4").unlink(missing_ok=True)  # sinon publie avec les shorts
    (workdir / "shorts.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nTermine. {len(meta)} short(s) dans : {workdir.resolve()}")


def selfcheck():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(65.5) == "0:01:05.50"
    assert slugify("GTA 6 : le GROS secret !") == "gta-6-le-gros-secret"
    cl = parse_clips('{"clips":[{"start":10,"end":40,"title":"Hook","score":88,'
                     '"hashtags":["#a","#b"]},{"start":5,"end":3,"title":"bad","score":99}]}',
                     total=300)
    assert len(cl) == 1 and cl[0]["title"] == "Hook", cl  # clip invalide (end<start) rejete
    assert cl[0]["hashtags"] == "#a #b", cl  # liste de hashtags aplatie
    fb = fallback_clips(600, 3)
    assert len(fb) == 3 and fb[0]["start"] == 0
    assert fb[-1]["end"] <= 600
    assert "caption" in fb[0] and "hashtags" in fb[0]  # schema uniforme
    # recadrage : source large -> rogne en largeur, centre sur cx
    cw, ch, x, y = compute_crop(1920, 1080, cx=0.75, cy=0.5)
    assert ch == 1080 and cw == 608 and y == 0 and 0 <= x <= 1920 - cw, (cw, ch, x, y)
    assert x > (1920 - cw) / 2  # decale a droite car cx=0.75
    cw2, ch2, x2, y2 = compute_crop(1080, 1920, cx=0.5, cy=0.5)  # deja plus haut que 9:16
    assert cw2 == 1080 and x2 == 0, (cw2, ch2, x2, y2)
    # reglages : defauts, valeurs invalides ignorees, plage limitee a une heure
    o = parse_opts("{}")
    assert o["format"] == "9:16" and o["cut"] and o["title"] and o["start"] is None, o
    o = parse_opts('{"format":"4:5","cat":"irl","cut":false,"find":"  il  tombe ",'
                   '"start":60,"end":600}')
    assert (o["format"], o["cat"], o["cut"], o["find"], o["start"]) == \
        ("4:5", "irl", False, "il tombe", 60.0), o
    assert parse_opts('{"format":"x","cat":"rm -rf","start":0,"end":9999}')["start"] is None
    assert parse_opts("pas du json")["format"] == "9:16"
    assert compute_crop(1920, 1080, ratio=1)[:2] == (1080, 1080)
    # retrait des blancs : trou de 3 s entre deux phrases -> deux morceaux
    ws = [{"word": "a", "start": 10.0, "end": 10.5}, {"word": "b", "start": 10.6, "end": 11.0},
          {"word": "c", "start": 14.0, "end": 14.5}]
    segs = keep_segments(ws, 9.0, 16.0)
    assert segs == [(9.85, 11.15), (13.85, 14.65)], segs
    assert abs(remap(14.0, segs) - 1.45) < 1e-9 and abs(remap(12.0, segs) - 1.3) < 1e-9
    assert keep_segments([], 5, 9) == [(5, 9)]
    assert ass_text("a{\\b1}\\Nc") == "a(/b1)/Nc"
    # suivi du visage : bruit ignore (zone morte), vrai deplacement suivi avec un glissement
    smp = [(i * .5, .5 + (.01 if i % 2 else 0), .5, .05) for i in range(6)] + \
          [(3 + i * .5, .8, .5, .05) for i in range(6)]
    tr = smooth_track(smp)
    assert tr[0] == (0.0, .5) and tr[-1][1] == .8 and len(tr) == 3, tr
    assert smooth_track([(0, None, None, None)]) == []
    e = piecewise_expr([(0.0, 100), (2.0, 300)])
    assert e == "if(lt(t,2.000),100+(200)*(t-0.000)/2.000,300)", e
    assert piecewise_expr([(0.0, 7)]) == "7"
    # mise en page : petit visage fixe dans un coin d'une source 16:9 -> ecran partage
    cam = [(i * .5, .85, .8, .06) for i in range(10)]
    assert plan_layout(cam, parse_opts("{}"), 1920, 1080) == "split"
    noisy = cam + [(5.0, .5, .46, .055)]  # un faux visage dans le jeu
    assert plan_layout(noisy, parse_opts("{}"), 1920, 1080) == "split"
    assert plan_layout(cam, parse_opts('{"format":"1:1"}'), 1920, 1080) == "full"
    assert plan_layout(cam, parse_opts('{"layout":"full"}'), 1920, 1080) == "full"
    assert plan_layout(cam, parse_opts('{"cat":"irl"}'), 1920, 1080) == "full"
    big = [(i * .5, .5, .4, .25) for i in range(10)]
    assert plan_layout(big, parse_opts("{}"), 1920, 1080) == "full"
    assert plan_layout(big, parse_opts('{"layout":"split"}'), 1920, 1080) == "split"
    assert plan_layout([], parse_opts('{"layout":"split"}'), 1920, 1080) == "full"
    bw, bh, bx, by = facecam_crop(cam, 1920, 1080, 1080 / 768)
    assert bx + bw <= 1920 and by + bh <= 1080 and bw % 2 == 0 and bh % 2 == 0, (bw, bh, bx, by)
    x, y = track_crop(cam, [(0, 5)], 1920, 1080, 608, 1080)
    assert y == "0" and x.isdigit() and int(x) == 1920 - 608, (x, y)  # visage a droite, clampe
    # editeur : reglages invalides ramenes a des valeurs sures
    sp = parse_spec({"format": "4:5", "layout": "split", "trim": [2, 9],
                     "crop": {"x": .1, "y": 0, "w": .3, "h": 1}, "cam": {"x": 2, "y": 0, "w": .1, "h": .1},
                     "title": "Titre {x}", "track": "oui"}, dur=20)
    assert sp["format"] == "4:5" and sp["layout"] == "split" and sp["trim"] == [2.0, 9.0], sp
    assert sp["crop"]["w"] == .3 and sp["cam"] is None and sp["track"] is False, sp
    assert sp["title"] == "Titre (x)" and sp["cut"] and sp["subs"], sp
    assert parse_spec({"trim": [5, 99]}, dur=20)["trim"] == [0.0, 20], "trim hors extrait"
    assert parse_spec("n'importe quoi", dur=8)["format"] == "9:16"
    # cadre de l'editeur -> pixels au bon ratio, dans l'image
    w, h, x, y = rect_px({"x": .9, "y": .1, "w": .2, "h": .5}, 1920, 1080, 9 / 16)
    assert abs(w / h - 9 / 16) < .01 and x + w <= 1920 and y + h <= 1080, (w, h, x, y)
    w, h, x, y = rect_px({"x": 0, "y": 0, "w": 1, "h": 1}, 1920, 1080, 9 / 16)
    assert h == 1080 and w == 606, (w, h)
    a = auto_spec(cam, parse_opts("{}"), 1920, 1080, 3, 20, "T")
    assert a["layout"] == "split" and a["track"] and a["trim"] == [3, 20], a
    assert all(0 <= a["game"][k] <= 1 for k in "xywh"), a
    assert parse_spec(a, 30)["cam"] == a["cam"]  # ce que le moteur ecrit, l'editeur le relit
    # couleurs et cadrages moment par moment
    r1, r2 = {"x": .1, "y": 0, "w": .3, "h": 1}, {"x": .6, "y": 0, "w": .3, "h": 1}
    k = parse_spec({"trim": [2, 12], "crop": r1, "color": "#ffd60a", "subc": "rouge",
                    "keys": [{"t": 7, "crop": r2}, {"t": 2, "crop": r2}, {"t": "x"}, {"t": 5}]}, dur=20)
    assert k["color"] == "#FFD60A" and k["subc"] == "#FFFFFF", k
    assert [x["t"] for x in k["keys"]] == [2, 7], k["keys"]
    assert plans(k) == [(0, 5, {"crop": r2, "cam": None, "game": None}),
                        (5, 10, {"crop": r2, "cam": None, "game": None})], plans(k)
    k["keys"] = [{"t": 6, "crop": r2}]
    assert [(a, b, r["crop"]["x"]) for a, b, r in plans(k)] == [(0, 4, .1), (4, 10, .6)]
    o = parse_opts('{"dmin": 61, "dmax": 200, "loop": false}')
    assert (o["dmin"], o["dmax"], o["loop"]) == (61, 90, False), o
    assert (parse_opts('{"dmin": 80, "dmax": 82}')["dmin"], parse_opts("{}")["loop"]) == (40, True)
    # parole continue : un mot toutes les 0,5 s, "so" revient a 70 s
    talk = [{"word": "so" if i in (0, 140) else f"w{i}", "start": i * .5, "end": i * .5 + .4}
            for i in range(300)]
    lo = fit_clip({"start": 0, "end": 60}, talk, {**o, "dmin": 45, "dmax": 75, "loop": True, "cut": True})
    assert lo["start"] == 0 and abs(lo["end"] - 69.98) < .05, lo  # coupe juste avant le "so"
    nl = fit_clip({"start": 0, "end": 20}, talk, {**o, "dmin": 45, "dmax": 75, "loop": False, "cut": True})
    assert 45 <= kept(talk, nl["start"], nl["end"], True) <= 75, nl
    assert ass_color("#FF4322") == "&H002243FF" and ink_on("#FFD60A") == "#0A0A0B" and ink_on("#111111") == "#FFFFFF"
    ch = chunk_lines("aaaaaa\nbbbbbb\ncccccc", size=15)
    assert ch == ["aaaaaa\nbbbbbb\n", "cccccc\n"], ch
    cl = [{"start": 0, "end": 30, "score": 70}, {"start": 20, "end": 50, "score": 90},
          {"start": 60, "end": 80, "score": 50}]
    assert [c["score"] for c in best_distinct(cl, 3)] == [90, 50]
    assert slugify("J'achète un supermarché !") == "jachete-un-supermarche"
    # habillages : compte a rebours, reglages valides, fenetres selon la mise en page
    import datetime
    assert gta_kicker(datetime.date(2026, 11, 16)) == "GTA 6 · 3 DAYS LEFT"
    assert gta_kicker(datetime.date(2026, 11, 19)) == "GTA 6 · OUT TODAY"
    assert gta_kicker(datetime.date(2026, 11, 21)) == "GTA 6 · DAY 3"
    t = parse_spec({"theme": "neon", "kicker": "a{b}", "lab1": "x" * 50}, 10)
    assert t["theme"] == "neon" and t["kicker"] == "a(b)" and len(t["lab1"]) == 20, t
    assert parse_spec({"theme": "rm -rf"}, 10)["theme"] == "none"
    assert parse_opts('{"theme":"news"}')["theme"] == "news" and parse_opts("{}")["theme"] == "none"
    assert theme_windows(t) == THEMES["neon"]["full"]
    assert theme_windows({**t, "layout": "split", "cam": r1, "game": r2}) == THEMES["neon"]["split"]
    assert theme_windows({**t, "format": "4:5"}) is None and theme_windows({**t, "theme": "none"}) is None
    assert len(theme_windows({**t, "theme": "versus"})) == 2  # toujours deux plans
    assert all(w[2] % 2 == 0 and w[3] % 2 == 0 for th in THEMES.values() for k in ("full", "split") for w in th[k])
    print("selfcheck OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
