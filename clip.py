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
    s = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
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
    }
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


def select_clips(words, key, n, cat="auto", find=""):
    print("[3/4] choix des meilleurs moments (LLM Groq)...")
    total = int(float(words[-1]["end"]))
    sys_msg = ("Tu es un monteur expert en shorts viraux (TikTok, Reels, YouTube Shorts). "
               "A partir d'une transcription horodatee, tu reperes les meilleurs extraits "
               "autonomes a transformer en shorts verticaux.")
    user_msg = (
        f"Voici la transcription d'une video de {total}s. Chaque ligne commence par son "
        f"temps de debut en secondes.\n\n{transcript_lines(words)}\n\n"
        f"Choisis les {n} meilleurs extraits. Reponds STRICTEMENT en JSON :\n"
        '{\"clips\":[{\"start\":<debut_en_secondes>,\"end\":<fin_en_secondes>,'
        '\"title\":\"<titre accrocheur, max 60 caracteres>\",'
        '\"score\":<viralite 0-100>,\"reason\":\"<pourquoi ca marche, 1 phrase>\",'
        '\"caption\":\"<legende de post, 1-2 phrases accrocheuses>\",'
        '\"hashtags\":[\"<4 a 6 hashtags pertinents>\"]}]}\n'
        "Regles : chaque extrait dure entre 15 et 60 secondes, start et end sont des "
        "nombres en secondes pris dans la video, les extraits ne se chevauchent pas, "
        "tries du meilleur score au moins bon. Titre, legende et hashtags dans la "
        "langue de la video, sans aucun emoji. Rien d'autre que le JSON.")
    if CATS.get(cat):
        user_msg += "\n\nContexte : " + CATS[cat]
    if find:
        user_msg += ("\n\nLe spectateur cherche un moment precis : \u00ab " + find + " \u00bb. "
                     "Ne retiens que des extraits qui y correspondent, du plus probable au "
                     "moins probable (le score dit a quel point ca correspond). S'il n'y a "
                     "pas de correspondance exacte, donne les passages les plus proches. "
                     "Titre, legende et hashtags restent dans la langue de la video.")
    try:
        raw = groq_chat([{"role": "system", "content": sys_msg},
                         {"role": "user", "content": user_msg}], key)
        clips = parse_clips(raw, total)
        if clips:
            return clips[:n]
        print("[info] reponse LLM inexploitable, decoupage regulier en secours.")
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError) as e:
        print(f"[info] LLM indisponible ({e}), decoupage regulier en secours.")
    return fallback_clips(total, n)


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
        if isinstance(tags, list):
            tags = " ".join(str(t).strip() for t in tags)
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


def fallback_clips(total, n):
    """Secours : n fenetres de 30s reparties si le LLM echoue."""
    dur = min(30.0, max(15.0, total / max(n, 1)))
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


def build_ass(words, segs, out_path, fmt="9:16", title="", group=3, margin_v=None,
              subs=True):
    """Sous-titres ASS : groupes de mots gros et centres, titre en haut si demande.
    `segs` = morceaux gardes de la source ; les temps sont recales dessus."""
    W, H, fs = FORMATS[fmt]
    header = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
        f"Style: Pop,Arial Black,{fs},&H00FFFFFF,&H00000000,&H00000000,"
        f"-1,1,6,2,2,80,80,{round(H * .135) if margin_v is None else margin_v}\n"
        # titre : texte noir sur bandeau orange (BorderStyle 3 = boite opaque)
        f"Style: Title,Arial Black,{round(fs * .62)},&H000B0A0A,&H002243FF,&H00000000,"
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
        text = ass_text(" ".join(w["word"].strip() for w in chunk).upper())
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
    }


def auto_spec(samples, opts, iw, ih, s0, s1, title):
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
    }


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
    split = spec["layout"] == "split" and spec["cam"] and spec["game"] and spec["format"] in ("9:16", "4:5")
    if split:
        ht, hb = split_heights(H)
        # sous-titres a cheval sur la jonction des deux panneaux
        build_ass(words, segs, subs, spec["format"], spec["title"],
                  margin_v=hb - round(fs * .5), subs=spec["subs"])
        fw_, fh_, fx, fy = rect_px(spec["cam"], iw, ih, W / ht)
        gw, gh, gx, gy = rect_px(spec["game"], iw, ih, W / hb)
        graph = (f"[0:v]{pre}split=2[a][b];"
                 f"[a]crop={fw_}:{fh_}:{fx}:{fy},scale={W}:{ht},setsar=1[top];"
                 f"[b]crop={gw}:{gh}:{gx}:{gy},scale={W}:{hb},setsar=1[bot];"
                 f"[top][bot]vstack,subtitles={subs.name}[v]")
        vargs = ["-filter_complex", graph, "-map", "[v]", "-map", "0:a?"]
    else:
        build_ass(words, segs, subs, spec["format"], spec["title"], subs=spec["subs"])
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
    spec = auto_spec(samples, opts, iw, ih, s0, s1, clip["title"] if opts["title"] else "")
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
        "Le titre reste dans la langue de la video, sans emoji.")
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
    if spec["track"] and spec["layout"] == "full":
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
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
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
    clips = select_clips(words, key, args.clips, opts["cat"], opts["find"])
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
    print("selfcheck OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
