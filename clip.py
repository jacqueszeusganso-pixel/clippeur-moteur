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


def detect_face_center(video, start, end):
    """Centre median du visage dominant sur l'extrait (OpenCV Haar). (0.5, 0.5) si rien."""
    try:
        import cv2
        cascade = _face_cascade()
    except Exception as e:  # cv2 absent ou KO -> recadrage centre
        print(f"[info] detection visage indisponible ({e}), recadrage centre.")
        return 0.5, 0.5
    cap = cv2.VideoCapture(str(video))
    dur = max(0.1, end - start)
    nsamp = int(min(15, max(4, dur)))
    xs, ys = [], []
    for i in range(nsamp):
        t = start + dur * (i + 0.5) / nsamp
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            continue
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        h, w = gray.shape[:2]
        xs.append((fx + fw / 2) / w)
        ys.append((fy + fh / 2) / h)
    cap.release()
    if not xs:
        return 0.5, 0.5
    clamp = lambda v: min(max(v, 0.0), 1.0)
    # ponytail: recadrage statique par clip (centre median). Suffit pour une tete
    # qui parle ; passer a un suivi image par image si les sujets bougent beaucoup.
    return clamp(statistics.median(xs)), clamp(statistics.median(ys))


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


def build_ass(words, segs, out_path, fmt="9:16", title="", group=3):
    """Sous-titres ASS : groupes de mots gros et centres, titre en haut si demande.
    `segs` = morceaux gardes de la source ; les temps sont recales dessus."""
    W, H, fs = FORMATS[fmt]
    header = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
        f"Style: Pop,Arial Black,{fs},&H00FFFFFF,&H00000000,&H00000000,"
        f"-1,1,6,2,2,80,80,{round(H * .135)}\n"
        # titre : texte noir sur bandeau orange (BorderStyle 3 = boite opaque)
        f"Style: Title,Arial Black,{round(fs * .62)},&H000B0A0A,&H002243FF,&H00000000,"
        f"-1,3,14,0,8,90,90,{round(H * .07)}\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lo, hi = segs[0][0], segs[-1][1]
    total = sum(b - a for a, b in segs)
    win = [w for w in words if float(w["end"]) > lo and float(w["start"]) < hi]
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


def render_clip(video, words, clip, workdir, index, iw, ih, use_face=True, opts=None):
    opts = opts or parse_opts("{}")
    slug = slugify(clip["title"])
    name = f"short_{index:02d}_{clip['score']:03d}_{slug}"
    subs = workdir / f"{name}.ass"
    s0, s1 = clip["start"], clip["end"]
    segs = keep_segments(words, s0, s1) if opts["cut"] else [(s0, s1)]
    build_ass(words, segs, subs, opts["format"], clip["title"] if opts["title"] else "")
    W, H, _ = FORMATS[opts["format"]]
    cx, cy = detect_face_center(video, s0, s1) if use_face else (0.5, 0.5)
    cw, ch, x, y = compute_crop(iw, ih, cx, cy, W / H)
    out = workdir / f"{name}.mp4"
    vf = f"crop={cw}:{ch}:{x}:{y},scale={W}:{H},setsar=1,subtitles={subs.name}"
    af = []
    if segs != [(s0, s1)]:  # retrait des blancs : on ne garde que les morceaux parles
        sel = "+".join(f"between(t,{a - s0:.3f},{b - s0:.3f})" for a, b in segs)
        vf = f"select='{sel}',setpts=N/FRAME_RATE/TB," + vf
        af = ["-af", f"aselect='{sel}',asetpts=N/SR/TB"]
    # cwd=workdir pour eviter l'echappement des chemins Windows dans le filtre subtitles
    run(["ffmpeg", "-y", "-ss", str(s0), "-to", str(s1),
         "-i", str(video.resolve()), "-vf", vf, *af, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-c:a", "aac", "-b:a", "128k", str(out.name)], cwd=str(workdir))
    subs.unlink(missing_ok=True)
    # pack de publication, pret a copier-coller
    tags = clip.get("hashtags", "")
    (workdir / f"{name}.txt").write_text(
        f"{clip['title']}\n\n{clip.get('caption', '')}\n\n{tags}\n", encoding="utf-8")
    return out


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
    ap.add_argument("--selfcheck", action="store_true", help="tests internes rapides")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
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

    meta = []
    for i, clip in enumerate(clips, 1):
        out = render_clip(video, words, clip, workdir, i, iw, ih,
                          use_face=not args.no_face, opts=opts)
        meta.append({**clip, "file": out.name})
        print(f"  - {out.name}  (score {clip['score']}/100)  {clip['title']}")
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
    print("selfcheck OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
