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
import shutil
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
from gta_taxonomy import CATEGORY_LABELS, classify_text  # plugin GTA (MIT), meme dossier

# type de video -> consigne ajoutee au choix des moments
CATS = {
    "auto": "",
    "chat": "C'est un live de discussion face camera (Just Chatting) : privilegie les "
            "reactions fortes, les punchlines, les histoires racontees et les moments droles.",
    "irl": "C'est un live IRL dans la rue : privilegie les interactions avec des inconnus, "
           "les situations imprevues et les grosses reactions.",
    "gaming": "C'est un live de jeu video : privilegie la rage, les victoires, les moments "
              "droles et les reactions du streamer (tu ne vois que ce qui est dit).",
    "gta": "C'est une video de GTA (V ou VI) : les grands moments sont les poursuites avec la "
           "police, les crashs et carambolages, les cascades reussies ou ratees, les fusillades, le "
           "chaos des PNJ, les missions et braquages reussis ou rates, les morts absurdes (wasted, "
           "busted) et les bugs, toujours avec la reaction du joueur. Garde environ 8 s de mise en "
           "place avant l'evenement et 10 s de reaction apres ; hook_text = l'evenement ou le cri. "
           "Un recit au passe, un plan pour plus tard ou une hypothese ne sont pas des moments. "
           "Ecarte les cinematiques (dialogues entre personnages du jeu), la lecture du chat, les "
           "menus et les temps calmes. Le titre dit l'action GTA concrete (qui, quoi : 'Police "
           "chase ends in a 5-car pileup'), jamais un mot vague comme 'Shock' ou 'Moment'.",
    "podcast": "C'est un podcast ou une interview : privilegie les idees fortes, les "
               "anecdotes et les avis tranches qui se comprennent seuls.",
}
# mode campagne (Content Rewards) : texte d'appel incruste + credit dans la legende, sans voix off
CAMPAIGNS = {
    # box = cadre fixe de la facecam (x, y, l, h normalises), mesure sur le live du 2026-11-18
    "tomo": {"cta": "SUBSCRIBE TO TOMO ON YOUTUBE @Tomovids", "tags": "#tomovids #gta6 @Tomovids",
             "box": (0.698, 0.426, 0.302, 0.315)},
    "ruckus": {"cta": "WATCH DASGASDOM ON KICK: kick.com/dasgasd3m",
               "tags": "#dasgasdom3 #gta6 kick.com/dasgasd3m"},
}
MAX_RANGE = 3600  # une heure max par passage (quota audio Groq gratuit)
LAYOUTS = ("auto", "full", "split")
ACCENT = "#FF4322"  # orange maison, couleur par defaut du titre
HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
MAX_KEYS = 40
DUR_MIN, DUR_MAX = 40, 90  # bornes du reglage "duree des shorts", en secondes
DMIN_DEF = 65  # TikTok ne paie que les videos de plus d'une minute (video UNBORED, 2026-11-18)
HOOK_MAX = 3.5  # ouverture sur la reaction : longueur max du passage repris en tete du short
SPLIT_TOP = 0.34  # ecran partage : part de la hauteur donnee a la facecam (le jeu prend le reste)
SUB_SPLIT = 0.24  # ecran partage : sous-titres a 24 % du bas (au-dessus de l'interface TikTok/Reels)
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
        "dmin": DMIN_DEF, "dmax": DUR_MAX, "loop": d.get("loop", True) is not False,
        "hook": d.get("hook", True) is not False,
        "theme": d.get("theme") if d.get("theme") in THEMES else "none",
        "bleed": d.get("bleed", True) is not False,
        "voice": d.get("voice") is True,
        "camp": d.get("camp") if d.get("camp") in CAMPAIGNS else "",
        "badge": parse_badge(d.get("badge")), "logo": parse_logo(d.get("logo")),
        "lang": d.get("lang") if d.get("lang") in LANGS else "en",
        "mode": "recap" if d.get("mode") == "recap" else "clips",  # moments forts ou resume continu
    }
    if o["mode"] == "recap":
        o["hook"] = False  # un resume suit l'ordre de la video : pas de reaction recopiee en tete
    if o["camp"]:
        o["voice"] = False  # les campagnes refusent une voix fabriquee
        o["lang"] = "en"  # et sont publiees en anglais
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


def plan_layout(samples, opts, iw, ih, box=None):
    """'split' (facecam en haut, jeu en bas) ou 'full' (plein cadre qui suit le visage)."""
    faces = [smp for smp in samples if smp[1] is not None]
    vertical = opts["format"] in ("9:16", "4:5") and iw > ih
    if not faces or not vertical or opts["layout"] == "full":
        return "full"
    if opts["layout"] == "split":
        return "split"
    # cadre de facecam detecte (ou streamer de campagne) : toujours visage en haut, jeu en bas.
    # Les visages des cinematiques du jeu faussaient le test "visage fixe" (shorts au visage centre).
    if box or opts.get("camp"):
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


def in_box(smp, box):
    return box[0] <= smp[1] <= box[0] + box[2] and box[1] <= smp[2] <= box[1] + box[3]


def facecam_box(video, samples, s0, s1, n=24):
    """Bords du cadre de la facecam incrustee autour du visage median. Ces bords restent fixes
    pendant que le jeu bouge : la moyenne des contours sur n images les fait ressortir, la
    mediane sur la largeur ecarte le texte du chat (hache). (x, y, w, h) normalises ou None."""
    faces = [smp for smp in samples if smp[1] is not None]
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if not faces or s1 <= s0:
        return None
    cap = cv2.VideoCapture(str(video))
    acc_x = acc_y = None
    k = 0
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (s0 + (s1 - s0) * (i + .5) / n) * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(fr, (480, round(480 * fr.shape[0] / fr.shape[1]))),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx, gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)), np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
        acc_x, acc_y = (gx, gy) if acc_x is None else (acc_x + gx, acc_y + gy)
        k += 1
    cap.release()
    if k < 4:
        return None
    H, W = acc_x.shape
    cx = statistics.median(f[1] for f in faces) * W
    cy = statistics.median(f[2] for f in faces) * H
    fw = statistics.median(f[3] for f in faces) * W

    def edge(prof, a, b, default):
        a, b = int(max(0, min(a, b))), int(min(len(prof), max(a, b)))
        if b - a < 2:
            return default
        i = int(prof[a:b].argmax())
        return a + i if prof[a + i] > 4 * np.median(prof) else default
    near, nv = 0.8 * fw, 0.6 * fw  # le cadre colle souvent au menton
    col = np.median(acc_x[int(max(0, cy - fw)):int(min(H, cy + fw))], axis=0)
    left, right = edge(col, 3, cx - near, 0), edge(col, cx + near, W - 3, W)
    row = np.median(acc_y[:, int(left) + 2:max(int(left) + 3, int(right) - 2)], axis=1)
    top, bot = edge(row, 3, cy - nv, 0), edge(row, cy + nv, H - 3, H)
    if (right - left) < 2 * fw or (bot - top) < 1.5 * fw or (right - left) * (bot - top) > .5 * W * H:
        return None  # pas un cadre de facecam plausible
    return left / W, top / H, (right - left) / W, (bot - top) / H


def game_crop(iw, ih, ratio, box=None):
    """Cadre du jeu (panneau du bas), centre, decale pour ne pas mordre sur la facecam."""
    cw, ch, x, y = compute_crop(iw, ih, 0.5, 0.5, ratio)
    if box and box[3] > .2:
        bl, br = box[0] * iw, (box[0] + box[2]) * iw
        if x + cw > bl and x < br:  # chevauche le cadre de la camera
            if bl - cw >= 0 and (bl + br) / 2 > iw / 2:
                x = int(bl - cw) // 2 * 2
            elif br + cw <= iw:
                x = int(br + 1) // 2 * 2
    return cw, ch, x, y


def facecam_crop(samples, iw, ih, ratio, box=None):
    """Cadre autour de la facecam, au ratio du panneau du haut. Avec `box` (cadre detecte),
    le plus grand cadre tenant dedans, centre sur le visage : rien du jeu ne deborde."""
    faces = [smp for smp in samples if smp[1] is not None]
    if box and faces:
        # rentre de 1,5 % : la mesure des bords se fait au pixel pres sur une image reduite
        m = .015 * ih
        bx, by, bw_, bh_ = box[0] * iw + m, box[1] * ih + 2.5 * m, box[2] * iw - 2 * m, box[3] * ih - 3.5 * m  # le chat deborde souvent sur le haut
        cx = statistics.median(f[1] for f in faces) * iw
        cy = statistics.median(f[2] for f in faces) * ih
        w, h = (bh_ * ratio, bh_) if bw_ / bh_ > ratio else (bw_, bw_ / ratio)
        w, h = int(w) // 2 * 2, int(h) // 2 * 2
        x = int(min(max(cx - w / 2, bx), bx + bw_ - w)) + 1
        y = int(min(max(cy - h / 2, by), by + bh_ - h)) + 1
        return w, h, min(x, iw - w), min(y, ih - h)
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


def groq_chat(messages, key, json_mode=True, extra=None):
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.4, **(extra or {})}
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
    words, heard = [], ""
    if size <= AUDIO_CHUNK_BYTES:
        resp = _transcribe_one(audio.read_bytes(), key)
        words, heard = resp.get("words") or [], resp.get("language") or ""
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
            heard = heard or resp.get("language") or ""
            for w in resp.get("words") or []:
                w["start"] = float(w.get("start", 0)) + start
                w["end"] = float(w.get("end", 0)) + start
                words.append(w)
    words = [w for w in words if w.get("word") and w.get("end") is not None]
    if not words:
        sys.exit("[erreur] transcription vide (pas de paroles detectees ?)")
    return words, spoken_lang(heard, words)


def spoken_lang(heard, words):
    """Langue entendue par Whisper ("French" ou "fr") -> code ; "" si inconnue."""
    h = str(heard).strip().lower()
    code = WHISPER_LANGS.get(h, h if len(h) == 2 else "")
    return code or ("en" if is_english(words) else "")


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
CHUNK_CHARS = 11000
PER_CALL = 6  # extraits demandes au LLM par appel  # ~3500 jetons de transcription par appel
# Langue des shorts (2026-11-24, demande de Jacques : l'outil n'est pas que pour l'anglais) :
# titres, legendes, hashtags ET sous-titres dans la langue choisie, anglais par defaut. Si la video
# parle une autre langue, les sous-titres sont traduits. Pour ajouter une langue : une ligne ici,
# une dans LANGS de site/api/_gh.mjs et une option dans le select #lang du site.
# Consigne en anglais : en francais, le modele repondait parfois en francais.
LANGS = {"en": ("ENGLISH", "Part"), "fr": ("FRENCH", "Partie"), "es": ("SPANISH", "Parte")}
WHISPER_LANGS = {"english": "en", "french": "fr", "spanish": "es"}


def lang_rule(lang="en"):
    name = LANGS.get(lang, LANGS["en"])[0]
    return (f"IMPORTANT: write every title, caption and hashtag in {name}, whatever the language "
            f"of the video. Never answer in another language.")


ENGLISH = lang_rule("en")


def chunk_lines(text, size=CHUNK_CHARS):
    chunks, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) > size:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    return chunks + [cur] if cur else chunks


def chat_retry(messages, key, tries=4, json_mode=True, extra=None):
    """groq_chat qui attend la fin de la minute quand le quota par minute est atteint, et
    reessaie quand Groq coupe la connexion ou tombe (5xx)."""
    for i in range(tries):
        try:
            return groq_chat(messages, key, json_mode, extra)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or i == tries - 1:
                raise
            wait = min(float(e.headers.get("retry-after") or 30), 90) + 1 if e.code == 429 else 5
            print(f"[info] Groq {e.code}, pause de {wait:.0f} s")
            time.sleep(wait)
        except OSError as e:  # connexion coupee sans reponse (RemoteDisconnected), delai depasse
            if i == tries - 1:
                raise urllib.error.URLError(e)
            print(f"[info] connexion a Groq coupee ({type(e).__name__}), nouvel essai")
            time.sleep(5)


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
    allw = words
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
        sc += 1.0 if gap > .5 else 0  # fin sur une pause : la chute est dite, pas une idee coupee
        if opts["loop"] and nxt and first and norm_word(nxt["word"]) == first:
            sc += 3  # la phrase d'ouverture revient : la derniere image s'enchaine sur la premiere
        if best_sc is None or sc > best_sc:
            best, best_sc = (e, nxt), sc
    if best is None:  # pas assez de parole : on garde ce qui rentre, au plus pres de dmax
        last = [w for w in ws if kept(words, start, float(w["end"]) + .08, cut) <= dmax]
        best = (float((last or ws)[-1]["end"]), None)
        start = pull_start(allw, start, best[0] + .08, opts)
    e, nxt = best
    end = e + .08 if not nxt else min(e + .08, float(nxt["start"]))
    return {**clip, "start": round(start, 2), "end": round(end, 2)}


def pull_start(words, start, end, opts):
    """Pas assez de parole apres le debut (fin de la video, longs silences) : le debut recule de
    phrase en phrase jusqu'a la duree mini, sans depasser la duree maxi (2026-11-24 : un short de
    33 s pour 40 s demandees, l'extrait commencait 36 s avant la fin de la video)."""
    dmin, dmax, cut = opts["dmin"], opts["dmax"], opts["cut"]
    if kept(words, start, end, cut) >= dmin:
        return start
    before = [w for w in words if float(w["end"]) <= start + .1]
    best = start
    for k in range(len(before) - 1, -1, -1):
        s = max(0.0, float(before[k]["start"]) - .05)
        d = kept(words, s, end, cut)
        if d > dmax:
            break
        head = k == 0 or SENT_END.search(before[k - 1]["word"].strip()) or             float(before[k]["start"]) - float(before[k - 1]["end"]) > .5  # debut de phrase
        if head:
            best = s
            if d >= dmin:
                break
    return best


def select_clips(words, key, n, cat="auto", find="", opts=None):
    opts = opts or parse_opts("{}")
    print("[3/4] choix des meilleurs moments (LLM Groq)...")
    total = int(float(words[-1]["end"]))
    text = transcript_lines(words)
    # au-dela de 6 extraits par reponse, le modele rend un JSON invalide (erreur 400
    # json_validate_failed, constate avec 10 le 2026-11-19) : assez de tranches pour 6 chacune
    k = min(n, PER_CALL)
    chunks = chunk_lines(text, min(CHUNK_CHARS, len(text) // math.ceil(n / PER_CALL) + 1))
    found = []
    for i, part in enumerate(chunks):
        if len(chunks) > 1:
            print(f"    tranche {i + 1}/{len(chunks)}")
        for ask in (k, max(2, k // 2)):  # reponse invalide : on redemande moins d'extraits
            try:
                found += select_in(part, total, key, ask, cat, find, len(chunks) > 1, opts)
                break
            except urllib.error.HTTPError as e:
                print(f"[info] LLM indisponible sur cette tranche ({e}), essai avec moins d'extraits")
            except (urllib.error.URLError, KeyError) as e:
                print(f"[info] LLM indisponible sur cette tranche ({e})")
                break
    if not found:
        print("[info] aucun extrait propose par le LLM, decoupage regulier en secours.")
        found = fallback_clips(total, n, opts["dmax"])
    return best_distinct([fit_clip(c, words, opts) for c in found], n)


def gta_hints(transcript, limit=30):
    """Lignes de la transcription ou le plugin GTA repere un evenement (poursuite, crash...)."""
    found = []
    for m in re.finditer(r"^\[[\d:]+\] \((\d+)s\) (.*)$", transcript, re.M):
        for cue in classify_text(m.group(2)):
            found.append((cue.strength, int(m.group(1)), CATEGORY_LABELS[cue.category], cue.phrase))
    best = sorted(found, reverse=True)[:limit]
    return "\n".join(f'- {t}s {lab} : "{ph}"' for _, t, lab, ph in sorted(best, key=lambda x: x[1]))


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
        '\"hashtags\":[\"<4 a 6 hashtags pertinents>\"],'
        '\"hook_text\":\"<2 a 8 mots recopies tels quels de la transcription, au moment le plus fort '
        'de l extrait : le cri, l echec, la reaction>\"}]}\n'
        f"Regles : chaque extrait dure entre {opts['dmin']} et {opts['dmax']} secondes, start et end sont des "
        "nombres en secondes pris dans la video, les extraits ne se chevauchent pas, "
        "tries du meilleur score au moins bon. Sans aucun emoji. Rien d'autre que le JSON.\n"
        "Ce qui fait un bon short (a respecter) :\n"
        "- il commence sur la reaction, pas sur la mise en place : le premier mot entendu "
        "doit deja accrocher (cri, echec, punchline), l'histoire se rattrape ensuite ;\n"
        "- il se termine sur la chute, jamais au milieu d'une idee ;\n"
        "- il se comprend sans avoir vu le live ni connaitre le contexte ;\n"
        "- le score mesure la chance qu'on le regarde jusqu'au bout : un debut lent, une fin "
        "coupee ou une blague interne font baisser le score.\n"
        + lang_rule(opts["lang"]))
    if opts["loop"]:
        user_msg += ("\n\nLe short sera lu en boucle : apres la chute, si la premiere phrase "
                     "de l'extrait revient juste apres dans la video, arrete-toi juste avant.")
    if CATS.get(cat):
        user_msg += "\n\nContexte : " + CATS[cat]
    hints = gta_hints(transcript) if cat in ("gta", "gaming") else ""
    if hints:
        user_msg += ("\n\nEvenements GTA reperes dans ce qui est dit (indices, a confirmer avec la "
                     "transcription) :\n" + hints)
    if find:
        user_msg += ("\n\nLe spectateur cherche un moment precis : \u00ab " + find + " \u00bb. "
                     "Ne retiens que des extraits qui y correspondent, du plus probable au "
                     "moins probable (le score dit a quel point ca correspond). S'il n'y a "
                     "pas de correspondance exacte, donne les passages les plus proches.\n"
                     + lang_rule(opts["lang"]))
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
        tags = tags_str(c.get("hashtags"))
        out.append({
            "start": start, "end": end,
            "title": str(c.get("title", "Short"))[:80],
            "score": int(c.get("score", 0) or 0),
            "reason": str(c.get("reason", ""))[:200],
            "caption": str(c.get("caption", ""))[:400],
            "hashtags": tags,
            "hook_text": str(c.get("hook_text", ""))[:120],
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
              subs=True, color=ACCENT, subc="#FFFFFF", font="Arial Black", accent=None, cta="", top=0):
    """Sous-titres ASS : groupes de mots gros et centres, titre en haut si demande.
    `segs` = morceaux gardes de la source ; les temps sont recales dessus.
    `accent` : le mot le plus long de chaque groupe prend cette couleur (l'oeil s'y accroche)."""
    W, H, fs = FORMATS[fmt]
    # top = hauteur du panneau facecam (ecran partage) : titre et appel passent juste dessous
    t_mv = top + 24 if top else round(H * .07)
    c_mv = (t_mv + round(fs * .62 * 1.3) + 40 if title else t_mv) if top else round(H * .125)
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
        f"-1,3,14,0,8,90,90,{t_mv}\n"
        # appel a l'action des campagnes : petit bandeau sombre sous le titre (le bas de l'ecran
        # est mange par l'interface de TikTok et Reels)
        f"Style: Cta,Arial Black,{round(fs * .4)},&H00FFFFFF,&H50000000,&H00000000,"
        f"-1,3,10,0,8,60,60,{c_mv}\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lo, hi = segs[0][0], segs[-1][1]
    total = sum(b - a for a, b in segs)
    win = [w for w in words if float(w["end"]) > lo and float(w["start"]) < hi] if subs else []
    events = []
    if title:
        events.append(f"Dialogue: 1,{ass_time(0)},{ass_time(total)},Title,,0,0,0,,{ass_text(title)}")
    if cta:
        events.append(f"Dialogue: 1,{ass_time(0)},{ass_time(total)},Cta,,0,0,0,,{ass_text(cta)}")
    # mot par mot : le groupe reste affiche, le mot prononce s'allume a l'instant ou il est dit
    hl = accent or "#FFD60A"
    if hl.upper() == subc.upper():
        hl = ACCENT
    on, off = f"{{\\c{ass_color(hl)}\\fscx110\\fscy110}}", f"{{\\c{ass_color(subc)}\\fscx100\\fscy100}}"
    for i in range(0, len(win), group):
        chunk = win[i:i + group]
        ws = [ass_text(w["word"].strip().upper()) for w in chunk]
        t = [remap(float(w["start"]), segs) for w in chunk]
        end = min(total, max(t[0] + 0.3, remap(float(chunk[-1]["end"]), segs)))
        a = t[0]
        for j in range(len(chunk)):
            b = end if j == len(chunk) - 1 else max(a, min(t[j + 1], end))  # horodatages parfois en desordre
            if b - a < .02:
                continue
            text = " ".join(on + x + off if k == j else x for k, x in enumerate(ws))
            events.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Pop,,0,0,0,,{text}")
            a = b
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
        "auto": d.get("auto") is True,  # editeur : "refais le cadrage tout seul"
        "theme": d.get("theme") if d.get("theme") in THEMES else "none",
        "bleed": d.get("bleed", True) is not False,
        "voice": d.get("voice") is True,
        "cta": ass_text(str(d.get("cta") or ""))[:80],
        "badge": parse_badge(d.get("badge")), "logo": parse_logo(d.get("logo")),
        "lang": d.get("lang") if d.get("lang") in LANGS else "en",
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
            # mise en page de ce plan ; anciennes cles sans elle : deduite des cadres valides
            lay = k.get("layout") if k.get("layout") in ("full", "split") else                 "full" if r["crop"] and not (r["cam"] or r["game"]) else "split"
            out[t] = {"t": t, "layout": lay, **{n: v for n, v in r.items() if v}}
    return [out[t] for t in sorted(out)][:MAX_KEYS]


def plans(spec):
    """Decoupe le short en plans [(debut, fin, cadres)] en temps du short (0 = debut de coupe)."""
    s0, s1 = spec["trim"]
    cur = {n: spec[n] for n in ("crop", "cam", "game", "layout")}
    out = [[s0, cur]]
    for k in spec.get("keys") or []:
        r = {**out[-1][1], **{n: k[n] for n in ("crop", "cam", "game", "layout") if k.get(n)}}
        if k["t"] <= s0 + .05:
            out[-1][1] = r
        elif k["t"] < s1 - .05:
            out.append([k["t"], r])
    return [(round(a - s0, 3), round((out[i + 1][0] if i + 1 < len(out) else s1) - s0, 3), r)
            for i, (a, r) in enumerate(out)]


def auto_spec(samples, opts, iw, ih, s0, s1, title, tags="", box=None):
    """Reglages de depart, habillage compris (theme choisi a la generation)."""
    sp = base_spec(samples, opts, iw, ih, s0, s1, title, box)
    sp.update(theme=opts.get("theme", "none"), bleed=opts.get("bleed", True), kicker=gta_kicker(), lab1="", lab2="",
              credit="", ticker=" ".join(tags.split()[:8]) or title)
    if sp["theme"] == "versus" and iw > ih:
        sp["layout"] = "split"  # deux plans differents : le visage en haut, l'action en bas
    wins = theme_windows(sp)
    if wins:  # cadres a la forme des fenetres du theme, le plus large possible, centres sur le visage
        faces = [smp for smp in samples if smp[1] is not None]
        cx = statistics.median(f[1] for f in faces) if faces else 0.5
        cy = statistics.median(f[2] for f in faces) if faces else 0.5
        r0 = wins[0][2] / wins[0][3]
        sp["crop"] = norm_rect(compute_crop(iw, ih, cx, cy, r0), iw, ih)
        if len(wins) == 2:
            r1 = wins[1][2] / wins[1][3]
            if sp["theme"] == "versus":
                cw, ch, _, _ = compute_crop(iw, ih, cx, cy, r0)
                zw = min(cw, faces and statistics.median(f[3] for f in faces) * iw * 4 or cw * .55)
                cam = rect_px({"x": cx - zw / iw / 2, "y": cy - zw / r0 / ih / 2, "w": zw / iw, "h": zw / r0 / ih},
                              iw, ih, r0)
                sp["cam"] = norm_rect(cam, iw, ih)
                sp["game"] = norm_rect(compute_crop(iw, ih, cx, cy, r1), iw, ih)
            else:  # vraie facecam dans un coin : on la garde, le jeu prend le reste
                sp["cam"] = norm_rect(facecam_crop(samples, iw, ih, r0, box), iw, ih) if faces else sp["cam"]
                sp["game"] = norm_rect(game_crop(iw, ih, r1, box), iw, ih)
    return sp


def base_spec(samples, opts, iw, ih, s0, s1, title, box=None):
    """Reglages de depart d'un short : ce que le moteur choisit seul, modifiable ensuite."""
    W, H, _ = FORMATS[opts["format"]]
    ht, hb = split_heights(H)
    faces = [smp for smp in samples if smp[1] is not None]
    cx = statistics.median(f[1] for f in faces) if faces else 0.5
    cy = statistics.median(f[2] for f in faces) if faces else 0.5
    cam = facecam_crop(samples, iw, ih, W / ht, box) if faces else \
        compute_crop(iw, ih, 0.12, 0.15, W / ht)
    return {
        "format": opts["format"], "layout": plan_layout(samples, opts, iw, ih, box),
        "crop": norm_rect(compute_crop(iw, ih, cx, cy, W / H), iw, ih),
        "cam": norm_rect(cam, iw, ih),
        "game": norm_rect(game_crop(iw, ih, W / hb, box), iw, ih),
        "track": bool(faces), "trim": [round(s0, 2), round(s1, 2)],
        "cut": opts["cut"], "subs": True, "title": title, "ask": "",
        "color": ACCENT, "subc": "#FFFFFF", "keys": [], "voice": opts.get("voice", False),
        "cta": CAMPAIGNS[opts["camp"]]["cta"] if opts.get("camp") else "",
        "badge": opts.get("badge"), "logo": opts.get("logo"), "lang": opts.get("lang", "en"),
    }


# ----------------------------------------------------------------------------- realisateur auto
# Refait ce que fait un monteur a la main : a chaque plan, ou est ce qui interesse ?
# Un seul centre d'interet -> plein cadre cale dessus. Deux centres trop eloignes pour
# tenir ensemble dans le 9:16 -> ecran partage, un panneau sur chacun.

def win_ratios(spec, iw, ih):
    """Ratio largeur/hauteur de chaque cadre : crop (plein cadre), cam (haut), game (bas)."""
    W, H, _ = FORMATS[spec["format"]]
    ht, hb = split_heights(H)
    out = {"crop": W / H, "cam": W / ht, "game": W / hb}
    th = THEMES.get(spec.get("theme"))
    if th and spec["format"] == "9:16":
        b = "b" if spec.get("bleed", True) else ""
        f, sp = th[b + "full"], th[b + "split"]
        out = {"crop": f[0][2] / f[0][3], "cam": sp[0][2] / sp[0][3], "game": sp[-1][2] / sp[-1][3]}
    return out


def _saliency(gray):
    """Residu spectral (Hou 2007) : ce qui sort du lot dans l'image, sans reseau."""
    import cv2
    import numpy as np
    g = cv2.resize(gray, (64, 64)).astype(np.float32)
    f = np.fft.fft2(g)
    log = np.log(np.abs(f) + 1e-6)
    sal = np.abs(np.fft.ifft2(np.exp(log - cv2.blur(log, (3, 3)) + 1j * np.angle(f)))) ** 2
    return cv2.resize(cv2.GaussianBlur(sal.astype(np.float32), (9, 9), 2.5), gray.shape[::-1])


def _window(prof, w):
    """Fenetre de largeur w (en colonnes) qui capte le plus d'interet : (debut, masse)."""
    import numpy as np
    c = np.concatenate([[0], np.cumsum(prof)])
    sums = c[w:] - c[:-w]
    i = int(np.argmax(sums))
    return i, float(sums[i])


def director(raw, s0, s1, iw, ih, spec, step=0.25, chunk=3.0):
    """Plans auto [(t, layout, {crop} ou {cam, game})] sur [s0, s1], [] si OpenCV manque."""
    try:
        import cv2
        import numpy as np
        find = _face_finder()
    except Exception as e:
        print(f"[info] realisateur auto indisponible ({e})")
        return []
    cap = cv2.VideoCapture(str(raw))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, round(fps * step))
    cap.set(cv2.CAP_PROP_POS_MSEC, s0 * 1000)
    frames = []  # (t, gris 320 px, histogramme, image 960 px)
    for i in range(int((s1 - s0) * fps)):
        if i % every:
            if not cap.grab():
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        big = cv2.resize(img, (960, round(960 * img.shape[0] / img.shape[1])))
        small = cv2.resize(big, (320, round(320 * img.shape[0] / img.shape[1])))
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
        frames.append((s0 + i / fps, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32),
                       cv2.normalize(hist, hist).flatten(), big))
    cap.release()
    if len(frames) < 4:
        return []
    h, w = frames[0][1].shape
    # interface du jeu (minimap, etoiles, logo) : ne bouge jamais d'un bout a l'autre
    hud = (np.std([f[1] for f in frames], axis=0) < 6).astype(np.float32)
    hud = cv2.dilate(hud, np.ones((5, 5), np.uint8))
    # changements de plan : la palette de couleurs saute d'une image a l'autre
    bounds = [0] + [i for i in range(1, len(frames))
                    if cv2.compareHist(frames[i - 1][2], frames[i][2], cv2.HISTCMP_CORREL) < 0.55]
    bounds.append(len(frames))
    groups = []
    for a, b in zip(bounds, bounds[1:]):
        n = max(1, round((b - a) * step / chunk))
        groups += [(a + (b - a) * j // n, a + (b - a) * (j + 1) // n, j == 0) for j in range(n)]
    rat = win_ratios(spec, iw, ih)
    gy, gx = np.mgrid[0:h, 0:w]
    centre = 0.6 + 0.4 * np.exp(-(((gx / w - .5) / .35) ** 2 + ((gy / h - .5) / .45) ** 2))
    two_ok = iw > ih and spec["format"] in ("9:16", "4:5")
    out = []
    for a, b, cut in groups:
        fs = frames[a:b]
        mo, mw = np.zeros((h, w), np.float32), 0.45
        if len(fs) > 1:
            mo = np.mean([np.abs(x[1] - y[1]) for x, y in zip(fs, fs[1:])], axis=0)
            if (mo > 8).mean() > 0.25:  # camera qui roule : tout le decor bouge, ca ne dit rien
                mw = 0.07
            mo = np.clip(mo - np.median(mo) * 1.5, 0, None)  # on retire le mouvement de camera
        mid = fs[len(fs) // 2]
        face = np.zeros((h, w), np.float32)
        k = w / mid[3].shape[1]
        for fx, fy, fw, fh in find(mid[3]):
            cv2.circle(face, (int((fx + fw / 2) * k), int((fy + fh / 2) * k)), max(4, int(fw * k)), 1.0, -1)
        maps = [m / (m.sum() + 1e-6) for m in (mo, _saliency(mid[1]), face)]
        imp = (mw * maps[0] + 0.45 * maps[1] + 0.8 * maps[2]) * (1 - 0.85 * hud) * centre
        imp = cv2.GaussianBlur(imp.astype(np.float32), (0, 0), 3)
        prof = imp.sum(axis=0)
        tot = float(prof.sum()) + 1e-9
        fw_ = max(2, min(w, round(h * rat["crop"])))  # largeur du plein cadre, en colonnes
        x1, m1 = _window(prof, fw_)
        # deux centres : le plus fort, puis le plus fort une fois le premier efface
        nw = max(2, round(fw_ * 0.7))
        xa, ma = _window(prof, nw)
        rest = prof.copy()
        rest[max(0, xa - nw // 3):xa + nw + nw // 3] = 0
        xb, mb = _window(rest, nw)
        far = abs(xa - xb) > fw_ * 0.9
        # partage : deux vrais pics (pas un decor uniforme), le 2e presque aussi fort que le 1er,
        # trop loin l'un de l'autre pour tenir ensemble dans le plein cadre
        # image sombre ou terne (std < 28) : le 2e "centre" n'est que du bruit
        two = two_ok and far and mb > 0.55 * ma and (ma + mb) / tot > 2.6 * nw / w and mid[1].std() > 28

        def box(x0, ww, ratio):
            """Cadre au ratio voulu sur la bande [x0, x0+ww], centre en hauteur sur l'interet."""
            col = imp[:, x0:x0 + ww].sum(axis=1)
            cy = float((col * np.arange(h)).sum() / (col.sum() + 1e-9)) / h
            bw = max(ww * 1.6 / w, 0.3)  # assez large pour garder le contexte et la nettete
            bh = bw * iw / ih / ratio
            if bh > 1:
                bh, bw = 1.0, ratio * ih / iw
            return norm_rect(rect_px({"x": (x0 + ww / 2) / w - bw / 2, "y": cy - bh / 2, "w": bw, "h": bh},
                                     iw, ih, ratio), iw, ih)
        if two:
            if out and out[-1][1] == "split" and abs(xb / w - out[-1][2]["game"]["x"]) < abs(xa / w - out[-1][2]["game"]["x"]):
                xa, xb = xb, xa  # chaque sujet garde son panneau d'un plan a l'autre
            r = {"game": box(xa, nw, rat["game"]), "cam": box(xb, nw, rat["cam"])}
            g, c = r["game"], r["cam"]
            two = min(g["x"] + g["w"], c["x"] + c["w"]) - max(g["x"], c["x"]) < 0.25 * min(g["w"], c["w"])
        if two:  # panneaux qui se recouvrent : on verrait deux fois la meme chose
            out.append((fs[0][0], "split", r, cut))
        else:
            cx = (x1 + fw_ / 2) / w
            out.append((fs[0][0], "full", {"crop": norm_rect(compute_crop(iw, ih, cx, .5, rat["crop"]), iw, ih)}, cut))
    # on ne garde que les vrais changements : autre mise en page, ou cadre deplace
    keep = []
    for t, lay, r, cut in out:
        if keep:
            pt, pl, pr = keep[-1]
            moved = any(abs(r[n]["x"] - pr[n]["x"]) + abs(r[n]["y"] - pr[n]["y"]) > 0.08 for n in r if n in pr)
            if (lay == pl and not moved) or t - pt < (1.0 if cut else 2.0):
                continue
        keep.append((round(t, 2), lay, r))
    return keep[:MAX_KEYS + 1]


def apply_director(spec, cuts):
    """Plans du realisateur -> base du short + cadrages "a partir d'ici"."""
    if not cuts:
        return spec
    _, lay, r = cuts[0]
    return {**spec, "layout": lay, "track": False, **r,
            "keys": [{"t": t, "layout": l, **rr} for t, l, rr in cuts[1:]]}


# ----------------------------------------------------------------------------- themes GTA
# Habillages 9:16 inspires de l'univers GTA (couleurs de Vice City, etoiles de recherche,
# ecrans de chargement). Aucun logo ni visuel de Rockstar : seulement l'esprit.
# Fenetres = (x, y, largeur, hauteur, angle en degres) en pixels 1080x1920. "full" = une
# fenetre (cadre unique), "split" = deux (facecam/plan 1 puis jeu/plan 2).
# Le site garde une copie de ces fenetres (THEMES dans index.html) : les deux doivent suivre.
FONTS = pathlib.Path(__file__).parent / "assets" / "fonts"
PINK, ORANGE, CYAN, GOLD, PLUM = "#FF2E88", "#FF8A3D", "#2EE6F0", "#FFC83D", "#16071F"
# Modes : full/split = video dans des fenetres sur un decor ; bfull/bsplit = plein ecran,
# la video couvre tout le 9:16 et le decor se pose par-dessus.
BLEED = {"bfull": [(0, 0, 1080, 1920, 0)], "bsplit": [(0, 0, 1080, 768, 0), (0, 768, 1080, 1152, 0)]}
HALVES = [(0, 0, 1080, 960, 0), (0, 960, 1080, 960, 0)]
THEMES = {
    "neon": {"full": [(40, 360, 1000, 1240, 0)], "split": [(40, 340, 1000, 500, 0), (40, 860, 1000, 760, 0)],
             **BLEED, "sub": {"full": 1460, "split": 1480, "bfull": 1560, "bsplit": 1560}, "accent": PINK},
    "wanted": {"full": BLEED["bfull"], "split": BLEED["bsplit"], **BLEED,
               "sub": {"full": 1560, "split": 1560, "bfull": 1560, "bsplit": 1560}, "accent": GOLD},
    "loading": {"full": [(60, 420, 960, 1270, -1.5)],
                "split": [(80, 420, 920, 500, -2), (50, 960, 980, 740, 1.5)], **BLEED,
                "sub": {"full": 1540, "split": 1560, "bfull": 1560, "bsplit": 1560}, "accent": GOLD},
    "versus": {"full": [(0, 260, 1080, 820, 0), (0, 1090, 1080, 820, 0)],
               "split": [(0, 260, 1080, 820, 0), (0, 1090, 1080, 820, 0)], "bfull": HALVES, "bsplit": HALVES,
               "sub": {"full": 1700, "split": 1700, "bfull": 1700, "bsplit": 1700}, "accent": GOLD},
    "news": {"full": [(0, 250, 1080, 1080, 0)], "split": [(0, 230, 1080, 420, 0), (0, 650, 1080, 760, 0)],
             **BLEED, "sub": {"full": 1200, "split": 1290, "bfull": 1300, "bsplit": 1300}, "accent": "#FFD23F"},
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


def theme_mode(spec):
    """full, split, bfull ou bsplit (b = plein ecran)."""
    two = spec["theme"] == "versus" or (spec["layout"] == "split" and spec["cam"] and spec["game"])
    return ("b" if spec.get("bleed", True) else "") + ("split" if two else "full")


def theme_windows(spec):
    """Fenetres du theme pour ce short, ou None si pas de theme (ou format autre que 9:16)."""
    th = THEMES.get(spec.get("theme"))
    if not th or spec["format"] != "9:16":
        return None
    return th[theme_mode(spec)]


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
    bleed = wins[0][:3] == (0, 0, 1080)  # plein ecran : tout le decor passe par-dessus la video
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

    def shade(top_h, bottom_y, a_top=190, a_bot=200):
        """Haut et bas assombris en fondu : le texte reste lisible sur n'importe quel plan."""
        m = Image.new("L", (1, H))
        for yy in range(H):
            a = max(0.0, 1 - yy / top_h) * a_top + max(0.0, (yy - bottom_y) / (H - bottom_y)) * a_bot
            m.putpixel((0, yy), int(min(a, 215)))
        fg.paste(Image.new("RGBA", (W, H), (0, 0, 0, 255)), (0, 0), m.resize((W, H)))

    def gutter(y):
        """Jointure entre les deux plans en plein ecran."""
        d_fg.rectangle((0, y - 7, W, y + 7), fill=(10, 10, 12, 255))

    # La video d'abord : titre de 2 lignes au plus, tenu dans le haut de l'ecran. En ecran
    # partage plein ecran, il se pose sur la jointure (bas de la facecam, haut du jeu) pour
    # ne cacher ni le visage ni l'action.
    seam = wins[1][1] if bleed and two else None

    def title_lines(size, smallest, maxw=900):
        f, lines = _fit(title, "Anton-Regular.ttf", maxw, 2, size, smallest)
        return f, lines, f.size * 1.08

    if name == "neon":
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if bleed:  # cadre neon sur le bord de l'ecran, halo par-dessus l'image
            shade(380, 1500, 150, 170)
            rects = [(34, 34, W - 68, H - 68, 0)]
            if two:
                gutter(wins[1][1])
                d_fg.rectangle((34, wins[1][1] - 3, W - 34, wins[1][1] + 3), fill=_rgb(PINK))
        else:
            d_bg.rectangle((0, 0, W, H), fill=_rgb(PLUM, 175))
            rects = wins
        strokes = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for (x, y, w, h, _), (c1, c2) in zip(rects, ((PINK, CYAN), (CYAN, PINK))):
            frame = _grad(w + 40, h + 40, c1, c2)
            m = Image.new("L", (w + 40, h + 40), 0)
            ImageDraw.Draw(m).rectangle((14, 14, w + 26, h + 26), outline=255, width=12)
            glow.paste(frame, (x - 20, y - 20), m)
            m2 = Image.new("L", (w + 40, h + 40), 0)
            ImageDraw.Draw(m2).rectangle((15, 15, w + 25, h + 25), outline=255, width=6)
            strokes.paste(frame, (x - 20, y - 20), m2)
        under = fg if bleed else bg  # en plein ecran le halo deborde sur l'image
        under.alpha_composite(glow.filter(ImageFilter.GaussianBlur(22)))
        under.alpha_composite(glow.filter(ImageFilter.GaussianBlur(6)))
        fg.alpha_composite(strokes)
        if kicker:
            _pill(fg, W / 2, 118, kicker, _font("Anton-Regular.ttf", 34), _rgb(PINK), (255, 255, 255, 255),
                  h=58)
        if title:
            f, lines, lh = title_lines(80, 56)
            y0 = (seam or 255) - lh * (len(lines) - 1) / 2
            for i, ln in enumerate(lines):
                if i == len(lines) - 1:
                    _grad_text(fg, (W / 2, y0 + i * lh), ln, f, PINK, ORANGE)
                else:
                    _text(fg, (W / 2, y0 + i * lh), ln, f, (255, 255, 255, 255), shadow=10)
        x, y, w, h, _ = (70, H - 210, W - 140, 64, 0) if bleed else wins[-1]
        progress(x, y + h + 26, w, 10, PINK, CYAN)
        if spec.get("credit"):
            _text(fg, (W / 2, y + h + 70), spec["credit"].upper(), _font("BarlowCondensed-Bold.ttf", 32),
                  (255, 255, 255, 170))

    elif name == "wanted":
        shade(400, 1500, 150, 170)
        stars = [(W - 80 - i * 68, 96) for i in range(5)][::-1]
        for cx, cy in stars:
            d_fg.polygon(_star(cx, cy, 29), fill=(0, 0, 0, 120), outline=(255, 255, 255, 230), width=3)
        if kicker:
            _text(fg, (60, 96), kicker, _font("Anton-Regular.ttf", 38), (255, 255, 255, 255), anchor="lm",
                  stroke=4)
        if title:
            f, lines, lh = title_lines(84, 58, 960)
            y0 = (seam or 250) - lh * (len(lines) - 1) / 2
            for i, ln in enumerate(lines):
                last = i == len(lines) - 1
                _text(fg, (W / 2, y0 + i * lh), ln, f, _rgb(GOLD) if last else (255, 255, 255, 255),
                      stroke=7, shadow=12)
        # etoiles qui s'allument au fil du clip : on reste pour voir la 5e
        for i, (cx, cy) in enumerate(stars):
            s = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
            sd = ImageDraw.Draw(s)
            sd.polygon(_star(48, 48, 33), fill=(0, 0, 0, 255))
            sd.polygon(_star(48, 48, 28), fill=_rgb(GOLD))
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

    elif name == "loading" and bleed:
        # plein ecran : la video devient la case de BD, bordure d'affiche et coins peints
        shade(420, 1500, 130, 150)
        corners = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dc = ImageDraw.Draw(corners)
        for pts, c in ((((0, 0), (190, 0), (0, 190)), "#FF4F9A"), (((W, H), (W - 220, H), (W, H - 220)), "#1EB7C2")):
            dc.polygon(pts, fill=_rgb(c))
        for yy in range(0, H, 20):  # trame d'imprimerie qui s'efface vers le centre
            for xx in range(0, W, 20):
                k = max(1 - (xx + yy) / 280, 1 - ((W - xx) + (H - yy)) / 300)
                if k > 0:
                    r = 1.5 + 5 * k
                    dc.ellipse((xx - r, yy - r, xx + r, yy + r), fill=(10, 10, 12, 120))
        fg.alpha_composite(corners)
        if two:
            gutter(wins[1][1])
            d_fg.rectangle((0, wins[1][1] - 3, W, wins[1][1] + 3), fill=(255, 255, 255, 255))
        d_fg.rectangle((0, 0, W - 1, H - 1), outline=(10, 10, 12, 255), width=30)
        d_fg.rectangle((30, 30, W - 31, H - 31), outline=(255, 255, 255, 255), width=10)
        lf = _font("Anton-Regular.ttf", 40)  # roue de chargement, coin bas droit comme dans le jeu
        d_fg.arc((W - 128, H - 132, W - 78, H - 82), 40, 330, fill=(255, 255, 255, 255), width=7)
        _text(fg, (W - 148, H - 106), "LOADING", lf, (255, 255, 255, 255), anchor="rm", stroke=4)
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
    if name == "loading":
        if title:
            f, lines = _fit(title, "Anton-Regular.ttf", 880, 2, 70, 52)
            lh = f.size * 1.16
            strip = Image.new("RGBA", (W, int(lh * len(lines) + 70)), (0, 0, 0, 0))
            sd = ImageDraw.Draw(strip)
            for i, ln in enumerate(lines):
                tw = f.getlength(ln) + 56
                cy = 35 + lh * (i + .5)
                sd.rectangle((W / 2 - tw / 2, cy - lh / 2, W / 2 + tw / 2, cy + lh / 2), fill=(10, 10, 12, 255))
                _text(strip, (W / 2, cy + 4), ln, f, (255, 255, 255, 255))
            strip = strip.rotate(2, resample=Image.Resampling.BICUBIC, expand=True)
            y = seam - strip.height // 2 if seam else 200 if bleed else 132  # colle au bord haut de l'affiche
            fg.alpha_composite(strip, (int(W / 2 - strip.width / 2), int(y)))
        if kicker:
            tag = Image.new("RGBA", (W, 140), (0, 0, 0, 0))
            _pill(tag, W / 2, 70, kicker, _font("Anton-Regular.ttf", 32), _rgb(GOLD), (10, 10, 12, 255), h=56,
                  outline=(10, 10, 12, 255))
            tag = tag.rotate(-3, resample=Image.Resampling.BICUBIC)
            fg.alpha_composite(tag, (0, 30 if bleed else 20))

    elif name == "versus":
        if bleed:
            shade(340, 1560, 150, 150)
        else:
            d_bg.rectangle((0, 0, W, H), fill=(8, 8, 10, 150))
        (x1, y1, w1, h1, _), (x2, y2, w2, h2, _) = wins
        mid = (y1 + h1 + y2) / 2
        d_fg.rectangle((0, mid - 6, W, mid + 6), fill=_rgb(GOLD))
        d_fg.ellipse((W / 2 - 54, mid - 54, W / 2 + 54, mid + 54), fill=(10, 10, 12, 255), outline=_rgb(GOLD),
                     width=6)
        _text(fg, (W / 2, mid + 3), "VS", _font("Anton-Regular.ttf", 60), _rgb(GOLD))
        lf = _font("Anton-Regular.ttf", 40)
        for (x, y, w, h, _), lab, col in ((wins[0], spec.get("lab1") or "BEFORE", "#FFFFFF"),
                                          (wins[1], spec.get("lab2") or "AFTER", PINK)):
            tw = lf.getlength(lab.upper()) + 40
            ly = y + h - 110 if bleed else y + 24  # plein ecran : etiquette en bas de chaque moitie
            d_fg.rectangle((x + 30, ly, x + 30 + tw, ly + 60), fill=(10, 10, 12, 220))
            d_fg.rectangle((x + 30, ly, x + 38, ly + 60), fill=_rgb(col))
            _text(fg, (x + 30 + tw / 2 + 4, ly + 32), lab.upper(), lf, _rgb(col))
        if title:
            f, lines, lh = title_lines(76, 54, 980)
            y0 = (190 if bleed else y1 / 2) - lh * (len(lines) - 1) / 2 + 6
            for i, ln in enumerate(lines):
                _text(fg, (W / 2, y0 + i * lh), ln, f, _rgb(GOLD) if i == len(lines) - 1 else (255, 255, 255, 255),
                      shadow=10)

    elif name == "news":
        if bleed:  # plein ecran : bandeau du bas pose sur l'image, LIVE en haut
            shade(300, 1300, 150, 110)
            x, y, w, h, _ = 0, 0, W, 1440, 0
            top = 200
            if two:
                gutter(wins[1][1])
        else:
            d_bg.rectangle((0, 0, W, H), fill=(6, 10, 22, 165))
            x, y, w, h, _ = wins[-1]
            top = wins[0][1]
        red, ink = _rgb("#E5202E"), (12, 12, 14, 255)
        bf = _font("Anton-Regular.ttf", 34)
        d_fg.rectangle((48, top - 80, 48 + bf.getlength("LIVE") + 64, top - 30), fill=red)
        d_fg.ellipse((62, top - 62, 76, top - 48), fill=(255, 255, 255, 255))
        _text(fg, (88, top - 55), "LIVE", bf, (255, 255, 255, 255), anchor="lm")
        if kicker:
            _text(fg, (48 + bf.getlength("LIVE") + 88, top - 55), kicker, bf, (255, 255, 255, 255), anchor="lm",
                  shadow=6)
        yb = y + h
        tag, band, tick_h = 54, 180, 48  # hauteurs fixes (deux lignes) : l'apercu du site tombe juste
        d_fg.rectangle((0, yb + tag, W, yb + tag + band), fill=(246, 244, 238, 255))
        d_fg.rectangle((0, yb + tag, 12, yb + tag + band), fill=red)
        if title:
            tf, lines = _fit(title, "Anton-Regular.ttf", 970, 2, 66, 48)
            lh = tf.size * 1.12
            for i, ln in enumerate(lines):
                _text(fg, (46, yb + tag + band / 2 + lh * (i - (len(lines) - 1) / 2) + 3), ln, tf, ink,
                      anchor="lm")
        tagf = _font("Anton-Regular.ttf", 36)
        d_fg.rectangle((0, yb + 4, tagf.getlength("BREAKING NEWS") + 56, yb + tag), fill=red)
        _text(fg, (28, yb + 4 + (tag - 4) / 2), "BREAKING NEWS", tagf, (255, 255, 255, 255), anchor="lm")
        ty = yb + tag + band
        d_fg.rectangle((0, ty, W, ty + tick_h), fill=ink)
        tick = "   •   ".join(t for t in (spec.get("ticker") or title or "GTA 6").split()) + "   •   "
        tkf = _font("BarlowCondensed-Bold.ttf", 32)
        tw = int(tkf.getlength(tick))
        strip = Image.new("RGBA", (tw * 2, tick_h), (0, 0, 0, 0))
        _text(strip, (0, tick_h // 2 + 1), tick + tick, tkf, _rgb("#FFD23F"), anchor="lm")
        extras.append((save(strip, "tick"), "", f"-mod(t*150,{tw})", ty, ""))
        progress(0, ty + tick_h, W, 6, "#E5202E", "#FFD23F")
    return save(bg, "bg"), save(fg, "fg"), extras


def theme_assets(outdir):
    """Calques des themes sans titre ni pastille, en 270x480, pour l'apercu de l'editeur du site."""
    from PIL import Image
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, th in THEMES.items():
        for lay in ("full", "split", "bfull", "bsplit"):
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


# ----------------------------------------------------------------------------- incrustations
# Demande de Jacques (2026-11-19), en plus de tout le reste : un texte "theme de la video" au
# style GTA (police, couleurs au choix) et un logo (celui de la chaine, ou un autre), chacun pose
# ou on veut et deplacable dans l'editeur. Dessines par Pillow, colles par-dessus l'image finale :
# ca marche pareil pour tous les formats, habillages et mises en page.
# x, y = centre de l'element en fraction de l'image finale ; size = echelle (texte) ou largeur (logo).
BADGE_STYLES = ("vice", "france", "wanted", "street", "neon")
LOGO_RE = re.compile(r"^data:image/(png|webp|jpeg);base64,[A-Za-z0-9+/]+={0,2}$")
LOGO_MAX = 24000  # caracteres : le logo voyage dans les reglages (entrees du workflow, ligne de commande)
FR_BLUE, FR_RED = "#0055A4", "#EF4135"


def _frac(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return round(min(max(v, lo), hi), 4) if math.isfinite(v) else default


def parse_badge(d):
    """Texte theme : {"text", "style", "x", "y", "size"}, ou None si pas de texte."""
    if not isinstance(d, dict):
        return None
    text = re.sub(r"\s+", " ", str(d.get("text") or "")).strip()[:40]
    if not text:
        return None
    return {"text": text, "style": d.get("style") if d.get("style") in BADGE_STYLES else "vice",
            "x": _frac(d.get("x"), 0, 1, .5), "y": _frac(d.get("y"), 0, 1, .2),
            "size": _frac(d.get("size"), .5, 2, 1)}


def logo_image(data):
    """Image Pillow du logo (data URL), ou None si illisible."""
    import base64
    import io
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1], validate=True)))
        if im.width * im.height > 1024 * 1024:
            return None
        return im.convert("RGBA")
    except Exception:
        return None


def parse_logo(d):
    """Logo : {"img" (data URL png/webp/jpeg), "x", "y", "size" (largeur / largeur de l'image)}."""
    if not isinstance(d, dict):
        return None
    img = str(d.get("img") or "")
    if len(img) > LOGO_MAX or not LOGO_RE.match(img) or logo_image(img) is None:
        return None
    return {"img": img, "x": _frac(d.get("x"), 0, 1, .86), "y": _frac(d.get("y"), 0, 1, .07),
            "size": _frac(d.get("size"), .05, .6, .18)}


def badge_image(b, W):
    """Le texte theme en RGBA transparent, a l'echelle d'une image de largeur W."""
    from PIL import Image, ImageDraw, ImageFilter
    fs = max(16, round(W * .075 * b["size"]))
    text, st = b["text"].upper(), b["style"]
    font = _font("Anton-Regular.ttf", fs)
    sw = max(2, round(fs * (.09 if st in ("street", "france", "vice") else .06)))
    l, t, r, bt = font.getbbox(text, stroke_width=sw)
    tw, th = r - l, bt - t
    pad = round(fs * (.42 if st == "wanted" else .3))
    w, h = tw + pad * 2, th + pad * 2 + (round(fs * .2) if st == "france" else 0)
    org = (pad - l, pad - t)
    clear = (0, 0, 0, 0)

    def mask(stroke=0):
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).text(org, text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
        return m

    def solid(c):
        return Image.new("RGBA", (w, h), c)
    core, edge = mask(), mask(sw)
    out = solid(clear)
    if st == "wanted":  # plaque noire, liseré or, texte or
        d = ImageDraw.Draw(out)
        d.rectangle((0, 0, w - 1, h - 1), fill=(10, 10, 12, 225), outline=_rgb(GOLD), width=max(3, fs // 14))
        out.paste(solid(_rgb(GOLD)), (0, 0), core)
        return out
    shadow = edge.filter(ImageFilter.GaussianBlur(fs * .08)).point(lambda v: int(v * .7))
    out.paste(solid((0, 0, 0, 255)), (round(fs * .04), round(fs * .07)), shadow)
    if st == "neon":  # halo cyan, coeur blanc
        glow = edge.filter(ImageFilter.GaussianBlur(fs * .12))
        for _ in range(2):
            out.alpha_composite(Image.composite(solid(_rgb(CYAN)), solid(clear), glow))
        out.paste(solid((255, 255, 255, 255)), (0, 0), core)
        return out
    out.paste(solid((12, 8, 16, 255)), (0, 0), edge)  # contour sombre
    fill = solid((255, 255, 255, 255))
    x0, x1 = org[0] + l, org[0] + r
    if st == "vice":  # degrade rose -> orange de haut en bas
        g = Image.linear_gradient("L").resize((w, h))
        fill = Image.composite(solid(_rgb(ORANGE)), solid(_rgb(PINK)), g)
    elif st == "france":  # bleu, blanc, rouge en trois bandes
        d = ImageDraw.Draw(fill)
        for i, col in enumerate((FR_BLUE, "#FFFFFF", FR_RED)):
            d.rectangle((x0 + (x1 - x0) * i / 3, 0, x0 + (x1 - x0) * (i + 1) / 3, h), fill=_rgb(col))
        d.rectangle((0, 0, x0, h), fill=_rgb(FR_BLUE))
        d.rectangle((x1, 0, w, h), fill=_rgb(FR_RED))
    out.paste(fill, (0, 0), core)
    if st == "france":  # filet tricolore sous le texte
        d = ImageDraw.Draw(out)
        y0, bh = pad + th + round(fs * .04), max(4, round(fs * .1))
        for i, col in enumerate((FR_BLUE, "#FFFFFF", FR_RED)):
            d.rectangle((pad + (w - 2 * pad) * i / 3, y0, pad + (w - 2 * pad) * (i + 1) / 3, y0 + bh),
                        fill=_rgb(col))
    if st in ("vice", "france"):  # penche vers l'avant, comme les titres de la serie
        k = .16
        out = out.transform((w + round(k * h), h), Image.Transform.AFFINE, (1, k, -k * h, 0, 1, 0),
                            Image.Resampling.BICUBIC)
    return out


def stamp_layers(spec, W, H, workdir):
    """PNG du logo et du texte theme, avec leur coin haut-gauche dans l'image finale."""
    from PIL import Image
    out = []

    def place(im, x, y, tag):
        px = round(min(max(x * W - im.width / 2, 0), max(0, W - im.width)))
        py = round(min(max(y * H - im.height / 2, 0), max(0, H - im.height)))
        f = pathlib.Path(workdir) / f"_stamp_{tag}_{os.getpid()}.png"
        im.save(f)
        out.append((f, px, py))
    lg = spec.get("logo")
    if lg:
        im = logo_image(lg["img"])
        if im:
            lw = max(8, round(W * lg["size"]))
            lh = max(8, round(im.height * lw / im.width))
            if lh > H * .5:
                lh, lw = round(H * .5), max(8, round(im.width * H * .5 / im.height))
            place(im.resize((lw, lh), Image.Resampling.LANCZOS), lg["x"], lg["y"], "logo")
    b = spec.get("badge")
    if b:
        im = badge_image(b, W)
        if im.width > W:
            im = im.resize((W, round(im.height * W / im.width)), Image.Resampling.LANCZOS)
        place(im, b["x"], b["y"], "badge")
    return out


def stamp(cmd, spec, W, H, workdir):
    """Ajoute le texte theme et le logo a une commande ffmpeg de rendu (-vf, ou -filter_complex
    qui sort [v]). Renvoie (commande, fichiers temporaires)."""
    layers = stamp_layers(spec, W, H, workdir)
    if not layers:
        return cmd, []
    cmd = list(cmd)
    if "-vf" in cmd:
        i = cmd.index("-vf")
        cmd[i:i + 2] = ["-filter_complex", f"[0:v]{cmd[i + 1]}[v]", "-map", "[v]", "-map", "0:a?"]
    i = cmd.index("-filter_complex")
    n = sum(c == "-i" for c in cmd[:i])
    cmd[i:i] = sum((["-loop", "1", "-i", str(f.resolve())] for f, _, _ in layers), [])
    i = cmd.index("-filter_complex")
    g, cur = cmd[i + 1], "v"
    for j, (_, x, y) in enumerate(layers):
        g += f";[{cur}][{n + j}:v]overlay={x}:{y}:shortest=1[st{j}]"
        cur = f"st{j}"
    cmd[i + 1] = g + f";[{cur}]format=yuv420p[stv]"
    cmd[cmd.index("[v]")] = "[stv]"
    return cmd, [f for f, _, _ in layers]


def run_stamped(cmd, spec, af, W, H, workdir, **kw):
    """run() d'un rendu, avec texte theme et logo si le short en a."""
    cmd, tmp = stamp(cmd, spec, W, H, workdir)
    try:
        return run(fold_audio(cmd, af), **kw)
    finally:
        for f in tmp:
            f.unlink(missing_ok=True)


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


def audio_cut(segs, s0):
    """Filtre son qui ne garde que les morceaux parles. atrim + concat et non aselect : sur le
    ffmpeg 8.0.1 du serveur, aselect ne coupe rien (teste le 2026-11-19 : son 35 s, image 17,8 s ;
    atrim + concat : 17,72 s). Un seul flux en entree et en sortie : marche en -af comme en graphe."""
    k = len(segs)
    cut = [f"atrim={a - s0:.3f}:{b - s0:.3f},asetpts=PTS-STARTPTS" for a, b in segs]
    if k == 1:
        return cut[0]
    return (f"asplit={k}" + "".join(f"[x{i}]" for i in range(k)) + ";"
            + ";".join(f"[x{i}]{c}[y{i}]" for i, c in enumerate(cut)) + ";"
            + "".join(f"[y{i}]" for i in range(k)) + f"concat=n={k}:v=0:a=1")


def fold_audio(cmd, af):
    """-af a cote de -filter_complex : le ffmpeg 8.0 du serveur l'ignore. Image sans les blancs,
    son entier : la bouche, la voix et les sous-titres se decalent (constate le 2026-11-19).
    On met le filtre son dans le graphe, ce que toutes les versions appliquent."""
    if not af or "-filter_complex" not in cmd:
        return cmd
    cmd = list(cmd)
    i = cmd.index("-filter_complex")
    cmd[i + 1] += f";[0:a]{af[1]}[aout]"
    cmd[cmd.index("0:a?")] = "[aout]"
    k = cmd.index("-af")
    del cmd[k:k + 2]
    return cmd


def check_sync(out):
    """Image et son doivent durer pareil : sinon le short est decale, on le dit fort."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
                        "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    d = {}
    for line in r.stdout.split():
        kind, _, val = line.partition(",")
        try:
            d[kind] = float(val)
        except ValueError:
            pass
    if "video" in d and "audio" in d and abs(d["video"] - d["audio"]) > 0.5:
        print(f"[alerte] {out.name} : image {d['video']:.1f} s, son {d['audio']:.1f} s, short decale")
    return d


def has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def render_spec(raw, words, spec, out, samples=None, subwords=None):
    """Rend un short a partir de son extrait brut. `words` en temps de l'extrait.
    `subwords` : mots a sous-titrer s'ils different de ceux qu'on entend (voix off)."""
    sw = subwords or words
    iw, ih = video_dims(raw)
    W, H, fs = FORMATS[spec["format"]]
    s0, s1 = spec["trim"]
    segs = keep_segments(words, s0, s1) if spec["cut"] else [(s0, s1)]
    cuts = plans(spec)
    lays = [r["layout"] for _, _, r in cuts]
    if spec["format"] not in ("9:16", "4:5") or iw <= ih:
        lays = ["full"] * len(cuts)
    if len(set(lays)) > 1:  # plein cadre et ecran partage alternes : une version de chaque, recollees
        return splice_layouts(raw, words, spec, out, samples, cuts, lays, segs, subwords)
    spec = {**spec, "layout": lays[0], "keys": [{**k, "layout": lays[0]} for k in spec.get("keys") or []]}
    subs = out.with_suffix(".ass")
    pre, af = "", []
    if segs != [(s0, s1)]:  # retrait des blancs : on ne garde que les morceaux parles
        sel = "+".join(f"between(t,{a - s0:.3f},{b - s0:.3f})" for a, b in segs)
        pre = f"select='{sel}',setpts=N/FRAME_RATE/TB,"
        af = ["-af", audio_cut(segs, s0)] if has_audio(raw) else []
    wins = theme_windows(spec)
    if wins:  # habillage GTA : son propre graphe, le reste de la fonction ne change pas
        th = THEMES[spec["theme"]]
        sub_y = th["sub"][theme_mode(spec)]
        build_ass(sw, segs, subs, "9:16", "", subs=spec["subs"], cta=spec.get("cta", ""), subc=spec.get("subc", "#FFFFFF"),
                  margin_v=round(H - sub_y - fs * .55), font="Anton", accent=th["accent"])
        fonts = out.parent / "fonts"
        fonts.mkdir(exist_ok=True)
        for f in FONTS.glob("*.ttf"):
            (fonts / f.name).write_bytes(f.read_bytes())
        inputs, g, cur, tmp = themed_vargs(spec, wins, iw, ih, segs, pre, samples, out.parent)
        g.append(f"[{cur}]subtitles={subs.name}:fontsdir=fonts,format=yuv420p[v]")
        print(f"    habillage : {spec['theme']} ({len(wins)} fenetre(s))")
        run_stamped(["ffmpeg", "-y", "-ss", str(s0), "-to", str(s1), "-i", str(raw.resolve()), *inputs,
                     "-filter_complex", ";".join(g), "-map", "[v]", "-map", "0:a?", *af,
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", "30",
                     "-c:a", "aac", "-b:a", "128k", str(out.name)], spec, af, W, H, out.parent,
                    cwd=str(out.parent.resolve()))
        check_sync(out)
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
        build_ass(sw, segs, subs, spec["format"], spec["title"], subs=spec["subs"], cta=spec.get("cta", ""), **look,
                  margin_v=round(H * SUB_SPLIT) if split else None, top=ht if split else 0)

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
        # sous-titres dans le jeu, sous le visage : jamais sur la tete du streamer
        build_ass(sw, segs, subs, spec["format"], spec["title"], top=ht,
                  margin_v=round(H * SUB_SPLIT), subs=spec["subs"], cta=spec.get("cta", ""), **look)
        fw_, fh_, fx, fy = rect_px(spec["cam"], iw, ih, W / ht)
        gw, gh, gx, gy = rect_px(spec["game"], iw, ih, W / hb)
        graph = (f"[0:v]{pre}split=2[a][b];"
                 f"[a]crop={fw_}:{fh_}:{fx}:{fy},scale={W}:{ht},setsar=1[top];"
                 f"[b]crop={gw}:{gh}:{gx}:{gy},scale={W}:{hb},setsar=1[bot];"
                 f"[top][bot]vstack,subtitles={subs.name}[v]")
        vargs = ["-filter_complex", graph, "-map", "[v]", "-map", "0:a?"]
    else:
        build_ass(sw, segs, subs, spec["format"], spec["title"], subs=spec["subs"], cta=spec.get("cta", ""), **look)
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
    run_stamped(["ffmpeg", "-y", "-ss", str(s0), "-to", str(s1), "-i", str(raw.resolve()),
                 *vargs, *af, "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "20", "-c:a", "aac", "-b:a", "128k", str(out.name)], spec, af, W, H, out.parent,
                cwd=str(out.parent.resolve()))
    subs.unlink(missing_ok=True)
    check_sync(out)
    return out


def splice_layouts(raw, words, spec, out, samples, cuts, lays, segs, subwords=None):
    """Short qui alterne plein cadre et ecran partage : chaque mise en page est rendue sur
    toute la duree (decor et sous-titres a leur place), puis on garde les bons morceaux."""
    s0 = spec["trim"][0]
    tmp = {}
    for lay in sorted(set(lays)):
        tmp[lay] = out.with_name(f"_{lay}_{out.name}")
        render_spec(raw, words, {**spec, "layout": lay, "badge": None, "logo": None,
                                 "keys": [{**k, "layout": lay} for k in spec["keys"]]}, tmp[lay], samples, subwords)
    order = sorted(tmp)
    pieces = []  # (mise en page, debut, fin) en temps du short final, plans voisins fusionnes
    for (a, b, _), lay in zip(cuts, lays):
        ta, tb = remap(s0 + a, segs), remap(s0 + b, segs)
        if tb - ta < 0.02:
            continue
        if pieces and pieces[-1][0] == lay:
            pieces[-1][2] = tb
        else:
            pieces.append([lay, ta, tb])
    pieces[-1][2] = 1e6  # le dernier morceau va jusqu'au bout
    g = [f"[{order.index(lay)}:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[q{i}]"
         for i, (lay, a, b) in enumerate(pieces)]
    g.append("".join(f"[q{i}]" for i in range(len(pieces))) + f"concat=n={len(pieces)}:v=1:a=0[v]")
    W, H, _ = FORMATS[spec["format"]]  # texte theme et logo poses une fois, sur le short recolle
    run_stamped(["ffmpeg", "-y", *sum((["-i", str(tmp[k].resolve())] for k in order), []),
                 "-filter_complex", ";".join(g), "-map", "[v]", "-map", "0:a?", "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "20", "-c:a", "copy", str(out.resolve())], spec, [], W, H, out.parent)
    for p in tmp.values():
        p.unlink(missing_ok=True)
    print(f"    {len(pieces)} passages plein cadre / ecran partage")
    return out


# ----------------------------------------------------------------------------- voix off
# Option "voice" (case a cocher, jamais par defaut) : commentaire anglais americain reecrit par Groq,
# lu avec la voix de Jacques (voix.py). Retour de Jacques apres le 1er essai (2026-11-16) : la voix
# d'origine ne doit plus s'entendre, le debit etait trop rapide et la voix finissait avant la video.
# Donc : son d'origine coupe, une phrase de commentaire par passage de la video, posee au moment
# ou ce passage passe a l'ecran, et des passages qui couvrent le short jusqu'a la derniere seconde.
# Les sous-titres suivent la voix off. Echantillon dechiffre par le workflow -> VOICE_REF.
# 3e retour de Jacques : pas de blancs, voix dynamique et heureuse qui motive, un hook fort a chaque
# video, et les bruits du jeu (moteurs, tirs) gardes sous sa voix, sans la voix d'origine : Demucs
# separe la voix d'origine du reste, et ce reste repasse sous la voix off, baisse quand elle parle.
VOICE_WPS = 2.7  # mots par seconde : remplit les passages (2.1 laissait des blancs) ; les Reels de
# reference parlent a 3,6-4 mots/s, Chatterbox suit moins vite sans sonner presse
SLOT_MIN, SLOT_MAX = 3.5, 9.0  # duree d'un passage commente (s)
MIN_TEMPO, MAX_TEMPO = 0.92, 1.08  # etirement accepte pour qu'une phrase remplisse son passage
BED_GAIN = 0.9  # bruits du jeu quand la voix se tait (ils reprennent le dessus entre deux phrases)
DUCK = "threshold=0.02:ratio=10:attack=15:release=350:makeup=1"  # quand la voix parle, le jeu s'efface


def unremap(t, segs):
    """Inverse de remap : temps dans le short final -> temps de l'extrait source."""
    acc = 0.0
    for a, b in segs:
        if t <= acc + b - a:
            return a + t - acc
        acc += b - a
    return segs[-1][1]


def voice_slots(heard, segs, dur):
    """Passages a commenter, en temps du short : [{"t", "d", "said"}] qui couvrent 0..dur sans trou.
    Coupe de preference en fin de phrase ou sur un silence de la video d'origine."""
    slots, last_end, brk = [{"t": 0.0, "said": []}], 0.0, False
    for w in heard:
        t = remap(float(w["start"]), segs)
        long = t - slots[-1]["t"]
        if slots[-1]["said"] and (long >= SLOT_MAX or (long >= SLOT_MIN and (brk or t - last_end > 0.5))):
            slots.append({"t": t, "said": []})
        slots[-1]["said"].append(w["word"].strip())
        last_end, brk = remap(float(w["end"]), segs), w["word"].strip()[-1:] in ".!?,;:"
    while dur - slots[-1]["t"] > SLOT_MAX + SLOT_MIN:  # longue fin sans paroles : on continue a commenter
        slots.append({"t": slots[-1]["t"] + SLOT_MAX, "said": []})
    if len(slots) > 1 and dur - slots[-1]["t"] < 2.0:  # dernier passage trop court : fusionne
        slots[-2]["said"] += slots.pop()["said"]
    for i, sl in enumerate(slots):
        sl["d"] = round((slots[i + 1]["t"] if i + 1 < len(slots) else dur) - sl["t"], 2)
        sl["said"] = " ".join(sl["said"])
    return slots


def parse_lines(raw, n):
    """Reponse "1. texte" ligne par ligne -> n lignes dans l'ordre (vide si un numero manque)."""
    got = {}
    for m in re.finditer(r"^\s*(\d+)[.)]\s*(.+?)\s*$", raw, re.M):
        k = int(m.group(1))
        if 1 <= k <= n and k not in got:
            t = m.group(2).replace("—", ", ").replace("–", ", ").replace("‑", "-")  # tirets longs : la voix les lit mal
            got[k] = re.sub(r"\s+", " ", t).strip().strip('"')
    return [got.get(k, "") for k in range(1, n + 1)]


def narration(slots, key, title=""):
    """Une phrase anglaise par passage, a la longueur du passage. Renvoie autant de lignes que de passages.
    Reponse en texte numerote : en mode JSON, Groq refusait souvent la sortie (400 json_validate_failed).
    Style appris (2026-11-16) sur deux Reels francais d'analyse GTA 6 que Jacques trouve naturels : on parle
    au spectateur, on pointe l'ecran, on explique chaque detail puis on reagit en deux mots, conseils concrets,
    hook concret et non crie. Le ton "commentateur surexcite" de la v3 sonnait faux."""
    rows = "\n".join(f'{i + 1}. [{sl["d"]:.1f} s, {slot_words(sl)} words max] '
                     f'original: {sl["said"] or "(no speech, just gameplay)"}' for i, sl in enumerate(slots))
    msg = (f"Short video titled: {title or 'GTA gameplay'}. It is cut into {len(slots)} consecutive moments; "
           f"for each one you get its length and what the original speaker says (may be any language):\n{rows}\n\n"
           "Write the English voice-over for each moment, in order, the way a real American GTA creator talks "
           "when he breaks the game down for his followers. It must sound like a human talking, not an AI or "
           "an announcer. How these creators talk:\n"
           "- They talk TO the viewer, casual and direct: 'So,', 'Now,', 'Look,', 'you guys', 'here's the thing'.\n"
           "- They point at the screen: 'see right here', 'look at this', 'watch what happens'.\n"
           "- They explain each detail in plain words ('that means...', 'basically...'), then react in two or "
           "three words ('Makes sense.', 'Huge.', 'Love that.').\n"
           "- They give concrete tips: 'if you swap your outfit after a robbery, the icon goes away'.\n"
           "- Line 1 is a concrete hook that makes people stay, often a promise: 'Here are the mistakes you "
           "can't make when GTA 6 drops, and at the end we max out the wanted level.' Never 'hey guys', never "
           "'in this video', no shouting, no hype words like 'insane' or 'mind-blowing' in every line.\n"
           "- Short spoken sentences, contractions, a little excitement when something is cool, calm otherwise.\n"
           "- The last line asks the viewer something or tells them to follow for more GTA 6 info.\n"
           "Do NOT just paraphrase the original line: rewrite it as spoken breakdown. Line 1 must be a NEW hook "
           "(not the original first sentence). Use screen pointers and short reactions where they fit naturally, "
           "in about a third of the lines, inside the sentence, never tacked on at the end ('..., look.').\n"
           "Example. Original: 'In GTA 5 Trevor carried a minigun in a tank top.' Spoken breakdown: 'Now look at "
           "Trevor in GTA 5, tank top, minigun, and nobody asked where he kept it. Classic.'\n"
           "Keep every useful fact, adapt freely (not a word-for-word translation). Each line must FILL its "
           "moment: get close to its word limit but NEVER go over it (the line is cut off when the moment ends). Lines flow as one continuous breakdown. "
           "No emoji, no hashtags, no stage directions.\n"
           f"Answer with exactly {len(slots)} lines and nothing else, each starting with its number: "
           "\"1. ...\", \"2. ...\" and so on.")
    lines = [""] * len(slots)
    for _ in range(3):  # ligne manquante ou reponse vide : on redemande
        raw = chat_retry([{"role": "system", "content": "You write timed voice-over scripts."},
                          {"role": "user", "content": msg}], key, json_mode=False,
                         extra={"reasoning_effort": "low", "max_completion_tokens": 6000})  # gpt-oss reflechissait jusqu a couper la reponse
        lines = parse_lines(raw, len(slots))
        if sum(1 for x in lines if x) >= len(slots) - 1:
            break
        print("[info] texte de la voix off incomplet, nouvel essai")
    # Le modele ignore souvent les longueurs (24 mots pour 3,7 s) : on lui fait raccourcir les
    # lignes trop longues, puis on coupe a la derniere virgule si ca deborde encore.
    caps = [slot_words(sl) for sl in slots]
    long = [i for i, (x, n) in enumerate(zip(lines, caps)) if len(x.split()) > n]
    if long:
        ask = "\n".join(f"{j + 1}. [max {caps[i]} words] {lines[i]}" for j, i in enumerate(long))
        try:
            raw = chat_retry([{"role": "user", "content":
                               "Shorten each voice-over line to at most its word limit. Keep the same casual "
                               "spoken tone and the key fact, drop the rest. Each must stay a natural, grammatical sentence a "
                               "person would say out loud (keep 'so', 'you', 'look'), never telegraphic. Answer with the numbered lines "
                               f"only, same numbers:\n{ask}"}], key, json_mode=False,
                             extra={"reasoning_effort": "low", "max_completion_tokens": 4000})
            for i, x in zip(long, parse_lines(raw, len(long))):
                lines[i] = x or lines[i]
        except (Exception, SystemExit) as e:
            print(f"[info] raccourcissement impossible ({type(e).__name__})")
    return [fit_words(x, n) for x, n in zip(lines, caps)]


def slot_words(sl):
    return max(4, round((sl["d"] - .3) * VOICE_WPS))


def fit_words(line, n):
    """Au plus n (+1 de tolerance) mots, coupe de preference apres une virgule ou un point."""
    ws = line.split()
    if len(ws) <= n + 1:
        return line
    for k in range(n, max(2, n // 2), -1):
        if ws[k - 1][-1] in ",.!?;:":
            return " ".join(ws[:k]).rstrip(",;:") + ("" if ws[k - 1][-1] in ".!?" else ".")
    return " ".join(ws[:n]).rstrip(",;:") + "."


def clean_tmp(tmp):
    for p in tmp.parent.glob(f"{tmp.name}*"):
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)


EN_WORDS = {"the", "and", "you", "to", "is", "it", "a", "of", "that", "in", "this", "we", "they", "so",
            "just", "can", "be", "on", "with", "for", "your", "what", "are", "was", "but", "not"}


def is_english(words):
    ws = [re.sub(r"\W", "", w["word"]).lower() for w in words]
    ws = [w for w in ws if w]
    return len(ws) >= 15 and sum(w in EN_WORDS for w in ws) / len(ws) >= 0.2


def separate(raw, segs, s0, s1, tmp):
    """Son du short (blancs deja coupes, comme a l'image) separe par Demucs (licence MIT) :
    (voix du narrateur d'origine, reste = moteurs, tirs, musique). (None, None) si impossible."""
    wav = tmp.with_name(f"{tmp.name}_src.wav")
    af = []
    if segs != [(s0, s1)]:
        sel = "+".join(f"between(t,{a - s0:.3f},{b - s0:.3f})" for a, b in segs)
        af = ["-af", audio_cut(segs, s0)]
    try:
        run(["ffmpeg", "-y", "-ss", str(s0), "-to", str(s1), "-i", str(raw), "-vn", *af,
             "-ac", "2", "-ar", "44100", str(wav)])
        run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-d", "cpu",
             "-o", str(tmp.parent / f"{tmp.name}_sep"), str(wav)])
    except (Exception, SystemExit) as e:
        print(f"[info] son d'origine non separe ({type(e).__name__}) : voix off seule")
        return None, None
    d = tmp.parent / f"{tmp.name}_sep" / "htdemucs" / wav.stem
    voc, bed = d / "vocals.wav", d / "no_vocals.wav"
    return (voc if voc.exists() else None), (bed if bed.exists() else None)


def style_ref(vocals, heard, segs, tmp, length=14.0):
    """Extrait de la voix du narrateur d'origine qui sert de modele de diction a Chatterbox (debit,
    intonations, facon d'expliquer). Seulement s'il parle anglais : sinon son accent passerait dans
    la voix de Jacques, et voix.py garde la voix americaine par defaut."""
    if not (vocals and is_english(heard)):
        return None
    starts = [remap(float(w["start"]), segs) for w in heard]
    best = max(starts, key=lambda t: sum(t <= x < t + length for x in starts))  # passage le plus parle
    out = tmp.with_name(f"{tmp.name}_style.wav")
    try:
        run(["ffmpeg", "-y", "-ss", f"{max(0.0, best - 0.2):.2f}", "-t", f"{length}", "-i", str(vocals),
             "-af", "loudnorm=I=-18", "-ac", "1", "-ar", "24000", str(out)])
    except SystemExit:
        return None
    return out


def voice_over(raw, words, spec, out, samples=None, key=None, subwords=None):
    """render_spec, avec la voix off de Jacques si spec["voice"]. Sans voix ou en cas de
    souci (echantillon absent, Chatterbox pas installe), le short sort sans voix off."""
    ref = os.environ.get("VOICE_REF", "")
    if not spec.get("voice"):
        return render_spec(raw, words, spec, out, samples, subwords)
    if not (key and ref and pathlib.Path(ref).exists()):
        print("[info] voix off demandee mais echantillon ou cle absents : short sans voix off")
        return render_spec(raw, words, spec, out, samples, subwords)
    s0, s1 = spec["trim"]
    segs = keep_segments(words, s0, s1) if spec["cut"] else [(s0, s1)]
    dur = sum(b - a for a, b in segs)
    heard = [w for w in words if float(w["start"]) >= s0 and float(w["end"]) <= s1]
    tmp = out.with_name(f"_voix_{out.stem}")
    try:
        import voix
        vocals, bed = separate(raw, segs, s0, s1, tmp)
        style = style_ref(vocals, heard, segs, tmp)
        print(f"    diction : {'narrateur d origine' if style else 'voix americaine par defaut'}")
        slots = voice_slots(heard, segs, dur)
        lines = narration(slots, key, spec.get("title", ""))
        print(f"    voix off : {len(slots)} passages, {sum(len(x.split()) for x in lines)} mots")
        t, ins, graph = time.time(), [], []
        for i, (sl, line) in enumerate(zip(slots, lines)):
            if not line:
                continue
            wav = voix.speak(line, ref, tmp.with_name(f"{tmp.name}_{i:02d}.wav"), style)
            room = sl["d"] - 0.25  # petit souffle avant le passage suivant
            tempo = min(max(ffprobe_duration(wav) / room, MIN_TEMPO), MAX_TEMPO)
            n = len(ins) // 2
            ins += ["-i", str(wav)]
            graph.append(f"[{n}:a]atempo={tempo:.3f},atrim=end={room:.3f},"
                         f"afade=t=out:st={max(0, room - .12):.3f}:d=0.12,"
                         f"adelay={int(sl['t'] * 1000) + 120}:all=1[v{n}]")
        print(f"    voix generee en {time.time() - t:.0f} s")
        k = len(ins) // 2
        fit = tmp.with_suffix(".m4a")
        graph.append("".join(f"[v{i}]" for i in range(k))
                     + f"amix=inputs={k}:normalize=0:duration=longest,apad=whole_dur={dur:.3f},"
                       f"atrim=end={dur:.3f}[a]")
        run(["ffmpeg", "-y", *ins, "-filter_complex", ";".join(graph), "-map", "[a]",
             "-ac", "1", "-ar", "24000", "-c:a", "aac", "-b:a", "128k", str(fit)])
        resp = _transcribe_one(fit.read_bytes(), key)  # horodatage des mots dits, pour les sous-titres
        said = [{"word": w["word"], "start": unremap(float(w["start"]), segs),
                 "end": unremap(min(float(w["end"]), dur), segs)} for w in resp.get("words") or []]
    except (Exception, SystemExit) as e:  # la voix off ne doit jamais faire perdre le short (run -> sys.exit)
        print(f"[info] voix off abandonnee ({type(e).__name__}: {e})")
        clean_tmp(tmp)
        return render_spec(raw, words, spec, out, samples, subwords)
    render_spec(raw, words, spec, out, samples, subwords=said or None)
    mixed = out.with_name(f"_mix_{out.name}")
    if bed:  # bruits du jeu sous la voix off : forts entre les phrases, effaces quand elle parle
        graph = (f"[1:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,asplit=2[vo][sc];"
                 f"[2:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={BED_GAIN}[bd];"
                 f"[bd][sc]sidechaincompress={DUCK}[duck];[duck][vo]amix=inputs=2:normalize=0:duration=first[a]")
        run(["ffmpeg", "-y", "-i", str(out), "-i", str(fit), "-i", str(bed), "-filter_complex", graph,
             "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", str(mixed)])
    else:  # separation impossible : voix off seule, jamais la voix d'origine dessous
        run(["ffmpeg", "-y", "-i", str(out), "-i", str(fit), "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", str(mixed)])
    mixed.replace(out)
    clean_tmp(tmp)
    return out


# ----------------------------------------------------------------------------- sous-titres traduits

def phrases(words, maxw=14, gap=.5):
    """Mots -> phrases entendues (coupees sur une pause, une ponctuation finale ou 14 mots)."""
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= maxw or float(w["start"]) - float(cur[-1]["end"]) > gap
                    or float(w["start"]) < float(cur[-1]["start"]) or re.search(r"[.!?]$", cur[-1]["word"].strip())):
            out.append(cur)
            cur = []
        cur.append(w)
    return out + [cur] if cur else out


def spread(text, a, b):
    """Mots d'une phrase traduite repartis sur [a, b] au prorata de leur longueur."""
    toks = text.split()
    wt = [len(t) + 2 for t in toks]
    tot, cum, out = sum(wt), 0, []
    for t, k in zip(toks, wt):
        out.append({"word": t, "start": round(a + (b - a) * cum / tot, 2),
                    "end": round(a + (b - a) * (cum + k) / tot, 2)})
        cum += k
    return out


def translate_words(words, lang, heard, key, batch=40):
    """Sous-titres dans la langue choisie quand la video en parle une autre : chaque phrase est
    traduite, puis ses mots sont repartis sur le temps de la phrase (les blancs coupes ne bougent
    pas : une phrase ne chevauche jamais une pause). None si rien a traduire ou en cas d'echec."""
    if not words or not key or heard == lang:
        return None
    name = LANGS[lang][0].capitalize()
    groups, out = phrases(words), []
    for i in range(0, len(groups), batch):
        part = groups[i:i + batch]
        rows = "\n".join(f"{j + 1}. " + " ".join(w["word"].strip() for w in g) for j, g in enumerate(part))
        try:
            raw = chat_retry([{"role": "system", "content": "You translate video subtitles."},
                              {"role": "user", "content":
                               f"Translate each numbered subtitle line into natural spoken {name}. The lines "
                               "follow each other in the video. Keep the meaning and the tone (slang, jokes, "
                               "swearing), keep names, add nothing, no emoji. If a line is already in "
                               f"{name}, copy it. Answer with exactly {len(part)} lines and nothing else, each "
                               f"starting with its number: \"1. ...\", \"2. ...\".\n\n{rows}"}],
                             key, json_mode=False, extra={"reasoning_effort": "low", "max_completion_tokens": 6000})
            lines = parse_lines(raw, len(part))
        except Exception as e:  # sous-titres d'origine plutot que pas de short
            print(f"[info] traduction des sous-titres impossible ({type(e).__name__}), sous-titres d'origine")
            return None
        for g, line in zip(part, lines):
            if line:
                out += spread(line, float(g[0]["start"]), max(float(w["end"]) for w in g))
            else:
                out += [{"word": w["word"].strip(), "start": w["start"], "end": w["end"]} for w in g]
    return out


# ----------------------------------------------------------------------------- mode resume
# Demande de Jacques (2026-11-24) : en plus des moments forts, un resume de la video en un seul
# short (ou plusieurs parties a la suite pour une longue video), qui raconte l'histoire en continu :
# accroche, contexte, montee, moment fort, chute. Passages de la video mis bout a bout, dans l'ordre.
RECAP_CHARS = 14000  # au-dela, on resume d'abord la transcription en temps forts (quota Groq par minute)
SENT_END = re.compile(r"[.!?…][\"'»)]*$")  # mot qui finit une phrase


def tags_str(tags):
    """Hashtags du LLM -> "#a #b" ("#" garanti, espaces internes retires : pret a publier)."""
    if isinstance(tags, str):
        tags = tags.split()
    if not isinstance(tags, list):
        return ""
    return " ".join("#" + re.sub(r"[\s#]+", "", str(t)) for t in tags if str(t).strip("# "))[:300]


def json_reply(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def chapter_bounds(words, n):
    """Video coupee en n chapitres de meme longueur, chaque coupe sur la plus longue pause proche."""
    total = float(words[-1]["end"])
    cuts = [0.0]
    for k in range(1, n):
        t = total * k / n
        gaps = [(float(b["start"]) - float(a["end"]), (float(a["end"]) + float(b["start"])) / 2)
                for a, b in zip(words, words[1:]) if abs(float(a["end"]) - t) < 60]
        cuts.append(max(gaps)[1] if gaps else t)
    return list(zip(cuts, cuts[1:] + [total + 1]))


def recap_text(ws, key):
    """Ce que le LLM lit pour monter le resume : la transcription, ou ses temps forts si trop longue."""
    text = transcript_lines(ws, 14)
    if len(text) <= RECAP_CHARS:
        return text, False
    beats = []
    for i, part in enumerate(chunk_lines(text)):
        print(f"    temps forts, tranche {i + 1}")
        msg = (f"Voici un passage de la transcription horodatee d'une video. Chaque ligne commence par son "
               f"temps de debut en secondes.\n\n{part}\n\nReleve dans l'ordre les temps forts de l'histoire : "
               "ce qui se passe, les revelations, les rebondissements, les phrases cles. Reponds STRICTEMENT en "
               'JSON : {"beats":[{"start":<secondes>,"end":<secondes>,"what":"<ce qui se passe, 1 phrase>"}]} '
               "avec 6 a 15 temps forts de 5 a 25 s, start et end pris dans la transcription.")
        try:
            for b in json_reply(chat_retry([{"role": "user", "content": msg}], key)).get("beats") or []:
                beats.append(f"[{int(float(b['start']))}-{int(float(b['end']))}s] {str(b['what'])[:160]}")
        except (urllib.error.URLError, KeyError, TypeError, ValueError, AttributeError) as e:
            print(f"[info] temps forts indisponibles sur cette tranche ({e})")
    return ("\n".join(beats), True) if beats else (text[:RECAP_CHARS], False)


def plan_recap(ws, key, opts, i, n, a, b):
    """Le LLM choisit les passages du resume d'un chapitre. None si la reponse est inexploitable."""
    text, beats = recap_text(ws, key)
    dmin, dmax = opts["dmin"], opts["dmax"]
    where = (f" (chapitre {i + 1} sur {n} : la video est coupee en {n} chapitres qui se suivent, celui-ci va "
             f"de {int(a)} s a {int(b)} s ; le resume ne couvre que ce chapitre"
             + (" et donne envie de voir la suite)" if i + 1 < n else ")")) if n > 1 else ""
    src = ("Voici les temps forts d'une video, avec leurs horaires en secondes (pas la transcription complete)"
           if beats else "Voici la transcription horodatee d'une video. Chaque ligne commence par son temps "
                         "de debut en secondes")
    msg = (
        f"{src}{where}.\n\n{text}\n\n"
        f"Monte UN resume vertical de {dmin} a {dmax} secondes (duree totale) qui raconte cette video en "
        "continu, comme si le spectateur la regardait en accelere sans jamais perdre le fil. Le resume est "
        "fait de passages de la video mis bout a bout. Reponds STRICTEMENT en JSON :\n"
        '{"title":"<titre accrocheur, max 60 caracteres>","score":<chance qu on le regarde jusqu au bout, 0-100>,'
        '"reason":"<pourquoi ca tient, 1 phrase>","caption":"<legende de post, 1-2 phrases accrocheuses>",'
        '"hashtags":["<4 a 6 hashtags pertinents>"],"parts":[{"start":<secondes>,"end":<secondes>}]}\n'
        "Regles :\n"
        "- 4 a 10 passages, dans l'ordre de la video, sans chevauchement, start et end pris dans la video ;\n"
        "- chaque passage est une idee complete de 3 a 20 s : il commence au debut d'une phrase et finit a "
        "la fin d'une phrase ;\n"
        "- le premier passage accroche des la premiere seconde : il pose l'enjeu ou une question qu'on veut "
        "voir resolue, jamais une salutation ni une introduction ;\n"
        "- mis bout a bout, les passages se comprennent sans avoir vu la video : le contexte utile, la "
        "montee, le moment fort, puis la chute ;\n"
        "- garde les moments forts, les rebondissements et les phrases qui retiennent l'attention ; ecarte "
        "les repetitions, digressions, pubs, remerciements et temps morts ;\n"
        "- le dernier passage conclut : la chute, la revelation ou la reponse a l'accroche ;\n"
        f"- la somme des durees des passages est entre {dmin} et {dmax} secondes.\n"
        "Sans aucun emoji. Rien d'autre que le JSON.\n" + lang_rule(opts["lang"]))
    if CATS.get(opts["cat"]):
        msg += "\n\nContexte : " + CATS[opts["cat"]]
    if opts["find"]:
        msg += "\n\nLe spectateur veut surtout voir : « " + opts["find"] + " ». Centre le resume dessus."
    d = json_reply(chat_retry([{"role": "system", "content": "Tu es un monteur expert en resumes video pour "
                                                            "TikTok, Reels et YouTube Shorts."},
                               {"role": "user", "content": msg}], key))
    parts = []
    for p in d.get("parts") or [] if isinstance(d, dict) else []:
        try:
            s, e = float(p["start"]), float(p["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if a - 2 <= s < e <= b + 2 and e - s >= 1:
            parts.append((s, e))
    if not parts:
        return None
    return {"title": str(d.get("title") or "Recap")[:80], "score": int(d.get("score") or 0),
            "reason": str(d.get("reason") or "")[:200], "caption": str(d.get("caption") or "")[:400],
            "hashtags": tags_str(d.get("hashtags")), "hook_text": "", "parts": parts}


def fallback_recap(ws, opts, k=5):
    """Secours si le LLM echoue : k passages repartis regulierement sur le chapitre."""
    a, b = float(ws[0]["start"]), float(ws[-1]["end"])
    ln = (opts["dmin"] + opts["dmax"]) / 2 / k
    step = max(ln, (b - a - ln) / max(k - 1, 1))
    parts = [(a + j * step, a + j * step + ln) for j in range(k) if a + j * step + ln <= b + 1]
    return {"title": "Recap", "score": 50, "reason": "decoupage regulier (IA indisponible)",
            "caption": "", "hashtags": "", "hook_text": "", "parts": parts or [(a, min(b, a + ln * k))]}


def fit_parts(parts, words, opts, a, b):
    """Cale chaque passage sur les mots (fin de phrase : jusqu'a la pause suivante, 3 s max), recolle
    les passages qui se touchent, puis tient la duree totale dans [dmin, dmax]."""
    cut = opts["cut"]
    ws_all = [w for w in words if a <= float(w["start"]) < b]

    def stop(j):  # le mot j termine une phrase (ponctuation de Whisper, sinon une vraie pause)
        w = ws_all[j]
        return (j + 1 >= len(ws_all) or bool(SENT_END.search(w["word"].strip()))
                or float(ws_all[j + 1]["start"]) - float(w["end"]) > .5)
    out = []
    for s, e in sorted(parts):
        lo = out[-1][1] if out else a
        idx = [j for j, w in enumerate(ws_all) if float(w["start"]) >= max(s - .3, lo) and float(w["end"]) <= e + .3]
        if not idx:
            continue
        i0 = idx[0]  # debut : recule jusqu'au debut de la phrase (5 s max)
        for k in range(idx[0], -1, -1):
            if float(ws_all[idx[0]]["start"]) - float(ws_all[k]["start"]) > 5 or float(ws_all[k]["start"]) < lo:
                break
            if k == 0 or stop(k - 1):
                i0 = k
                break
        j = idx[-1]  # fin : avance jusqu'a la fin de la phrase (6 s max), sinon recule sur la derniere finie
        while not stop(j) and float(ws_all[j + 1]["end"]) - e < 6:
            j += 1
        if not stop(j):
            j = next((k for k in range(idx[-1], i0, -1) if stop(k)), j)
        start = max(lo, float(ws_all[i0]["start"]) - .05)
        end = float(ws_all[j]["end"]) + .08
        if j + 1 < len(ws_all):
            end = min(end, float(ws_all[j + 1]["start"]))
        if out and start - out[-1][1] < .3:
            out[-1][1] = end  # passages qui se suivent : un seul, pas de coupe inutile
        elif end - start >= 1:
            out.append([start, end])

    def total():
        return sum(kept(words, x, y, cut) for x, y in out)
    while len(out) > 2 and total() > opts["dmax"]:  # trop long : on retire le passage du milieu le plus adapte
        extra = total() - opts["dmax"]
        drop = min(range(1, len(out) - 1), key=lambda i: (kept(words, *out[i], cut) < extra,
                                                           abs(kept(words, *out[i], cut) - extra)))
        out.pop(drop)
    for whole in (True, False):  # encore trop long : la fin du dernier passage recule de phrase en phrase,
        x = out[-1] if out else None  # puis de mot en mot si une phrase ne suffit pas
        if not x or total() <= opts["dmax"]:
            break
        for k in reversed([k for k, w in enumerate(ws_all) if x[0] < float(w["start"]) < x[1]]):
            if whole and not stop(k):
                continue
            x[1] = float(ws_all[k]["end"]) + .08
            if total() <= opts["dmax"] or x[1] - x[0] < 3:
                break
    if out and total() < opts["dmin"]:  # trop court : le dernier passage continue, phrase par phrase
        x = out[-1]
        for k in [k for k, w in enumerate(ws_all) if float(w["start"]) >= x[1]]:
            x[1] = float(ws_all[k]["end"]) + .08
            if stop(k) and total() >= opts["dmin"]:
                break
    return [(round(max(0.0, x), 2), round(y, 2)) for x, y in out]


def select_recaps(words, key, n, opts):
    """n resumes : un par chapitre de la video (n = 1 : toute la video en un seul short)."""
    print("[3/4] montage du resume (LLM Groq)...")
    total = float(words[-1]["end"])
    n = max(1, min(n, int(total // (2 * opts["dmin"])) or 1))  # chaque chapitre doit pouvoir nourrir un resume
    out = []
    for i, (a, b) in enumerate(chapter_bounds(words, n)):
        ws = [w for w in words if a <= float(w["start"]) < b]
        if not ws:
            continue
        if n > 1:
            print(f"    chapitre {i + 1}/{n} : {a:.0f}-{min(b, total):.0f} s")
        rc = None
        try:
            rc = plan_recap(ws, key, opts, i, n, a, b)
        except (urllib.error.URLError, KeyError, TypeError, ValueError, AttributeError) as e:
            print(f"[info] LLM indisponible pour ce resume ({e})")
        if not rc:
            print("[info] resume de secours : passages repartis regulierement")
            rc = fallback_recap(ws, opts)
        parts = fit_parts(rc["parts"], words, opts, a, b)
        if not parts:
            continue
        title = rc["title"]
        if n > 1:
            title = f"{title[:60]} ({LANGS[opts['lang']][1]} {i + 1}/{n})"
        out.append({**rc, "title": title, "start": parts[0][0], "end": parts[-1][1], "parts": parts})
        print(f"    {len(parts)} passages, {sum(kept(words, x, y, opts['cut']) for x, y in parts):.0f} s : {title}")
    return out


def shift_parts(words, parts):
    """Mots de la source -> temps dans le resume monte (meme ordre que concat_parts)."""
    out, off = [], 0.0
    for a, b in parts:
        for w in words:
            if float(w["start"]) >= a - .02 and float(w["end"]) <= b + .02:
                out.append({"word": w["word"].strip(), "start": round(max(0.0, float(w["start"]) - a) + off, 2),
                            "end": round(min(float(w["end"]), b) - a + off, 2)})
        off += b - a
    return out


def hook_span(words, clip):
    """Ou se trouve, dans l'extrait, la reaction designee par l'IA (hook_text) ? Renvoie
    (debut, fin) dans la source, ou None si introuvable ou si le short s'ouvre deja dessus."""
    want = [x for x in (norm_word(w) for w in str(clip.get("hook_text", "")).split()) if x]
    ws = [w for w in words if clip["start"] <= float(w["start"]) and float(w["end"]) <= clip["end"]]
    n, best = len(want), None
    for i in range(len(ws)):
        hit = sum(norm_word(w["word"]) == x for w, x in zip(ws[i:i + n], want))
        if n and hit >= max(1, -(-2 * n // 3)) and (not best or hit > best[0]):
            best = (hit, i)
    if not best:
        return None
    seg = ws[best[1]:best[1] + n]
    a = max(clip["start"], float(seg[0]["start"]) - .15)
    b = min(clip["end"], max(float(w["end"]) for w in seg) + .35, a + HOOK_MAX)
    if a - clip["start"] < 3 or b - a < .8:
        return None
    return round(a, 2), round(b, 2)


def hook_raw(video, raw, r0, s, hook, r1):
    """Extrait brut avec la reaction recopiee juste avant le debut du short :
    [marge avant] + [reaction] + [le short et la marge apres]. False si ffmpeg echoue."""
    ok = concat_parts(video, raw, [(r0, s), hook, (s, r1)])
    if not ok:
        print("[info] ouverture sur la reaction impossible (pas de piste son ?), extrait normal")
    return ok


def concat_parts(video, raw, parts):
    """Passages de la source mis bout a bout (coupes exactes, reencodes). False si ffmpeg echoue."""
    parts = [p for p in parts if p[1] - p[0] > .05]
    cmd = ["ffmpeg", "-y"]
    for a, b in parts:
        cmd += ["-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(video.resolve())]
    g = "".join(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];[{i}:a]asetpts=PTS-STARTPTS[a{i}];" for i in range(len(parts)))
    g += "".join(f"[v{i}][a{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=1[v][a]"
    cmd += ["-filter_complex", g, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "21", "-r", "30", "-c:a", "aac", "-b:a", "160k", str(raw)]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE).returncode == 0


def shift_words(words, r0, s, hook, r1):
    """Mots de la source -> temps dans l'extrait brut (meme decoupage que hook_raw)."""
    def w_(w, t):
        return {"word": w["word"].strip(), "start": round(float(w["start"]) + t, 2),
                "end": round(float(w["end"]) + t, 2)}
    if not hook:
        return [w_(w, -r0) for w in words if float(w["end"]) > r0 and float(w["start"]) < r1]
    ha, hb = hook
    pre = [w_(w, -r0) for w in words if float(w["end"]) > r0 and float(w["start"]) < s]
    mid = [w_(w, s - r0 - ha) for w in words if float(w["start"]) >= ha and float(w["end"]) <= hb]
    post = [w_(w, hb - ha - r0) for w in words if float(w["start"]) >= s and float(w["start"]) < r1]
    return pre + mid + post


def render_clip(video, words, clip, workdir, index, iw, ih, use_face=True, opts=None,
                total=None, key=None):
    """Extrait brut (moment + marge) puis short. Renvoie l'entree de shorts.json."""
    opts = opts or parse_opts("{}")
    if opts.get("camp"):
        need = CAMPAIGNS[opts["camp"]]["tags"].split()
        have = [t for t in clip.get("hashtags", "").split() if t.lower() not in {n.lower() for n in need}]
        clip = {**clip, "hashtags": " ".join(need + have)}
    name = f"short_{index:02d}_{clip['score']:03d}_{slugify(clip['title'])}"
    total = total or ffprobe_duration(video)
    raw = workdir / f"raw_{index:02d}.mp4"
    hook, T = None, 0.0
    if clip.get("parts"):  # resume : les passages mis bout a bout forment l'extrait brut
        if not concat_parts(video, raw, clip["parts"]):
            sys.exit("[erreur] montage du resume impossible (ffmpeg)")
        r0, r1 = 0.0, sum(b - a for a, b in clip["parts"])
        rel = shift_parts(words, clip["parts"])
        s0, s1 = 0.0, r1
    else:
        r0, r1 = max(0.0, clip["start"] - EDGE), min(total, clip["end"] + EDGE)
        hook = hook_span(words, clip) if opts["hook"] else None
        if hook and not hook_raw(video, raw, r0, clip["start"], hook, r1):
            hook = None
        if not hook:
            # extrait reencode (coupe exacte) : c'est lui que l'editeur montre et recadre
            run(["ffmpeg", "-y", "-ss", str(r0), "-to", str(r1), "-i", str(video.resolve()),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "160k",
                 str(raw)])
        rel = shift_words(words, r0, clip["start"], hook, r1)
        T = hook[1] - hook[0] if hook else 0.0
        s0, s1 = clip["start"] - r0, clip["end"] - r0 + T
    subw = translate_words(rel, opts["lang"], opts.get("heard", ""), key)
    if subw:
        print(f"    sous-titres traduits ({opts.get('heard') or '?'} -> {opts['lang']})")
    if hook:
        print(f"    ouverture sur la reaction : {hook[0]:.1f}-{hook[1]:.1f} s repris en tete")
    samples = face_samples(raw, s0, s1) if use_face else []
    box = CAMPAIGNS.get(opts.get("camp") or "", {}).get("box")
    if box:  # streamer connu : seuls les visages dans sa facecam comptent (pas ceux des cinematiques)
        samples = [smp if smp[1] is not None and in_box(smp, box) else (smp[0], None, None, None)
                   for smp in samples]
    elif samples:
        box = facecam_box(raw, samples, s0, s1)
    if box:
        print(f"    cadre de la facecam : x {box[0]:.2f} y {box[1]:.2f} l {box[2]:.2f} h {box[3]:.2f}")
    spec = auto_spec(samples, opts, iw, ih, s0, s1, clip["title"] if opts["title"] else "",
                     clip.get("hashtags", ""), box)
    if opts["layout"] == "auto" and spec["layout"] == "full" and spec["theme"] != "versus":
        # pas de facecam de streamer : le realisateur choisit plein cadre ou partage plan par plan
        spec = apply_director(spec, director(raw, s0, s1, iw, ih, spec))
    out = voice_over(raw, rel, spec, workdir / f"{name}.mp4", samples, key, subw)
    # pack de publication, pret a copier-coller
    tags = clip.get("hashtags", "")
    (workdir / f"{name}.txt").write_text(
        f"{clip['title']}\n\n{clip.get('caption', '')}\n\n{tags}\n", encoding="utf-8")
    return {**clip, "file": out.name, "raw": raw.name, "dur": round(r1 - r0 + T, 2), "spec": spec,
            "hook": list(hook) if hook else None,
            "words": [[w["word"], w["start"], w["end"]] for w in rel],
            **({"subwords": [[w["word"], w["start"], w["end"]] for w in subw]} if subw else {})}


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
        "Sans emoji.\n" + lang_rule(spec.get("lang", "en")))
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
    if spec.pop("auto", False):
        spec = apply_director(spec, director(raw, *spec["trim"], *video_dims(raw), spec))
    samples = None
    if spec["track"] and spec["layout"] == "full" and not spec["keys"]:
        samples = face_samples(raw, *spec["trim"])
    out = pathlib.Path(args.edit_out)
    subw = [{"word": w, "start": a, "end": b} for w, a, b in entry.get("subwords") or []] or None
    voice_over(raw, words, spec, out, samples, load_groq_key() if spec["voice"] else None, subw)
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

    words, opts["heard"] = transcribe(video, workdir, key)
    print(f"    langue entendue : {opts['heard'] or 'inconnue'} ; shorts en : {opts['lang']}")
    if opts["mode"] == "recap":
        clips = select_recaps(words, key, args.clips, opts)
    else:
        clips = select_clips(words, key, args.clips, opts["cat"], opts["find"], opts)
    iw, ih = video_dims(video)
    print(f"[4/4] rendu de {len(clips)} short(s)...")

    meta, total = [], ffprobe_duration(video)
    for i, clip in enumerate(clips, 1):
        entry = render_clip(video, words, clip, workdir, i, iw, ih,
                            use_face=not args.no_face, opts=opts, total=total, key=key)
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
    # voix off : option relue telle quelle, temps du short -> temps de l'extrait (inverse de remap)
    assert parse_opts('{"voice":true}')["voice"] and not parse_opts('{"voice":"oui"}')["voice"]
    assert auto_spec(cam, parse_opts('{"voice":true}'), 1920, 1080, 3, 20, "T")["voice"] is True
    assert parse_spec({"voice": True}, 9)["voice"] and not parse_spec({}, 9)["voice"]
    for t in (0.0, 0.5, 1.29, 1.31, 2.0):
        assert abs(remap(unremap(t, segs), segs) - t) < 1e-9, t
    assert unremap(99, segs) == segs[-1][1]
    sl = voice_slots([{"word": f"w{i}.", "start": 9.0 + i, "end": 9.5 + i} for i in range(20)], [(9.0, 30.0)], 21.0)
    assert sl[0]["t"] == 0 and abs(sum(x["d"] for x in sl) - 21.0) < .05, sl  # tout le short est couvert
    assert all(SLOT_MIN - .01 <= x["d"] for x in sl[:-1]) and all(x["d"] <= SLOT_MAX + 2 for x in sl), sl
    assert voice_slots([], [(0.0, 30.0)], 30.0)[-1]["d"] > 0  # sans paroles : commente quand meme
    en = [{"word": w} for w in "so now look at this you can see the car is gone and that is huge".split()]
    assert is_english(en) and not is_english([{"word": w} for w in ("alors maintenant regardez la voiture " * 3).split()])
    assert fit_words("So, look at this car, it is gone and that is huge news", 6) == "So, look at this car."
    assert fit_words("one two three", 6) == "one two three" and len(fit_words("a " * 30, 8).split()) == 8
    assert parse_lines("Sure:\n1. Hey there.\n2) \"Look at this\"\n4. extra", 3) == ["Hey there.", "Look at this", ""]
    # couleurs et cadrages moment par moment
    r1, r2 = {"x": .1, "y": 0, "w": .3, "h": 1}, {"x": .6, "y": 0, "w": .3, "h": 1}
    k = parse_spec({"trim": [2, 12], "crop": r1, "color": "#ffd60a", "subc": "rouge",
                    "keys": [{"t": 7, "crop": r2}, {"t": 2, "crop": r2}, {"t": "x"}, {"t": 5}]}, dur=20)
    assert k["color"] == "#FFD60A" and k["subc"] == "#FFFFFF", k
    assert [x["t"] for x in k["keys"]] == [2, 7], k["keys"]
    assert plans(k) == [(0, 5, {"crop": r2, "cam": None, "game": None, "layout": "full"}),
                        (5, 10, {"crop": r2, "cam": None, "game": None, "layout": "full"})], plans(k)
    # mise en page par plan : explicite, ou deduite des cadres d'une ancienne cle
    m = parse_spec({"trim": [0, 10], "keys": [{"t": 3, "cam": r1, "game": r2}, {"t": 6, "crop": r2},
                                              {"t": 8, "crop": r1, "layout": "split"}]}, dur=10)
    assert [r["layout"] for *_, r in plans(m)] == ["full", "split", "full", "split"], plans(m)
    k["keys"] = [{"t": 6, "crop": r2}]
    assert [(a, b, r["crop"]["x"]) for a, b, r in plans(k)] == [(0, 4, .1), (4, 10, .6)]
    o = parse_opts('{"dmin": 61, "dmax": 200, "loop": false}')
    assert (o["dmin"], o["dmax"], o["loop"]) == (61, 90, False), o
    assert (parse_opts('{"dmin": 80, "dmax": 82}')["dmin"], parse_opts("{}")["loop"]) == (DMIN_DEF, True)
    assert parse_opts("{}")["hook"] and not parse_opts('{"hook":false}')["hook"]
    # ouverture sur la reaction : "oh my gosh" a 20 s d'un short qui commence a 0
    hk = [{"word": w, "start": i * .5, "end": i * .5 + .4} for i, w in enumerate(("blah " * 40 + "Oh my GOSH! blah").split())]
    sp_ = hook_span(hk, {"start": 0, "end": 30, "hook_text": "oh my gosh"})
    assert sp_ and abs(sp_[0] - 19.85) < .01 and abs(sp_[1] - 21.75) < .01, sp_
    assert hook_span(hk, {"start": 19, "end": 30, "hook_text": "oh my gosh"}) is None  # deja au debut
    assert hook_span(hk, {"start": 0, "end": 30, "hook_text": "nothing like it"}) is None
    sw_ = shift_words(hk, 0, 5, sp_, 30)
    T_ = sp_[1] - sp_[0]
    assert [w["word"] for w in sw_ if 5 <= w["start"] < 5 + T_] == ["Oh", "my", "GOSH!"], sw_
    assert all(a["start"] <= b["start"] for a, b in zip(sw_, sw_[1:])), "mots en ordre"
    # sous-titres mot par mot : un evenement par mot, le mot dit est colore
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "s.ass"
        build_ass(hk[:3], [(0, 2)], p)
        ev = [l for l in p.read_text(encoding="utf-8").splitlines() if l.startswith("Dialogue: 0")]
        assert len(ev) == 3 and ",,BLAH {" in ev[1] and ev[1].count("fscx110") == 1, ev  # 2e mot allume
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
    assert t["bleed"] and theme_windows(t) == THEMES["neon"]["bfull"]  # plein ecran par defaut
    assert theme_windows({**t, "bleed": False}) == THEMES["neon"]["full"]
    assert theme_windows({**t, "layout": "split", "cam": r1, "game": r2, "bleed": False}) == THEMES["neon"]["split"]
    assert parse_opts('{"bleed":false}')["bleed"] is False and parse_spec({"bleed": 0}, 9)["bleed"]
    assert theme_windows({**t, "format": "4:5"}) is None and theme_windows({**t, "theme": "none"}) is None
    assert len(theme_windows({**t, "theme": "versus"})) == 2  # toujours deux plans
    assert all(w[2] % 2 == 0 and w[3] % 2 == 0 for th in THEMES.values()
               for k in ("full", "split", "bfull", "bsplit") for w in th[k])
    assert all(set(th["sub"]) == {"full", "split", "bfull", "bsplit"} for th in THEMES.values())
    # realisateur : un sujet au centre -> plein cadre ; deux sujets eloignes -> partage
    try:
        import cv2
        import numpy as np
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            def clip_of(xs):
                p = pathlib.Path(td) / f"d{len(xs)}.avi"
                vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
                for i in range(90):
                    img = np.full((360, 640, 3), 40, np.uint8)
                    for x in xs:
                        cv2.circle(img, (int(x * 640), 180 + int(40 * math.sin(i / 5))), 45, (255, 255, 255), -1)
                    vw.write(img)
                vw.release()
                return director(p, 0, 3, 640, 360, parse_spec({}, 3))
            one, two = clip_of([.5]), clip_of([.12, .88])
            assert one and one[0][1] == "full" and abs(one[0][2]["crop"]["x"] + one[0][2]["crop"]["w"] / 2 - .5) < .1, one
            assert two and two[0][1] == "split", two
    except ImportError:
        print("[info] OpenCV absent : realisateur non teste")
    # plugin GTA + mode campagne
    h = gta_hints("[00:05] (5s) oh the cops are chasing me\n[00:09] (9s) if I crashed it would be bad")
    assert "5s Chase" in h and "9s" not in h, h  # l'hypothese est ecartee
    o = parse_opts('{"camp":"tomo","voice":true,"cat":"gta"}')
    assert o["camp"] == "tomo" and o["voice"] is False and o["cat"] == "gta"
    assert parse_opts('{"camp":"x"}')["camp"] == ""
    assert base_spec([], o, 1920, 1080, 0, 9, "T")["cta"].startswith("SUBSCRIBE")
    assert parse_spec({"cta": "HI"}, 9)["cta"] == "HI" and parse_spec({}, 9)["cta"] == ""
    pc = pathlib.Path(tempfile.mkdtemp()) / "c.ass"
    build_ass([], [(0, 2)], pc, cta="WATCH")
    assert ",Cta,,0,0,0,,WATCH" in pc.read_text(encoding="utf-8")
    # cadre de facecam detecte : le cadrage du haut reste dedans, le jeu l'evite
    fb = (0.698, 0.426, 0.302, 0.315)
    camf = [(i * .5, .85, .62, .07) for i in range(10)]
    w_, h_, x_, y_ = facecam_crop(camf, 1920, 1080, 1080 / 652, fb)
    assert x_ >= 0.698 * 1920 and x_ + w_ <= 1920 and y_ >= .426 * 1080 and y_ + h_ <= .741 * 1080 + 1, (w_, h_, x_, y_)
    gw_, _, gx_, _ = game_crop(1920, 1080, 1080 / 1268, fb)
    assert gx_ + gw_ <= 0.698 * 1920 + 1, (gw_, gx_)
    pt = pathlib.Path(tempfile.mkdtemp()) / "t.ass"
    build_ass([], [(0, 2)], pt, title="T", cta="C", top=652)
    assert ",8,90,90,676" in pt.read_text(encoding="utf-8")  # titre juste sous la facecam
    # son coupe dans le meme graphe que l'image (sinon decalage bouche / sous-titres)
    ac = audio_cut([(5, 7), (9, 10)], 5)
    assert ac == ("asplit=2[x0][x1];[x0]atrim=0.000:2.000,asetpts=PTS-STARTPTS[y0];"
                  "[x1]atrim=4.000:5.000,asetpts=PTS-STARTPTS[y1];[y0][y1]concat=n=2:v=0:a=1"), ac
    fc = fold_audio(["ffmpeg", "-filter_complex", "[0:v]null[v]", "-map", "[v]", "-map", "0:a?",
                     "-af", "aselect='x'", "o.mp4"], ["-af", "aselect='x'"])
    assert "-af" not in fc and fc[2].endswith(";[0:a]aselect='x'[aout]") and "[aout]" in fc, fc
    assert fold_audio(["ffmpeg", "-vf", "null", "-af", "a"], ["-af", "a"]) == ["ffmpeg", "-vf", "null", "-af", "a"]
    moving = [(i * .5, .2 + (i % 3) * .3, .4, .1) for i in range(10)]  # visages du jeu qui bougent
    assert plan_layout(moving, parse_opts("{}"), 1920, 1080, (.7, .4, .3, .3)) == "split"
    assert in_box((0, .85, .6, .05), CAMPAIGNS["tomo"]["box"]) and not in_box((0, .3, .4, .1), CAMPAIGNS["tomo"]["box"])
    # langue et mode resume (2026-11-24)
    o = parse_opts('{"lang":"fr","mode":"recap"}')
    assert o["lang"] == "fr" and o["mode"] == "recap" and o["hook"] is False
    assert parse_opts('{"lang":"xx"}')["lang"] == "en" and parse_opts("{}")["mode"] == "clips"
    assert parse_opts('{"lang":"fr","camp":"tomo"}')["lang"] == "en"
    assert "FRENCH" in lang_rule("fr") and "ENGLISH" in lang_rule("zz")
    assert spoken_lang("French", []) == "fr" and spoken_lang("es", []) == "es"
    assert parse_spec({"lang": "es"}, 9)["lang"] == "es" and parse_spec({}, 9)["lang"] == "en"
    ws = [{"word": f"w{i}", "start": i * .4, "end": i * .4 + .3} for i in range(400)]  # 160 s de parole continue
    ws[50]["word"] = "fin."
    assert [len(g) for g in phrases(ws[:60])][:4] == [14, 14, 14, 9]  # 14 mots max, coupe apres "fin."
    sp = spread("un deux trois", 10, 13)
    assert sp[0]["start"] == 10 and sp[-1]["end"] == 13 and all(a["end"] == b["start"] for a, b in zip(sp, sp[1:]))
    assert translate_words(ws, "fr", "fr", "k") is None  # meme langue : rien a traduire
    ro = {**parse_opts('{"mode":"recap","dmin":40,"dmax":60,"cut":false}')}
    fp = fit_parts([(0, 10), (10.1, 20), (50, 70), (90, 100), (120, 140)], ws, ro, 0, 161)
    tot = sum(b - a for a, b in fp)
    assert 40 <= tot <= 60 + .5 and all(b <= c for (_, b), (c, _) in zip(fp, fp[1:])), (fp, tot)
    assert fp[0][0] < 1 and len(fp) >= 2  # (0,10) et (10.1,20) recolles en un seul passage
    fp = fit_parts([(0, 8)], ws, ro, 0, 161)
    assert sum(b - a for a, b in fp) >= 40, fp  # trop court : prolonge jusqu'a la duree mini
    sh = shift_parts(ws, [(0, 2), (10, 12)])
    assert sh[0]["start"] == 0 and min(w["start"] for w in sh if w["word"] == "w25") == 2.0, sh[:8]
    cb = chapter_bounds(ws, 2)
    assert len(cb) == 2 and 20 < cb[0][1] < 140 and cb[1][0] == cb[0][1]
    # fin de video : le debut recule pour atteindre la duree mini
    tail = [{"word": ("fin." if i % 10 == 9 else f"m{i}"), "start": i * .4, "end": i * .4 + .3} for i in range(200)]
    fc = fit_clip({"start": 70.0, "end": 80.0, "title": "t", "score": 1}, tail, {**ro, "loop": False})
    d = fc["end"] - fc["start"]
    assert 40 <= d <= 60 and fc["end"] > 79 and tail[int(fc["start"] / .4 + .5) - 1]["word"] == "fin.", fc
    assert tags_str(["a", "#b c"]) == "#a #bc" and tags_str("x y") == "#x #y" and tags_str(None) == ""
    print("selfcheck OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
