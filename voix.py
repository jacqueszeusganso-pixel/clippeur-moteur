# Voix off du clippeur maison avec la voix clonee de Jacques (Chatterbox, licence MIT).
# L'echantillon de Jacques est en francais : un clone direct herite de l'accent francais.
# Donc : une voix americaine native dit le texte (diction, accent, energie), puis la
# conversion de voix remplace seulement le timbre par celui de Jacques (choix de Jacques
# le 2026-11-16 apres ecoute des tests).
# Test : python voix.py REF.wav SORTIE.wav "texte anglais"
import re
import sys
import time

_MODELS = None


def _models():
    """Charge une seule fois (environ 30 s) : un travail peut faire jusqu'a 6 shorts."""
    global _MODELS
    if _MODELS is None:
        from chatterbox.tts import ChatterboxTTS
        from chatterbox.vc import ChatterboxVC
        _MODELS = ChatterboxTTS.from_pretrained(device="cpu"), ChatterboxVC.from_pretrained(device="cpu")
    return _MODELS


def sentences(text, maxc=220):
    """Phrases de taille raisonnable : Chatterbox plafonne vers 40 s d'audio par appel."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(s) > maxc:  # phrase fleuve : coupee a la derniere virgule avant la limite
            cut = s.rfind(",", 0, maxc)
            cut = cut if cut > 40 else maxc
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out


def speak(text, ref, out, style=None):
    """Texte anglais -> WAV 24 kHz mono avec le timbre de `ref`.
    `style` : extrait d'un narrateur anglophone humain (celui de la video d'origine). Chatterbox
    imite alors sa diction (debit, intonations) au lieu de la voix par defaut, qui sonnait IA ;
    la conversion ne garde que le timbre de Jacques."""
    import torch
    import torchaudio
    tts, vc = _models()
    gap = torch.zeros(1, int(tts.sr * 0.12))
    src = f"{out}.src.wav"
    parts = []
    for s in sentences(text):
        wav = (tts.generate(s, audio_prompt_path=str(style), exaggeration=0.55, cfg_weight=0.5) if style
               else tts.generate(s, exaggeration=0.75, cfg_weight=0.3))
        torchaudio.save(src, wav, tts.sr)
        parts += [vc.generate(src, target_voice_path=ref), gap]
    torchaudio.save(str(out), torch.cat(parts, dim=1), tts.sr)
    return out


if __name__ == "__main__":
    assert sentences("Wow! This is it. Go.") == ["Wow!", "This is it.", "Go."]
    long = "a" * 50 + ", " + "b" * 200 + "."
    assert all(len(s) <= 220 for s in sentences(long)), sentences(long)
    t = time.time()
    speak(sys.argv[3], sys.argv[1], sys.argv[2])
    print(f"voix prete en {time.time() - t:.0f}s")
