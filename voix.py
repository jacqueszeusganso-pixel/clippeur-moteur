# Voix off du clippeur maison : lit un texte anglais avec la voix clonee de Jacques (Chatterbox, licence MIT).
# Usage : python voix.py REF.wav SORTIE_PREFIXE "texte"  -> SORTIE_PREFIXE_<variante>.wav
# L'echantillon de Jacques est en francais : clone direct = accent francais herite.
# Variante "vc" : une voix americaine native dit le texte (diction, accent, energie),
# puis la conversion de voix remplace seulement le timbre par celui de Jacques.
import sys, time, torchaudio
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC

ref, out, text = sys.argv[1], sys.argv[2], sys.argv[3]


def step(name, fn):
    t = time.time()
    wav = fn()
    torchaudio.save(f"{out}_{name}.wav", wav, tts.sr)
    print(f"{name}: {wav.shape[-1]/tts.sr:.1f}s d'audio en {time.time()-t:.0f}s", flush=True)


tts = ChatterboxTTS.from_pretrained(device="cpu")
vc = ChatterboxVC.from_pretrained(device="cpu")
# 1. clone direct, cfg 0 : limite l'heritage d'accent de l'echantillon (conseil Chatterbox)
step("clone_cfg0", lambda: tts.generate(text, audio_prompt_path=ref, exaggeration=0.8, cfg_weight=0))
# 2. voix americaine par defaut de Chatterbox, enthousiaste (source de la conversion)
step("source_us", lambda: tts.generate(text, exaggeration=0.8, cfg_weight=0.3))
# 3. conversion : accent americain de la source, timbre de Jacques
step("vc", lambda: vc.generate(f"{out}_source_us.wav", target_voice_path=ref))
