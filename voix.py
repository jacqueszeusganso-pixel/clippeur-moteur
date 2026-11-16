# Voix off du clippeur maison : lit un texte anglais avec la voix clonee de Jacques (Chatterbox, licence MIT).
# Usage : python voix.py REF.wav SORTIE_PREFIXE "texte"  -> SORTIE_PREFIXE_<style>.wav pour chaque style
import sys, time, torchaudio
from chatterbox.tts import ChatterboxTTS

# (nom, exaggeration, cfg_weight) : exaggeration haute = plus d'energie, cfg bas = debit plus vif
STYLES = [("enthousiaste", 0.8, 0.3), ("neutre", 0.5, 0.5)]

ref, out, text = sys.argv[1], sys.argv[2], sys.argv[3]
t = time.time()
m = ChatterboxTTS.from_pretrained(device="cpu")
print(f"modele charge en {time.time()-t:.0f}s", flush=True)
for name, exa, cfg in STYLES:
    t = time.time()
    wav = m.generate(text, audio_prompt_path=ref, exaggeration=exa, cfg_weight=cfg)
    torchaudio.save(f"{out}_{name}.wav", wav, m.sr)
    print(f"{name}: {wav.shape[-1]/m.sr:.1f}s d'audio en {time.time()-t:.0f}s", flush=True)
