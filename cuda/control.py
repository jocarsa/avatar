import os
import re
import time
import math
import socket
import random
import tempfile
import threading
from pathlib import Path

import requests
import ttkbootstrap as tb
from ttkbootstrap.constants import BOTH, LEFT, RIGHT, X, Y, VERTICAL
from tkinter import Canvas, END

from gtts import gTTS
from pydub import AudioSegment
import pygame
import speech_recognition as sr


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
SHAPES_DIR = Path("shapes")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "phi4-mini:latest"

SYSTEM_PROMPT_DEFAULT = """Eres un asistente conversacional para un avatar 3D.
Hablas siempre en español.
Tus respuestas deben sonar naturales al ser leídas en voz alta.
Usa frases relativamente cortas y claras.
Evita listas largas y párrafos excesivamente extensos.
No uses markdown, ni emojis, ni formatos especiales.
Si la pregunta es técnica, responde con precisión pero de forma comprensible.
Si no sabes algo, dilo claramente sin inventar.
"""

TTS_LANG = "es"
TTS_TLD_DEFAULT = "es"
TTS_SLOW_DEFAULT = False
PLAYBACK_VOLUME_DEFAULT = 1.0
POST_SPEECH_SILENCE = 0.15

LISTEN_TIMEOUT = None
PHRASE_TIME_LIMIT = 8
AMBIENT_CALIBRATION = 0.6

ANIMATION_REFRESH_MS = 30

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# ------------------------------------------------------------
# SHAPES
# ------------------------------------------------------------
def list_shape_files():
    if not SHAPES_DIR.exists() or not SHAPES_DIR.is_dir():
        return []
    return sorted(
        [p for p in SHAPES_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".obj"],
        key=lambda p: p.name.lower()
    )

shape_files = list_shape_files()
shape_names = [p.stem for p in shape_files]
shape_map = {name.lower(): idx for idx, name in enumerate(shape_names)}


# ------------------------------------------------------------
# AUDIO
# ------------------------------------------------------------
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)


# ------------------------------------------------------------
# APP
# ------------------------------------------------------------
app = tb.Window(themename="darkly")
app.title("Morph Controller + Always Listening")
app.geometry("980x1100")

title = tb.Label(
    app,
    text="Dynamic Shape Morph Controller",
    font=("Segoe UI", 16, "bold"),
    bootstyle="light"
)
title.pack(pady=(12, 6))

info = tb.Label(
    app,
    text=(
        f"Shapes detected: {len(shape_names)}\n"
        f"Folder: {SHAPES_DIR.resolve()}\n"
        f"Ollama model: {OLLAMA_MODEL}"
    ),
    font=("Segoe UI", 10),
    bootstyle="secondary"
)
info.pack(pady=(0, 10))

outer = tb.Frame(app, padding=10)
outer.pack(fill=BOTH, expand=True)

canvas = Canvas(outer, highlightthickness=0)
scrollbar = tb.Scrollbar(outer, orient=VERTICAL, command=canvas.yview, bootstyle="round")
scrollable = tb.Frame(canvas)

def _on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scrollable.bind("<Configure>", _on_frame_configure)

window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

def _resize_inner(event):
    canvas.itemconfig(window_id, width=event.width)

canvas.bind("<Configure>", _resize_inner)

canvas.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill=Y)

slider_vars = []
value_labels = []
is_programmatic_update = False
blink_running = True

assistant_state = {
    "running": True,
    "listening_enabled": True,
    "speaking": False,
    "processing": False,
}

state_lock = threading.Lock()

# per-slider animation metadata
slider_meta = []


# ------------------------------------------------------------
# UDP
# ------------------------------------------------------------
def send_all_weights(*_):
    values = [f"{var.get():.6f}" for var in slider_vars]
    msg = ";".join(values).encode("utf-8")
    sock.sendto(msg, (UDP_IP, UDP_PORT))

def set_slider_index_value(idx, value, send_now=True):
    global is_programmatic_update
    if idx < 0 or idx >= len(slider_vars):
        return False

    value = max(0.0, min(1.0, float(value)))

    is_programmatic_update = True
    slider_vars[idx].set(value)
    value_labels[idx].config(text=f"{value:.2f}")
    is_programmatic_update = False

    if send_now:
        send_all_weights()
    return True

def set_channel_value(channel_name, value, send_now=True):
    idx = shape_map.get(channel_name.lower())
    if idx is None:
        return False
    return set_slider_index_value(idx, value, send_now=send_now)

def set_multiple_channels(mapping, send_now=True):
    global is_programmatic_update
    changed = False

    is_programmatic_update = True
    for channel_name, value in mapping.items():
        idx = shape_map.get(channel_name.lower())
        if idx is not None:
            value = max(0.0, min(1.0, float(value)))
            slider_vars[idx].set(value)
            value_labels[idx].config(text=f"{value:.2f}")
            changed = True
    is_programmatic_update = False

    if changed and send_now:
        send_all_weights()

    return changed

def clear_vowel_channels(send_now=True):
    mapping = {}
    for v in ("a", "e", "i", "o", "u"):
        if v in shape_map:
            mapping[v] = 0.0
    if mapping:
        set_multiple_channels(mapping, send_now=send_now)

def set_vowel_weight(vowel, weight, send_now=True):
    weight = max(0.0, min(1.0, float(weight)))
    mapping = {}

    for v in ("a", "e", "i", "o", "u"):
        if v in shape_map:
            mapping[v] = 0.0

    if vowel in shape_map:
        mapping[vowel] = weight
    elif "a" in shape_map:
        mapping["a"] = weight
    else:
        return False

    return set_multiple_channels(mapping, send_now=send_now)

def reset_all():
    global is_programmatic_update
    is_programmatic_update = True
    for i, (var, lbl) in enumerate(zip(slider_vars, value_labels)):
        var.set(0.0)
        lbl.config(text="0.00")
        if i < len(slider_meta):
            slider_meta[i]["base_value"] = 0.0
            slider_meta[i]["last_value"] = 0.0
    is_programmatic_update = False
    send_all_weights()


# ------------------------------------------------------------
# BLINK
# ------------------------------------------------------------
blink_enabled = "ojos cerrados" in shape_map

def do_blink():
    if not blink_enabled or not blink_running:
        return
    set_channel_value("ojos cerrados", 1.0, send_now=True)
    app.after(random.randint(90, 170), end_blink)

def end_blink():
    if not blink_enabled or not blink_running:
        return
    set_channel_value("ojos cerrados", 0.0, send_now=True)
    app.after(random.randint(1800, 5200), do_blink)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
VOWEL_GROUPS = {
    "a": set("aáàäâAÁÀÄÂ"),
    "e": set("eéèëêEÉÈËÊ"),
    "i": set("iíìïîIÍÌÏÎ"),
    "o": set("oóòöôOÓÒÖÔ"),
    "u": set("uúùüûUÚÙÜÛ"),
}

def normalize_vowel(ch):
    for base, group in VOWEL_GROUPS.items():
        if ch in group:
            return base
    return None

def find_nearest_vowel(text, center_index, radius=6):
    if not text:
        return None

    start = max(0, center_index - radius)
    end = min(len(text), center_index + radius + 1)

    best = None
    best_dist = 10**9

    for i in range(start, end):
        v = normalize_vowel(text[i])
        if v is not None:
            dist = abs(i - center_index)
            if dist < best_dist:
                best_dist = dist
                best = v

    return best

def safe_db_to_linear(db_value):
    if db_value == float("-inf"):
        return 0.0
    return 10 ** (db_value / 20.0)

def load_audio_envelope_and_duration(audio_path, frame_ms=35):
    audio = AudioSegment.from_file(audio_path)
    duration_sec = len(audio) / 1000.0

    chunks = []
    for start in range(0, len(audio), frame_ms):
        chunks.append(audio[start:start + frame_ms])

    if not chunks:
        return [], duration_sec

    raw_levels = []
    for chunk in chunks:
        raw_levels.append(safe_db_to_linear(chunk.dBFS))

    max_level = max(raw_levels) if raw_levels else 1.0
    if max_level <= 0.0:
        max_level = 1.0

    envelope = []
    prev = 0.0
    attack = 0.55
    release = 0.25

    for lv in raw_levels:
        norm = lv / max_level
        if norm < 0.08:
            norm = 0.0
        norm = math.sqrt(norm)

        if norm > prev:
            smooth = prev + (norm - prev) * attack
        else:
            smooth = prev + (norm - prev) * release

        smooth = max(0.0, min(1.0, smooth))
        envelope.append(smooth)
        prev = smooth

    return envelope, duration_sec

def amplitude_to_mouth_weight(a):
    if a <= 0.6:
        return 0.0
    w = 0.12 + (a ** 0.85) * 0.88
    return max(0.0, min(1.0, w))

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ------------------------------------------------------------
# SETTINGS UI
# ------------------------------------------------------------
settings_frame = tb.Labelframe(app, text="AI / TTS Settings", padding=10, bootstyle="info")
settings_frame.pack(fill=X, padx=10, pady=(0, 10))

tb.Label(settings_frame, text="System prompt", bootstyle="light").pack(anchor="w")
system_prompt_text = tb.Text(settings_frame, height=7, wrap="word", font=("Segoe UI", 10))
system_prompt_text.pack(fill=X, pady=(0, 10))
system_prompt_text.insert("1.0", SYSTEM_PROMPT_DEFAULT)

tts_row = tb.Frame(settings_frame)
tts_row.pack(fill=X)

tts_slow_var = tb.BooleanVar(value=TTS_SLOW_DEFAULT)
tb.Checkbutton(
    tts_row,
    text="Slow speech",
    variable=tts_slow_var,
    bootstyle="round-toggle"
).pack(side=LEFT, padx=(0, 16))

tb.Label(tts_row, text="Accent / tld", bootstyle="light").pack(side=LEFT, padx=(0, 8))
tts_tld_var = tb.StringVar(value=TTS_TLD_DEFAULT)
tb.Entry(tts_row, textvariable=tts_tld_var, width=8).pack(side=LEFT, padx=(0, 16))

tb.Label(tts_row, text="Volume", bootstyle="light").pack(side=LEFT, padx=(0, 8))
tts_volume_var = tb.DoubleVar(value=PLAYBACK_VOLUME_DEFAULT)
tb.Scale(
    tts_row,
    from_=0.0,
    to=1.0,
    variable=tts_volume_var,
    bootstyle="info"
).pack(side=LEFT, fill=X, expand=True)


# ------------------------------------------------------------
# SLIDERS UI
# ------------------------------------------------------------
if not shape_names:
    tb.Label(
        scrollable,
        text="No .obj files found in ./shapes",
        bootstyle="danger"
    ).pack(pady=20)
else:
    for idx, name in enumerate(shape_names):
        card = tb.Frame(scrollable, padding=10, bootstyle="secondary")
        card.pack(fill=X, padx=4, pady=5)

        top = tb.Frame(card)
        top.pack(fill=X)

        name_label = tb.Label(
            top,
            text=name,
            font=("Segoe UI", 11, "bold"),
            bootstyle="light"
        )
        name_label.pack(side=LEFT)

        value_label = tb.Label(
            top,
            text="0.00",
            font=("Segoe UI", 10),
            bootstyle="info"
        )
        value_label.pack(side=RIGHT)

        slider_var = tb.DoubleVar(value=0.0)
        slider_vars.append(slider_var)
        value_labels.append(value_label)

        row = tb.Frame(card)
        row.pack(fill=X, pady=(8, 0))

        meta = {
            "name": name,
            "index": idx,
            "base_value": 0.0,
            "last_value": 0.0,
            "anim_enabled_var": tb.BooleanVar(value=False),
            "anim_period_var": tb.DoubleVar(value=10.0),
        }
        slider_meta.append(meta)

        def make_callback(lbl, meta_ref):
            def _callback(value):
                v = float(value)
                lbl.config(text=f"{v:.2f}")
                meta_ref["last_value"] = v
                if not meta_ref["anim_enabled_var"].get():
                    meta_ref["base_value"] = v
                if not is_programmatic_update:
                    send_all_weights()
            return _callback

        scale = tb.Scale(
            row,
            from_=0.0,
            to=1.0,
            variable=slider_var,
            command=make_callback(value_label, meta),
            bootstyle="info"
        )
        scale.pack(fill=X)

        anim_row = tb.Frame(card)
        anim_row.pack(fill=X, pady=(8, 0))

        def on_anim_toggle(meta_ref=meta):
            if meta_ref["anim_enabled_var"].get():
                meta_ref["base_value"] = float(slider_vars[meta_ref["index"]].get())
            else:
                meta_ref["base_value"] = float(slider_vars[meta_ref["index"]].get())

        tb.Checkbutton(
            anim_row,
            text="Animate",
            variable=meta["anim_enabled_var"],
            command=on_anim_toggle,
            bootstyle="round-toggle"
        ).pack(side=LEFT, padx=(0, 14))

        tb.Label(anim_row, text="Animation time (s)", bootstyle="light").pack(side=LEFT, padx=(0, 8))

        tb.Entry(
            anim_row,
            textvariable=meta["anim_period_var"],
            width=8
        ).pack(side=LEFT)


# ------------------------------------------------------------
# CONVERSATION UI
# ------------------------------------------------------------
io_frame = tb.Labelframe(app, text="Conversation", padding=10, bootstyle="info")
io_frame.pack(fill=X, padx=10, pady=(0, 10))

recognized_title = tb.Label(io_frame, text="Recognized speech", bootstyle="light")
recognized_title.pack(anchor="w")

recognized_var = tb.StringVar(value="")
recognized_label = tb.Label(
    io_frame,
    textvariable=recognized_var,
    bootstyle="secondary",
    wraplength=900,
    justify="left"
)
recognized_label.pack(fill=X, pady=(0, 10), anchor="w")

answer_title = tb.Label(io_frame, text="Ollama answer", bootstyle="light")
answer_title.pack(anchor="w")

answer_text = tb.Text(io_frame, height=10, wrap="word", font=("Segoe UI", 10))
answer_text.pack(fill=X, pady=(0, 10))

status_var = tb.StringVar(value="Ready")
status_label = tb.Label(io_frame, textvariable=status_var, bootstyle="warning")
status_label.pack(anchor="w", pady=(0, 6))


# ------------------------------------------------------------
# OLLAMA
# ------------------------------------------------------------
def ask_ollama(question):
    system_prompt = system_prompt_text.get("1.0", END).strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": question,
        "system": system_prompt,
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


# ------------------------------------------------------------
# TTS + AVATAR
# ------------------------------------------------------------
def stop_audio():
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
    clear_vowel_channels(send_now=True)
    with state_lock:
        assistant_state["speaking"] = False
        assistant_state["processing"] = False
        assistant_state["listening_enabled"] = True
    status_var.set("Listening...")

def speak_with_avatar(text):
    text = clean_text(text)
    if not text:
        with state_lock:
            assistant_state["speaking"] = False
            assistant_state["processing"] = False
            assistant_state["listening_enabled"] = True
        return

    def worker():
        audio_path = None
        try:
            status_var.set("Generating audio...")

            current_tld = tts_tld_var.get().strip() or "es"
            current_slow = bool(tts_slow_var.get())
            current_volume = max(0.0, min(1.0, float(tts_volume_var.get())))

            tts = gTTS(
                text=text,
                lang=TTS_LANG,
                tld=current_tld,
                slow=current_slow
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                audio_path = tmp.name

            tts.save(audio_path)

            envelope, duration = load_audio_envelope_and_duration(audio_path, frame_ms=35)
            if duration <= 0:
                duration = 1.0

            status_var.set("Speaking...")
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.set_volume(current_volume)
            pygame.mixer.music.play()

            frame_count = max(1, len(envelope))
            frame_duration = duration / frame_count
            start = time.time()

            while pygame.mixer.music.get_busy():
                elapsed = time.time() - start
                frame_index = int(elapsed / frame_duration)
                frame_index = max(0, min(frame_count - 1, frame_index))

                amp = envelope[frame_index]
                mouth_weight = amplitude_to_mouth_weight(amp)

                if text:
                    text_pos = int((elapsed / duration) * len(text))
                    text_pos = max(0, min(len(text) - 1, text_pos))
                    vowel = find_nearest_vowel(text, text_pos, radius=6)
                else:
                    vowel = None

                if mouth_weight <= 0.01:
                    clear_vowel_channels(send_now=True)
                else:
                    if vowel is None:
                        vowel = "a"
                    set_vowel_weight(vowel, mouth_weight, send_now=True)

                time.sleep(0.02)

            time.sleep(POST_SPEECH_SILENCE)
            clear_vowel_channels(send_now=True)

        except Exception as e:
            clear_vowel_channels(send_now=True)
            status_var.set(f"Audio/TTS error: {e}")

        finally:
            try:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception:
                pass

            with state_lock:
                assistant_state["speaking"] = False
                assistant_state["processing"] = False
                assistant_state["listening_enabled"] = True

            status_var.set("Listening...")

    threading.Thread(target=worker, daemon=True).start()


# ------------------------------------------------------------
# STT LOOP
# ------------------------------------------------------------
recognizer = sr.Recognizer()
microphone = None

def process_user_text(user_text):
    user_text = clean_text(user_text)
    if not user_text:
        return

    with state_lock:
        if assistant_state["processing"] or assistant_state["speaking"]:
            return
        assistant_state["processing"] = True
        assistant_state["listening_enabled"] = False

    recognized_var.set(user_text)
    status_var.set("Thinking...")

    def worker():
        try:
            answer = ask_ollama(user_text)
            if not answer:
                answer = "No tengo respuesta en este momento."

            answer_text.delete("1.0", END)
            answer_text.insert("1.0", answer)

            with state_lock:
                assistant_state["speaking"] = True

            speak_with_avatar(answer)

        except Exception as e:
            with state_lock:
                assistant_state["speaking"] = False
                assistant_state["processing"] = False
                assistant_state["listening_enabled"] = True
            status_var.set(f"Ollama error: {e}")

    threading.Thread(target=worker, daemon=True).start()

def listen_loop():
    global microphone
    try:
        microphone = sr.Microphone()
    except Exception as e:
        status_var.set(f"Microphone init error: {e}")
        return

    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_CALIBRATION)
    except Exception as e:
        status_var.set(f"Mic calibration error: {e}")
        return

    status_var.set("Listening...")

    while True:
        with state_lock:
            running = assistant_state["running"]
            listening_enabled = assistant_state["listening_enabled"]
            speaking = assistant_state["speaking"]
            processing = assistant_state["processing"]

        if not running:
            break

        if not listening_enabled or speaking or processing:
            time.sleep(0.05)
            continue

        try:
            with microphone as source:
                audio = recognizer.listen(
                    source,
                    timeout=LISTEN_TIMEOUT,
                    phrase_time_limit=PHRASE_TIME_LIMIT
                )

            with state_lock:
                if not assistant_state["listening_enabled"]:
                    continue

            status_var.set("Recognizing...")
            text = recognizer.recognize_google(audio, language="es-ES").strip()

            if text:
                process_user_text(text)

        except sr.UnknownValueError:
            status_var.set("Listening...")
            continue
        except sr.WaitTimeoutError:
            status_var.set("Listening...")
            continue
        except sr.RequestError as e:
            status_var.set(f"Speech recognition error: {e}")
            time.sleep(1.0)
        except Exception as e:
            status_var.set(f"Listen loop error: {e}")
            time.sleep(1.0)


# ------------------------------------------------------------
# CONTINUOUS SLIDER ANIMATION
# ------------------------------------------------------------
def animation_tick():
    try:
        now = time.time()
        any_changed = False

        for meta in slider_meta:
            if not meta["anim_enabled_var"].get():
                continue

            idx = meta["index"]
            base = float(meta["base_value"])
            period = float(meta["anim_period_var"].get())

            if period <= 0.001:
                period = 10.0

            phase = (now % period) / period
            sine01 = 0.5 * (1.0 + math.sin(2.0 * math.pi * phase))
            value = base * sine01

            current = float(slider_vars[idx].get())
            if abs(current - value) > 0.002:
                set_slider_index_value(idx, value, send_now=False)
                any_changed = True

        if any_changed:
            send_all_weights()

    except Exception:
        pass

    if assistant_state["running"]:
        app.after(ANIMATION_REFRESH_MS, animation_tick)


# ------------------------------------------------------------
# CONTROL BUTTONS
# ------------------------------------------------------------
buttons_frame = tb.Frame(app, padding=(10, 0, 10, 10))
buttons_frame.pack(fill=X)

tb.Button(
    buttons_frame,
    text="Reset all",
    command=reset_all,
    bootstyle="warning-outline"
).pack(side=LEFT, padx=(0, 8))

tb.Button(
    buttons_frame,
    text="Stop audio",
    command=stop_audio,
    bootstyle="danger-outline"
).pack(side=LEFT, padx=(0, 8))


# ------------------------------------------------------------
# STARTUP
# ------------------------------------------------------------
def send_zero_once():
    send_all_weights()

app.after(100, send_zero_once)

if blink_enabled:
    app.after(random.randint(1200, 3000), do_blink)

app.after(ANIMATION_REFRESH_MS, animation_tick)
threading.Thread(target=listen_loop, daemon=True).start()


# ------------------------------------------------------------
# CLEAN EXIT
# ------------------------------------------------------------
def on_close():
    global blink_running
    blink_running = False

    with state_lock:
        assistant_state["running"] = False
        assistant_state["listening_enabled"] = False
        assistant_state["speaking"] = False
        assistant_state["processing"] = False

    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

    try:
        pygame.mixer.quit()
    except Exception:
        pass

    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)
app.mainloop()
