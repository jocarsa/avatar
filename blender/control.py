import json
import urllib.request
import tkinter as tk
from tkinter import ttk

# ============================================================
# CONFIGURATION
# ============================================================
BLENDER_URL = "http://127.0.0.1:8765/update"
TARGET_OBJECT = "avatar"


# ============================================================
# SEND DATA TO BLENDER
# ============================================================
def send_to_blender():
    payload = {
        "object": TARGET_OBJECT,
        "params": {
            "ojoscerrados": slider_ojos.get(),
            "sonrisa": slider_sonrisa.get(),
            "a": slider_a.get(),
            "cabezaabajo": slider_cabeza.get(),
            "rot_x_deg": slider_rot_x.get(),
            "rot_y_deg": slider_rot_y.get(),
            "rot_z_deg": slider_rot_z.get()
        }
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        BLENDER_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=1) as response:
            response.read()
        status_var.set("Sent to Blender")
    except Exception as e:
        status_var.set(f"Error: {e}")


def update_labels():
    label_ojos_value.config(text=f"{slider_ojos.get():.2f}")
    label_sonrisa_value.config(text=f"{slider_sonrisa.get():.2f}")
    label_a_value.config(text=f"{slider_a.get():.2f}")
    label_cabeza_value.config(text=f"{slider_cabeza.get():.2f}")

    label_rot_x_value.config(text=f"{slider_rot_x.get():.2f}°")
    label_rot_y_value.config(text=f"{slider_rot_y.get():.2f}°")
    label_rot_z_value.config(text=f"{slider_rot_z.get():.2f}°")


def on_slider_change(_=None):
    update_labels()
    send_to_blender()


def reset_all():
    slider_ojos.set(0.0)
    slider_sonrisa.set(0.0)
    slider_a.set(0.0)
    slider_cabeza.set(0.0)

    slider_rot_x.set(0.0)
    slider_rot_y.set(0.0)
    slider_rot_z.set(0.0)

    update_labels()
    send_to_blender()


# ============================================================
# UI
# ============================================================
root = tk.Tk()
root.title("Avatar Controller")
root.geometry("560x540")
root.resizable(False, False)

main = ttk.Frame(root, padding=15)
main.pack(fill="both", expand=True)

title = ttk.Label(main, text="Control avatar in Blender", font=("Arial", 13, "bold"))
title.pack(pady=(0, 12))

subtitle = ttk.Label(main, text="Object: avatar")
subtitle.pack(pady=(0, 10))

# ------------------------------------------------------------
# Shape keys
# ------------------------------------------------------------
ttk.Label(main, text="Shape Keys", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 8))

frame_ojos = ttk.Frame(main)
frame_ojos.pack(fill="x", pady=(0, 4))
ttk.Label(frame_ojos, text="ojoscerrados").pack(side="left")
label_ojos_value = ttk.Label(frame_ojos, text="0.00", width=8)
label_ojos_value.pack(side="right")
slider_ojos = ttk.Scale(main, from_=0.0, to=1.0, orient="horizontal", command=on_slider_change)
slider_ojos.set(0.0)
slider_ojos.pack(fill="x")

frame_sonrisa = ttk.Frame(main)
frame_sonrisa.pack(fill="x", pady=(12, 4))
ttk.Label(frame_sonrisa, text="sonrisa").pack(side="left")
label_sonrisa_value = ttk.Label(frame_sonrisa, text="0.00", width=8)
label_sonrisa_value.pack(side="right")
slider_sonrisa = ttk.Scale(main, from_=0.0, to=1.0, orient="horizontal", command=on_slider_change)
slider_sonrisa.set(0.0)
slider_sonrisa.pack(fill="x")

frame_a = ttk.Frame(main)
frame_a.pack(fill="x", pady=(12, 4))
ttk.Label(frame_a, text="a").pack(side="left")
label_a_value = ttk.Label(frame_a, text="0.00", width=8)
label_a_value.pack(side="right")
slider_a = ttk.Scale(main, from_=0.0, to=1.0, orient="horizontal", command=on_slider_change)
slider_a.set(0.0)
slider_a.pack(fill="x")

frame_cabeza = ttk.Frame(main)
frame_cabeza.pack(fill="x", pady=(12, 4))
ttk.Label(frame_cabeza, text="cabezaabajo").pack(side="left")
label_cabeza_value = ttk.Label(frame_cabeza, text="0.00", width=8)
label_cabeza_value.pack(side="right")
slider_cabeza = ttk.Scale(main, from_=0.0, to=1.0, orient="horizontal", command=on_slider_change)
slider_cabeza.set(0.0)
slider_cabeza.pack(fill="x")

# ------------------------------------------------------------
# Rotations
# ------------------------------------------------------------
ttk.Label(main, text="Object Rotation", font=("Arial", 10, "bold")).pack(anchor="w", pady=(18, 8))

frame_rot_x = ttk.Frame(main)
frame_rot_x.pack(fill="x", pady=(0, 4))
ttk.Label(frame_rot_x, text="rot_x_deg").pack(side="left")
label_rot_x_value = ttk.Label(frame_rot_x, text="0.00°", width=8)
label_rot_x_value.pack(side="right")
slider_rot_x = ttk.Scale(main, from_=-20.0, to=20.0, orient="horizontal", command=on_slider_change)
slider_rot_x.set(0.0)
slider_rot_x.pack(fill="x")

frame_rot_y = ttk.Frame(main)
frame_rot_y.pack(fill="x", pady=(12, 4))
ttk.Label(frame_rot_y, text="rot_y_deg").pack(side="left")
label_rot_y_value = ttk.Label(frame_rot_y, text="0.00°", width=8)
label_rot_y_value.pack(side="right")
slider_rot_y = ttk.Scale(main, from_=-20.0, to=20.0, orient="horizontal", command=on_slider_change)
slider_rot_y.set(0.0)
slider_rot_y.pack(fill="x")

frame_rot_z = ttk.Frame(main)
frame_rot_z.pack(fill="x", pady=(12, 4))
ttk.Label(frame_rot_z, text="rot_z_deg").pack(side="left")
label_rot_z_value = ttk.Label(frame_rot_z, text="0.00°", width=8)
label_rot_z_value.pack(side="right")
slider_rot_z = ttk.Scale(main, from_=-20.0, to=20.0, orient="horizontal", command=on_slider_change)
slider_rot_z.set(0.0)
slider_rot_z.pack(fill="x")

# ------------------------------------------------------------
# Buttons + status
# ------------------------------------------------------------
buttons = ttk.Frame(main)
buttons.pack(fill="x", pady=(20, 8))

ttk.Button(buttons, text="Reset All", command=reset_all).pack(side="left")

status_var = tk.StringVar(value="Ready")
status = ttk.Label(main, textvariable=status_var)
status.pack(anchor="w")

update_labels()
root.mainloop()
