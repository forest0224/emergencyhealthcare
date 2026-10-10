"""
interface.py
============
Python Tkinter GUI for the ED-Triage Clinical Decision Support API.

Usage:
    python interface.py

Requires the API server to be running:
    python run_full_pipeline.py --mode serve
    -- or --
    Start-Process -FilePath ".venv311/Scripts/python.exe" -ArgumentList \
        "-m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
"""
from __future__ import annotations

import json
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from typing import Any
import urllib.request
import urllib.error

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE  = "http://localhost:8000"
API_TOKEN = "dev-token-1"           # matches API_TOKENS in .env

# ── Colours ────────────────────────────────────────────────────────────────────
C_BG       = "#f0f4f8"
C_SURFACE  = "#ffffff"
C_BORDER   = "#cbd5e1"
C_ACCENT   = "#1d4ed8"
C_TEXT     = "#1e293b"
C_MUTED    = "#64748b"
C_RED      = "#dc2626"
C_ORANGE   = "#ea580c"
C_YELLOW   = "#ca8a04"
C_GREEN    = "#16a34a"
C_WARN_BG  = "#fef9c3"

ACUITY_COLOURS = {
    1: ("#fef2f2", C_RED,    "CRITICAL"),
    2: ("#fff7ed", C_ORANGE, "EMERGENT"),
    3: ("#fefce8", C_YELLOW, "URGENT"),
    4: ("#f0fdf4", C_GREEN,  "LESS URGENT"),
}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _post_json(endpoint: str, payload: dict) -> dict:
    """Simple urllib POST — no external dependencies."""
    url  = f"{API_BASE}{endpoint}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {API_TOKEN}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _get_json(endpoint: str) -> dict:
    url = f"{API_BASE}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _opt_float(val: str) -> float | None:
    s = val.strip()
    return float(s) if s else None


def _opt_int(val: str) -> int | None:
    s = val.strip()
    return int(s) if s else None


# ═════════════════════════════════════════════════════════════════════════════
# Main Application
# ═════════════════════════════════════════════════════════════════════════════

class TriageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ED-Triage Clinical Decision Support")
        self.geometry("980x780")
        self.minsize(820, 640)
        self.configure(bg=C_BG)
        self.resizable(True, True)

        self._bold   = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self._normal = tkfont.Font(family="Segoe UI", size=10)
        self._small  = tkfont.Font(family="Segoe UI", size=9)
        self._h1     = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self._mono   = tkfont.Font(family="Consolas",  size=9)

        self._build_ui()
        self._check_server_status()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header bar ──
        hdr = tk.Frame(self, bg=C_ACCENT, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(
            hdr, text="🏥  ED-Triage Clinical Decision Support",
            bg=C_ACCENT, fg="white",
            font=tkfont.Font(family="Segoe UI", size=13, weight="bold"),
        ).pack(side="left", padx=18, pady=12)
        self._status_lbl = tk.Label(
            hdr, text="● Checking…", bg=C_ACCENT, fg="#bfdbfe",
            font=self._small,
        )
        self._status_lbl.pack(side="right", padx=18)

        # ── Two-column body ──
        body = tk.Frame(self, bg=C_BG)
        body.pack(fill="both", expand=True, padx=14, pady=10)
        body.columnconfigure(0, weight=3, minsize=340)
        body.columnconfigure(1, weight=4, minsize=400)
        body.rowconfigure(0, weight=1)

        self._build_input_panel(body)
        self._build_results_panel(body)

        # ── Disclaimer footer ──
        tk.Label(
            self,
            text="⚠  Decision-support only. Not a diagnosis. Confirm with a licensed clinician.",
            bg=C_WARN_BG, fg=C_ORANGE, font=self._small,
            pady=5,
        ).pack(fill="x", side="bottom")

    # ── Input panel ───────────────────────────────────────────────────────────

    def _build_input_panel(self, parent):
        frame = tk.LabelFrame(
            parent, text="Patient Data",
            bg=C_SURFACE, fg=C_TEXT, font=self._bold,
            bd=1, relief="solid",
        )
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 7), pady=2)
        frame.columnconfigure(1, weight=1)

        self._fields: dict[str, tk.StringVar] = {}

        def row(label, key, row_idx, placeholder="", tooltip=""):
            tk.Label(
                frame, text=label, bg=C_SURFACE, fg=C_TEXT,
                font=self._normal, anchor="w",
            ).grid(row=row_idx, column=0, sticky="w", padx=(10, 6), pady=4)
            var = tk.StringVar(value=placeholder)
            ent = tk.Entry(
                frame, textvariable=var,
                font=self._normal, bg="#f8fafc", relief="solid", bd=1,
            )
            ent.grid(row=row_idx, column=1, sticky="ew", padx=(0, 10), pady=4)
            if tooltip:
                ent.bind("<Enter>", lambda e, t=tooltip: self._show_tip(t))
                ent.bind("<Leave>", lambda e: self._hide_tip())
            self._fields[key] = var

        section = lambda t, r: tk.Label(
            frame, text=t, bg="#e2e8f0", fg=C_MUTED,
            font=self._small, anchor="w",
        ).grid(row=r, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 2))

        section("── Demographics ──────────────────", 0)
        row("Age *",             "age",              1, "65",   "Age in years (0–120)")
        row("Gender",            "gender",           2, "M",    "'M', 'F', or leave blank")
        row("Chief Complaint *", "chief_complaint",  3, "chest pain", "Free text")

        section("── Arrival ───────────────────────", 4)
        row("Arrival Hour",      "arrival_hour",     5, "14",   "0–23 (24-h clock)")

        section("── Vital Signs ───────────────────", 6)
        row("Systolic BP",       "systolic_bp",      7, "80",   "mmHg  (40–300)")
        row("Diastolic BP",      "diastolic_bp",     8, "50",   "mmHg  (20–200)")
        row("Heart Rate",        "heart_rate",       9, "122",  "bpm   (20–300)")
        row("Resp. Rate",        "respiratory_rate", 10, "24",  "breaths/min (4–60)")
        row("Temperature (°C)",  "temperature",      11, "37.2","°C   (30–43)")
        row("SpO₂ (%)",          "spo2",             12, "93",  "% (50–100)")
        row("Pain Score",        "pain_score",       13, "8",   "0–10")

        section("── Optional ──────────────────────", 14)
        row("Encounter ID",      "encounter_id",     15, "",    "Audit linkage (hashed)")

        # ── Endpoint selector ──
        sel_frame = tk.Frame(frame, bg=C_SURFACE)
        sel_frame.grid(row=16, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 2))
        tk.Label(sel_frame, text="Endpoint:", bg=C_SURFACE, fg=C_TEXT,
                 font=self._normal).pack(side="left")
        self._endpoint = tk.StringVar(value="/v1/triage")
        for val, lbl in [("/v1/predict", "Predict (fast)"), ("/v1/triage", "Triage + Rationale")]:
            tk.Radiobutton(
                sel_frame, text=lbl, variable=self._endpoint, value=val,
                bg=C_SURFACE, fg=C_TEXT, font=self._small,
                activebackground=C_SURFACE,
            ).pack(side="left", padx=6)

        # ── Buttons ──
        btn_frame = tk.Frame(frame, bg=C_SURFACE)
        btn_frame.grid(row=17, column=0, columnspan=2, sticky="ew", padx=8, pady=10)
        self._run_btn = tk.Button(
            btn_frame, text="▶  Run Triage",
            bg=C_ACCENT, fg="white", font=self._bold,
            relief="flat", padx=14, pady=6, cursor="hand2",
            command=self._run_triage,
        )
        self._run_btn.pack(side="left", padx=(0, 8))
        tk.Button(
            btn_frame, text="⟳  Reset",
            bg=C_BORDER, fg=C_TEXT, font=self._normal,
            relief="flat", padx=10, pady=6, cursor="hand2",
            command=self._reset_fields,
        ).pack(side="left")

        # Tooltip label
        self._tip_var = tk.StringVar()
        tk.Label(frame, textvariable=self._tip_var, bg=C_SURFACE,
                 fg=C_MUTED, font=self._small).grid(
            row=18, column=0, columnspan=2, sticky="w", padx=10)

    # ── Results panel ─────────────────────────────────────────────────────────

    def _build_results_panel(self, parent):
        frame = tk.LabelFrame(
            parent, text="Results",
            bg=C_SURFACE, fg=C_TEXT, font=self._bold,
            bd=1, relief="solid",
        )
        frame.grid(row=0, column=1, sticky="nsew", padx=(7, 0), pady=2)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # ── Acuity badge ──
        self._acuity_frame = tk.Frame(frame, bg=C_SURFACE, height=70)
        self._acuity_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        self._acuity_frame.pack_propagate(False)
        self._acuity_lbl = tk.Label(
            self._acuity_frame,
            text="Submit a patient to see results",
            bg=C_SURFACE, fg=C_MUTED, font=self._h1,
        )
        self._acuity_lbl.pack(expand=True)

        # ── Tabs ──
        nb = ttk.Notebook(frame)
        nb.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        self._tab_summary  = self._make_scroll_tab(nb, "Summary")
        self._tab_shap     = self._make_scroll_tab(nb, "SHAP Features")
        self._tab_raw      = self._make_scroll_tab(nb, "Raw JSON")

        nb.add(self._tab_summary[0],  text="Summary")
        nb.add(self._tab_shap[0],     text="SHAP Features")
        nb.add(self._tab_raw[0],      text="Raw JSON")

    def _make_scroll_tab(self, parent, name):
        frame  = tk.Frame(parent, bg=C_SURFACE)
        scroll = tk.Scrollbar(frame)
        text   = tk.Text(
            frame, wrap="word", font=self._normal,
            bg="#fafafa", fg=C_TEXT, relief="flat",
            yscrollcommand=scroll.set, state="disabled",
            padx=8, pady=6,
        )
        scroll.config(command=text.yview)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        return frame, text

    # ── Server status ─────────────────────────────────────────────────────────

    def _check_server_status(self):
        def _check():
            try:
                d = _get_json("/v1/health")
                ok = d.get("status") == "ok"
            except Exception:
                ok = False
            self.after(0, self._update_status, ok)
            self.after(8000, self._check_server_status)

        threading.Thread(target=_check, daemon=True).start()

    def _update_status(self, ok: bool):
        if ok:
            self._status_lbl.config(text="● API online", fg="#86efac")
        else:
            self._status_lbl.config(text="● API offline", fg="#fca5a5")

    # ── Run triage ────────────────────────────────────────────────────────────

    def _run_triage(self):
        age_s = self._fields["age"].get().strip()
        cc_s  = self._fields["chief_complaint"].get().strip()
        if not age_s or not cc_s:
            messagebox.showwarning("Missing fields", "Age and Chief Complaint are required.")
            return

        payload: dict[str, Any] = {
            "age":            float(age_s),
            "chief_complaint": cc_s,
        }
        for key, cast in [
            ("gender",           lambda x: x.strip().upper() or None),
            ("arrival_hour",     _opt_int),
            ("systolic_bp",      _opt_float),
            ("diastolic_bp",     _opt_float),
            ("heart_rate",       _opt_float),
            ("respiratory_rate", _opt_float),
            ("temperature",      _opt_float),
            ("spo2",             _opt_float),
            ("pain_score",       _opt_float),
            ("encounter_id",     lambda x: x.strip() or None),
        ]:
            val = cast(self._fields[key].get())
            if val is not None:
                payload[key] = val

        self._run_btn.config(state="disabled", text="⏳  Running…")
        self._clear_results()

        def _worker():
            try:
                result = _post_json(self._endpoint.get(), payload)
                self.after(0, self._show_results, result)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode()
                self.after(0, self._show_error, f"HTTP {exc.code}: {body}")
            except Exception as exc:
                self.after(0, self._show_error, str(exc))
            finally:
                self.after(0, lambda: self._run_btn.config(
                    state="normal", text="▶  Run Triage"))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Display helpers ───────────────────────────────────────────────────────

    def _clear_results(self):
        self._acuity_lbl.config(
            text="Running…", bg=C_SURFACE, fg=C_MUTED)
        self._acuity_frame.config(bg=C_SURFACE)
        for _, txt in [self._tab_summary, self._tab_shap, self._tab_raw]:
            txt.config(state="normal")
            txt.delete("1.0", "end")
            txt.config(state="disabled")

    def _show_results(self, data: dict):
        # ── Determine acuity ──
        acuity = data.get("predicted") or data.get("predicted_acuity", "?")
        conf   = data.get("confidence", 0)
        bg, fg, label = ACUITY_COLOURS.get(int(acuity), (C_SURFACE, C_TEXT, "UNKNOWN"))

        self._acuity_frame.config(bg=bg)
        self._acuity_lbl.config(
            bg=bg, fg=fg,
            text=f"Acuity {acuity}  —  {label}   ({conf*100:.1f}% confidence)",
        )

        # ── Summary tab ──
        _, txt = self._tab_summary
        txt.config(state="normal")
        txt.delete("1.0", "end")
        txt.tag_config("h",    font=self._bold, foreground=C_ACCENT)
        txt.tag_config("item", font=self._normal, foreground=C_TEXT)
        txt.tag_config("flag", font=self._normal, foreground=C_RED)
        txt.tag_config("muted",font=self._small,  foreground=C_MUTED)

        def sec(title): txt.insert("end", f"\n{title}\n", "h")
        def li(line, tag="item"): txt.insert("end", f"  • {line}\n", tag)

        # Probability bar
        sec("Acuity Probabilities")
        proba = data.get("proba", {})
        for k in ["1", "2", "3", "4"]:
            p = proba.get(k, 0)
            bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
            _, col, lbl = ACUITY_COLOURS.get(int(k), (C_SURFACE, C_TEXT, ""))
            txt.tag_config(f"p{k}", font=self._mono, foreground=col)
            txt.insert("end", f"  {k} {lbl:<11} {bar}  {p*100:5.1f}%\n", f"p{k}")

        # Safety flags
        flags = data.get("safety_flags", [])
        if flags:
            sec("⚠  Safety Flags")
            for f in flags:
                li(f.replace("_", " "), "flag")

        # Imputed fields
        imp = data.get("imputed_fields", [])
        if imp:
            sec("Imputed Fields")
            li(", ".join(imp))

        # Triage-specific rationale
        for field, title in [
            ("abnormal_vitals",    "Abnormal Vitals"),
            ("differential_risks", "Differential Risks"),
            ("immediate_actions",  "Immediate Actions"),
            ("red_flags",          "Red Flags"),
        ]:
            items = data.get(field, [])
            if items:
                sec(title)
                for it in items:
                    li(it)

        note = data.get("safety_note", "")
        if note:
            sec("Safety Note")
            txt.insert("end", f"  {note}\n", "item")

        backend = data.get("llm_backend", "")
        if backend:
            txt.insert("end", f"\n  Backend: {backend}\n", "muted")

        txt.config(state="disabled")

        # ── SHAP tab ──
        _, stxt = self._tab_shap
        stxt.config(state="normal")
        stxt.delete("1.0", "end")
        stxt.tag_config("h",   font=self._bold,  foreground=C_ACCENT)
        stxt.tag_config("pos", font=self._mono,  foreground=C_RED)
        stxt.tag_config("neg", font=self._mono,  foreground=C_GREEN)
        stxt.tag_config("neu", font=self._mono,  foreground=C_TEXT)

        shap_top = data.get("shap_top", {})
        for cls_key in sorted(shap_top):
            _, col, lbl = ACUITY_COLOURS.get(int(cls_key), (C_SURFACE, C_TEXT, ""))
            stxt.tag_config(f"cls{cls_key}", font=self._bold, foreground=col)
            stxt.insert("end", f"\nAcuity {cls_key} — {lbl}\n", f"cls{cls_key}")
            for feat in shap_top[cls_key]:
                fname = feat.get("feature", "")
                fval  = feat.get("shap_value", 0)
                bar   = ("▲ " if fval > 0 else "▼ ") + f"{abs(fval):.4f}"
                tag   = "pos" if fval > 0 else "neg"
                stxt.insert("end", f"  {fname:<28} {bar}\n", tag)

        stxt.config(state="disabled")

        # ── Raw JSON tab ──
        _, rtxt = self._tab_raw
        rtxt.config(state="normal", font=self._mono)
        rtxt.delete("1.0", "end")
        rtxt.insert("end", json.dumps(data, indent=2))
        rtxt.config(state="disabled")

    def _show_error(self, msg: str):
        _, txt = self._tab_summary
        txt.config(state="normal")
        txt.delete("1.0", "end")
        txt.tag_config("err", font=self._bold, foreground=C_RED)
        txt.insert("end", f"Error\n\n{msg}", "err")
        txt.config(state="disabled")
        self._acuity_lbl.config(
            text="Request failed", bg=C_SURFACE, fg=C_RED)
        self._acuity_frame.config(bg=C_SURFACE)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_fields(self):
        defaults = {
            "age": "65", "gender": "M",
            "chief_complaint": "chest pain",
            "arrival_hour": "14",
            "systolic_bp": "80", "diastolic_bp": "50",
            "heart_rate": "122", "respiratory_rate": "24",
            "temperature": "37.2", "spo2": "93",
            "pain_score": "8", "encounter_id": "",
        }
        for k, v in defaults.items():
            self._fields[k].set(v)
        self._clear_results()
        self._acuity_lbl.config(text="Submit a patient to see results",
                                fg=C_MUTED, bg=C_SURFACE)

    # ── Tooltip ───────────────────────────────────────────────────────────────

    def _show_tip(self, text: str):
        self._tip_var.set(f"ℹ  {text}")

    def _hide_tip(self):
        self._tip_var.set("")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = TriageApp()
    app.mainloop()
