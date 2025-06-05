#!/usr/bin/env python3
"""
RAFT – Rapid Analysis of Faulty columns in image sTacks
=======================================================
One-window Tkinter GUI with

1. Parameter-setting mode  - interactive exploration of one *.h5* stack.
2. Batch-run mode          - analyse every *.h5* in a folder and emit CSV/PNG summaries.

Authors:
    - Rayan
    - Arsen
    - Fazeel
    - Tran Dong
"""
from __future__ import annotations
import csv, pathlib, threading, tkinter as tk
from collections import Counter, defaultdict
from functools import lru_cache
from tkinter import filedialog, messagebox
from typing import Dict, List, Set, Tuple

import h5py, numpy as np
from scipy.signal import find_peaks

# ────────── Matplotlib – embed with TkAgg ──────────────────────────────
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, TextBox, CheckButtons


# ═════════════════════ low-level helpers ═══════════════════════════════
def detect_peaks(img: np.ndarray, *, n_splits: int, const: float,
                 include_valleys: bool) -> Tuple[Set[int], Set[int]]:
    """Return non-fragmented and fragmented peak columns in a single frame."""
    H, _ = img.shape
    slice_h = H // n_splits
    per_slice: List[Set[int]] = []

    for i in range(n_splits):
        r0, r1 = i*slice_h, (i+1)*slice_h if i < n_splits-1 else H
        sl = img[r0:r1]
        col_m = sl.mean(axis=0)
        mu, sd = np.mean(col_m), np.std(col_m)
        pk, _ = find_peaks(col_m,          height=mu + const*sd)
        vl, _ = find_peaks(-col_m, height=const*sd - mu) if include_valleys else ([], {})
        per_slice.append(set(pk) | set(vl))

    union = set.union(*per_slice) if per_slice else set()
    non_frag = set.intersection(*per_slice) if per_slice else set()
    frag = union - non_frag
    return non_frag, frag


def analyse_stack(path: pathlib.Path, *, const: float, vmin: float, vmax: float,
                  splits: int, valleys: bool) -> Dict:
    frame_names, nonfrag_pf, frag_pf, union_pf = [], [], [], []
    with h5py.File(path, "r") as f:
        for fname in sorted(f, key=lambda x: int(x.split()[-1])):
            frame_names.append(fname)
            img = f[fname][:]
            img = np.clip(img, *np.percentile(img, [vmin, vmax]))
            nonf, frag = detect_peaks(img, n_splits=splits, const=const,
                                      include_valleys=valleys)
            nonfrag_pf.append(nonf)
            frag_pf.append(frag)
            union_pf.append(nonf | frag)

    stable   = set.intersection(*union_pf)
    blinking = set.union(*union_pf) - stable
    union_nf = set.union(*nonfrag_pf)
    union_fr = set.union(*frag_pf)

    noisy_frames = defaultdict(list)
    for fn, nonf in zip(frame_names, nonfrag_pf):
        for c in nonf:
            noisy_frames[c].append(fn)

    return dict(frame_names=frame_names,
                nonfrag_pf=nonfrag_pf, frag_pf=frag_pf, union_pf=union_pf,
                stable=stable, blinking=blinking,
                union_nf=union_nf, union_fr=union_fr,
                noisy_frames=noisy_frames)


# ────────── CSV & plotting utilities (unchanged in spirit) ────────────
def write_frame_summary(path: pathlib.Path, *, frame_names: List[str],
                        frag_pf: List[Set[int]], union_pf: List[Set[int]]):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Frame Name", "Fragmented Columns", "Non Fragmented Columns"])
        for fn, frag, uni in zip(frame_names, frag_pf, union_pf):
            non_frag = uni - frag
            w.writerow([fn,
                        ";".join(map(str, sorted(frag))) or "",
                        ";".join(map(str, sorted(non_frag))) or ""])


def write_column_summary(path: pathlib.Path, ana: Dict):
    cnf, cf, bnf, bf = set(), set(), set(), set()
    for col in ana["union_nf"] | ana["union_fr"] | ana["blinking"]:
        continuous = col in ana["stable"]
        fragmented = col in ana["union_fr"]
        if  continuous and not fragmented: cnf.add(col)
        elif continuous and     fragmented: cf.add(col)
        elif (not continuous) and (not fragmented): bnf.add(col)
        else: bf.add(col)

    def cat(col: int) -> str:
        if col in cnf: return "Continuous Non Fragmented"
        if col in cf:  return "Continuous Fragmented"
        if col in bnf: return "Blinking Non Fragmented"
        return          "Blinking Fragmented"

    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Column Number", "Categorization", "Noisy Frames"])
        for col in sorted(cnf | cf | bnf | bf):
            w.writerow([col, cat(col),
                        ";".join(ana["noisy_frames"].get(col, []))])


def _bar_top(path_png: pathlib.Path, counter: Counter[int],
             title: str, colour: str):
    if not counter:
        return
    top = counter.most_common(25)
    cols, freqs = zip(*top)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(cols)), freqs, color=colour)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_ylabel("# Frames")
    ax.set_title(title)
    ax.grid(axis="y", ls="--", alpha=.4)
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


def _bar_category_counts(path_png: pathlib.Path, *, cnf: set, cf: set,
                         bnf: set, bf: set):
    cats   = ["Continuous\nNon Frag.", "Continuous\nFrag.",
              "Blinking\nNon Frag.",  "Blinking\nFrag."]
    counts = [len(cnf), len(cf), len(bnf), len(bf)]
    colors = ["#fa5252", "#ff922b", "#4dabf7", "#20c997"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cats, counts, color=colors)
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha="center")
    ax.set_ylabel("# Columns")
    ax.set_title("Global category counts")
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


# ════════════════════ Tkinter GUI (single window) ═════════════════════
class RAFTApp(tk.Tk):
    # Coral-red, amber, royal-blue, teal  → distinct yet harmonious
    TITLE_COLS = ["#fa5252", "#f7b731", "#3772ff", "#20c997"]

    def __init__(self):
        super().__init__()
        self.title("RAFT – Rapid Analysis of Faulty columns")

        # full-screen / maximised by default (works on Win & Linux alike)
        try:                     # Windows & some Linux DEs
            self.state("zoomed")
        except tk.TclError:      # fallback: generic full-screen
            self.attributes("-fullscreen", True)

        # Root container we swap in/out
        self.content = tk.Frame(self)
        self.content.pack(fill="both", expand=True)
        self.show_home()

    # ---------- helpers ---------------------------------------------------
    def _clear(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _make_title(self, parent):
        bar = tk.Frame(parent, bg="white")
        bar.pack(pady=15)
        for ch, col in zip("RAFT", RAFTApp.TITLE_COLS):
            tk.Label(bar, text=ch, fg=col,
                     font=("Segoe UI", 52, "bold"), bg="white"
                     ).pack(side="left")
        tk.Frame(parent, height=2, bd=1, relief="sunken"
                 ).pack(fill="x", padx=20, pady=10)

    # ---------- HOME SCREEN ----------------------------------------------
    def show_home(self):
        self._clear()
        self.configure(bg="white")
        self._make_title(self.content)

        ctr = tk.Frame(self.content, bg="white")
        ctr.pack(expand=True)

        def big_btn(text, cmd):
            tk.Button(ctr, text=text, width=32, height=2,
                      font=("Segoe UI", 16), command=cmd
                      ).pack(pady=10)

        big_btn("Parameter setting mode", self._choose_param_file)
        big_btn("Batch run mode",         self._choose_batch_folder)
        big_btn("Exit",                   self.destroy)

    # ---------- PARAMETER MODE -------------------------------------------
    def _choose_param_file(self):
        fp = filedialog.askopenfilename(title="Select .h5 stack",
                                        filetypes=[("HDF-5 files", "*.h5"),
                                                   ("All files", "*.*")])
        if fp:
            self.show_parameter_mode(fp)

    def show_parameter_mode(self, file_path: str):
        self._clear()
        tk.Button(self.content, text="Home", font=("Segoe UI", 12),
                  command=self.show_home).pack(anchor="w", padx=6, pady=6)

        frm_plot = tk.Frame(self.content)
        frm_plot.pack(fill="both", expand=True)

        # ---- Prepare data ------------------------------------------------
        with h5py.File(file_path, "r") as f:
            all_frames = sorted(f.keys(), key=lambda x: int(x.split()[-1]))
        n_frames = len(all_frames)

        # Defaults
        p_vmax_d, p_vmin_d = 99, 1
        n_splits_d, const_d = 4, 4.0

        # ---- Build Matplotlib figure ------------------------------------
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(
            2, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [3, 1]}
        )
        plt.subplots_adjust(bottom=0.25)
        canvas = FigureCanvasTkAgg(fig, master=frm_plot)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Widgets inside figure
        ax_sf = plt.axes([0.25, 0.10, 0.65, 0.03])
        s_frame = Slider(ax_sf, "Frame #", 0, n_frames-1,
                         valinit=0, valstep=1)
        ax_ss = plt.axes([0.25, 0.05, 0.65, 0.03])
        s_split = Slider(ax_ss, "Number of Slices", 1, 10,
                         valinit=n_splits_d, valstep=1)
        ax_sc = plt.axes([0.25, 0.01, 0.65, 0.03])
        s_const = Slider(ax_sc, "Threshold", 0.0, 20.0,
                         valinit=const_d, valstep=0.1)

        ax_vmin = plt.axes([0.05, 0.10, 0.05, 0.03])
        ax_vmax = plt.axes([0.05, 0.05, 0.05, 0.03])
        t_vmin = TextBox(ax_vmin, "Vmin (%)", initial=str(p_vmin_d))
        t_vmax = TextBox(ax_vmax, "Vmax (%)", initial=str(p_vmax_d))

        ax_chk = plt.axes([0.05, 0.01, 0.1, 0.03])
        chk_val = CheckButtons(ax_chk, ["Show Valleys"], [False])

        # ---- Cached frame-stack analysis --------------------------------
        def _sig(n_s, c, vmin, vmax, val):
            return (int(n_s), round(c, 4), round(vmin, 3),
                    round(vmax, 3), val)

        @lru_cache(maxsize=8)
        def _analyse(sig):
            n_s, const, pvmin, pvmax, valleys = sig
            union_pf, nonfrag_pf, frag_pf, metric_pf = [], [], [], []
            noise_avg_col = defaultdict(list)
            for fname in all_frames:
                with h5py.File(file_path, "r") as f:
                    img = f[fname][:]
                img = np.clip(img, *np.percentile(img, [pvmin, pvmax]))
                nfr, frg, met = _process(img, n_s, const, valleys,
                                         noise_avg_col)
                nonfrag_pf.append(nfr)
                frag_pf.append(frg)
                union_pf.append(nfr | frg)
                metric_pf.append(met)
            stable = set.intersection(*union_pf)
            blink = set.union(*union_pf) - stable
            return dict(
                union_pf=union_pf, nonfrag_pf=nonfrag_pf, frag_pf=frag_pf,
                metric_pf=metric_pf, noise_avg_col=noise_avg_col,
                Continuous_NF   = stable & set.union(*nonfrag_pf),
                Blinking_NF     = blink  & set.union(*nonfrag_pf),
                Continuous_Frag = stable & set.union(*frag_pf),
                Blinking_Frag   = blink  & set.union(*frag_pf),
            )

        def _process(img, n_s, const, valleys, nav):
            H, W = img.shape
            slice_h = H // n_s
            all_sets, metric_col = [], defaultdict(list)
            for i in range(n_s):
                r0, r1 = i*slice_h, (i+1)*slice_h if i < n_s-1 else H
                sl = img[r0:r1]
                col_m = sl.mean(axis=0)
                mu, sd = np.mean(col_m), np.std(col_m)
                th_p, th_v = mu + const*sd, const*sd - mu
                pk, pprop = find_peaks(col_m,          height=th_p)
                vl, vprop = find_peaks(-col_m, height=th_v) if valleys else ([], {})
                both = np.concatenate([pk, vl]) if valleys else pk
                all_sets.append(set(both))
                for j, p in enumerate(pk):
                    metric_col[p].append((pprop["peak_heights"][j]-mu)/sd)
                for j, v in enumerate(vl):
                    metric_col[v].append((vprop["peak_heights"][j]+mu)/sd)
            union = set.union(*all_sets) if all_sets else set()
            nonf  = set.intersection(*all_sets) if all_sets else set()
            frag  = union - nonf
            for c, vals in metric_col.items():
                nav[c].append(np.mean(vals))
            return nonf, frag, metric_col

        def _get_max(peaks, nav):
            return max(peaks, key=lambda p: np.mean(nav[p])) if peaks else None

        def _overview(ax_main, ax_nf, ax_fr, ana, W):
            nav = ana["noise_avg_col"]
            mapping = [
                ("Continuous Non Fragmented", ana["Continuous_NF"],
                 ax_nf,  "#000000", "-"),
                ("Blinking  Non Fragmented",  ana["Blinking_NF"],
                 ax_nf,  "#f7b731", "-"),
                ("Continuous Fragmented",     ana["Continuous_Frag"],
                 ax_fr,  "#20c997", "-"),
                ("Blinking  Fragmented",      ana["Blinking_Frag"],
                 ax_fr,  "#3772ff", "-"),
            ]
            for cat, peaks, ax, col, ls in mapping:
                if not peaks:
                    continue
                pmax = _get_max(peaks, nav)
                for p in peaks:
                    y = np.mean(nav[p])
                    ax.vlines(p, 0, y, color=col, linestyle=ls, linewidth=1,
                              label=cat if p == min(peaks) else "")
                    ax.text(p, y, f"{y:.1f}", ha="center", va="bottom",
                            fontsize=8, color="red")
                    ax_main.axvline(p, color=col,
                                    linestyle="--" if "Fragmented" in cat else "-",
                                    linewidth=1,
                                    label=cat if p == min(peaks) else "")
                    if p == pmax:
                        ax_main.plot(p, y, "ro")
                ax.set_xlim(0, W)
                ax.set_ylim(0, 20)
                ax.set_title(f"{cat} ({len(peaks)})")
                ax.legend()
            ax_main.legend()

        # ---- update callback --------------------------------------------
        prev_sig, ana = None, None
        def _update(_=None):
            nonlocal prev_sig, ana
            sig = _sig(s_split.val, s_const.val,
                       float(t_vmin.text), float(t_vmax.text),
                       chk_val.get_status()[0])
            if sig != prev_sig:
                ana = _analyse(sig)
                prev_sig = sig

            idx = int(s_frame.val)
            fname = all_frames[idx]
            with h5py.File(file_path, "r") as f:
                img = f[fname][:]
            vmin, vmax = np.percentile(img,
                                       [float(t_vmin.text),
                                        float(t_vmax.text)])
            img = np.clip(img, vmin, vmax)

            for a in (ax1, ax2, ax3, ax4):
                a.clear()

            ax1.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax1.set_title("Original");  ax1.axis("off")
            ax2.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax2.set_title("Labeled");   ax2.axis("off")

            nf_now = ana["nonfrag_pf"][idx]
            fr_now = ana["frag_pf"][idx]
            met_now = ana["metric_pf"][idx]
            W = img.shape[1]

            for p in nf_now:
                y = np.mean(met_now.get(p, [0]))
                ax2.axvline(p, color="red")
                ax3.vlines(p, 0, y, color="red")
                ax3.text(p, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
            for p in fr_now:
                y = np.mean(met_now.get(p, [0]))
                ax2.axvline(p, color="blue", linestyle="--")
                ax4.vlines(p, 0, y, color="blue")
                ax4.text(p, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

            ax3.set_title(f"Non-Fragmented peaks – frame {idx+1}")
            ax4.set_title(f"Fragmented peaks – frame {idx+1}")
            for a in (ax3, ax4):
                a.set_xlim(0, W); a.set_ylim(0, 20); a.grid(True)

            if idx == n_frames-1:
                _overview(ax2, ax3, ax4, ana, W)

            canvas.draw_idle()

        # widget connections
        s_frame.on_changed(_update)
        s_split.on_changed(_update)
        s_const.on_changed(_update)
        t_vmin.on_submit(_update)
        t_vmax.on_submit(_update)
        chk_val.on_clicked(_update)

        _update()      # initial draw

    # ---------- BATCH MODE ----------------------------------------------
    def _choose_batch_folder(self):
        dp = filedialog.askdirectory(title="Select directory with .h5 files")
        if dp:
            self.show_batch_mode(dp)

    def show_batch_mode(self, dir_path: str):
        self._clear()
        tk.Button(self.content, text="Home", font=("Segoe UI", 12),
                  command=self.show_home).pack(anchor="w", padx=6, pady=6)

        frm = tk.Frame(self.content)
        frm.pack(pady=20)

        defaults = dict(threshold="4.0", vmin="1", vmax="99", splits="4")
        entries = {}
        for r, (lbl, default) in enumerate(defaults.items()):
            tk.Label(frm, text=f"{lbl}:", font=("Segoe UI", 14)
                     ).grid(row=r, column=0, sticky="e", padx=4, pady=4)
            e = tk.Entry(frm, width=10, font=("Segoe UI", 14))
            e.insert(0, default)
            e.grid(row=r, column=1, sticky="w", padx=4, pady=4)
            entries[lbl] = e

        val_var = tk.IntVar(value=0)
        tk.Checkbutton(frm, text="Include valleys", variable=val_var,
                       font=("Segoe UI", 14)
                       ).grid(row=len(defaults), column=0, columnspan=2, pady=6)

        tk.Button(frm, text="Run batch analysis", width=22,
                  font=("Segoe UI", 14, "bold"),
                  command=lambda: self._run_batch(dir_path, entries, val_var)
                  ).grid(row=len(defaults)+1, column=0, columnspan=2, pady=15)

    # ----- heavy batch work in background thread ------------------------
    def _run_batch(self, dir_path, entries, val_var):
        try:
            thr   = float(entries["threshold"].get())
            vmin  = float(entries["vmin"].get())
            vmax  = float(entries["vmax"].get())
            splits= int  (entries["splits"].get())
        except ValueError:
            messagebox.showerror("RAFT", "Invalid numeric values.")
            return

        threading.Thread(
            target=self._batch_worker,
            kwargs=dict(dp=dir_path, thr=thr, vmin=vmin, vmax=vmax,
                        splits=splits, valleys=bool(val_var.get())),
            daemon=True
        ).start()
        messagebox.showinfo("RAFT", "Batch processing started … you'll be "
                             "notified when it completes.")

    def _batch_worker(self, *, dp, thr, vmin, vmax, splits, valleys):
        src = pathlib.Path(dp).expanduser().resolve()
        h5s = sorted(src.glob("*.h5"))
        if not h5s:
            self.after(0, lambda: messagebox.showerror(
                "RAFT", f"No .h5 files found in {src}"))
            return

        out_root = pathlib.Path(__file__).with_name("results")
        out_root.mkdir(exist_ok=True)

        for h5 in h5s:
            ana = analyse_stack(h5, const=thr, vmin=vmin, vmax=vmax,
                                splits=splits, valleys=valleys)
            out = out_root / h5.stem
            out.mkdir(parents=True, exist_ok=True)
            write_frame_summary(out/"frame_fragmentation_summary.csv",
                                frame_names=ana["frame_names"],
                                frag_pf=ana["frag_pf"], union_pf=ana["union_pf"])
            write_column_summary(out/"column_category_summary.csv", ana)

            cnf = {c for c in ana["union_nf"] if c in ana["stable"]
                                           and c not in ana["union_fr"]}
            cf  = {c for c in ana["union_fr"] if c in ana["stable"]}
            bnf = {c for c in ana["union_nf"] if c not in ana["stable"]}
            bf  = {c for c in ana["union_fr"] if c not in ana["stable"]}

            cnt_f = Counter(); cnt_nf = Counter()
            for s in ana["frag_pf"]:    cnt_f.update(s)
            for s in ana["nonfrag_pf"]: cnt_nf.update(s)

            _bar_top(out/"top_fragmented_columns.png",
                     cnt_f,  "Top 25 fragmented columns", "#3772ff")
            _bar_top(out/"top_nonfragmented_columns.png",
                     cnt_nf, "Top 25 non-fragmented columns", "#20c997")
            _bar_category_counts(out/"category_counts.png",
                                 cnf=cnf, cf=cf, bnf=bnf, bf=bf)

        self.after(0, lambda: messagebox.showinfo(
            "RAFT", f"Batch run finished.\nResults written to:\n{out_root}"))


# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RAFTApp().mainloop()
