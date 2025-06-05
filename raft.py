#!/usr/bin/env python3
"""
RAFT – Rapid Analysis of Faulty columns in infrared image sTacks
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
        pk, _ = find_peaks(col_m,          height=mu + const*sd, distance=20)
        vl, _ = find_peaks(-col_m, height=const*sd - mu, distance=20) if include_valleys else ([], {})
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
    TITLE_COLS = ["#fa5252", "#f7b731", "#3772ff", "#20c997"]

    def __init__(self):
        super().__init__()
        self.title("RAFT – Rapid Analysis of Faulty columns")
        self.h5_file_handle = None

        try:
            self.state("zoomed")
        except tk.TclError:
            self.attributes("-fullscreen", True)

        self.content = tk.Frame(self)
        self.content.pack(fill="both", expand=True)
        self.show_home()

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

    def show_home(self):
        self._clear()
        if self.h5_file_handle:
            try:
                self.h5_file_handle.close()
                self.h5_file_handle = None
            except Exception as e:
                print(f"Could not close HDF5 file: {e}")

        self.configure(bg="white")
        self._make_title(self.content)
        ctr = tk.Frame(self.content, bg="white")
        ctr.pack(expand=True)

        def big_btn(text, cmd):
            tk.Button(ctr, text=text, width=32, height=2,
                      font=("Segoe UI", 16), command=cmd
                      ).pack(pady=10)

        big_btn("Parameter setting mode", self._choose_param_file)
        big_btn("Batch run mode", self._choose_batch_folder)
        big_btn("Exit", self.destroy)

    def _choose_param_file(self):
        fp = filedialog.askopenfilename(title="Select .h5 stack",
                                        filetypes=[("HDF-5 files", "*.h5"), ("All files", "*.*")])
        if fp:
            self.show_parameter_mode(fp)

    def show_parameter_mode(self, file_path: str):
        self._clear()

        try:
            self.h5_file_handle = h5py.File(file_path, "r")
            all_frame_keys = sorted(self.h5_file_handle.keys(), key=lambda x: int(x.split()[-1]))
            n_frames = len(all_frame_keys)
        except Exception as e:
            messagebox.showerror("HDF5 Error", f"Failed to open or read file: {e}")
            self.show_home()
            return

        # **OPTIMIZATION**: State and cache management dictionary
        state = {
            'raw_data': None,
            'clipped_image': None,
            'analysis_result': None,
            'last_frame_idx': -1,
            'last_clip_params': None,
            'last_analysis_params': None,
        }

        tk.Button(self.content, text="Home", font=("Segoe UI", 12),
                  command=self.show_home).pack(anchor="w", padx=6, pady=6)
        frm_plot = tk.Frame(self.content)
        frm_plot.pack(fill="both", expand=True)

        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(
            2, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [3, 1]}
        )
        plt.subplots_adjust(bottom=0.25)
        canvas = FigureCanvasTkAgg(fig, master=frm_plot)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        s_frame = Slider(plt.axes([0.25, 0.10, 0.65, 0.03]), "Frame #", 0, n_frames - 1, valinit=0, valstep=1)
        s_split = Slider(plt.axes([0.25, 0.05, 0.65, 0.03]), "Number of Slices", 1, 10, valinit=4, valstep=1)
        s_const = Slider(plt.axes([0.25, 0.01, 0.65, 0.03]), "Threshold", 0.0, 20.0, valinit=4.0, valstep=0.1)
        t_vmin = TextBox(plt.axes([0.05, 0.10, 0.05, 0.03]), "Vmin (%)", initial="1")
        t_vmax = TextBox(plt.axes([0.05, 0.05, 0.05, 0.03]), "Vmax (%)", initial="99")
        chk_val = CheckButtons(plt.axes([0.05, 0.01, 0.1, 0.03]), ["Show Valleys"], [False])

        def _process_peaks(img, n_s, const, valleys):
            H, W = img.shape
            slice_h = H // n_s
            all_sets, metric_col = [], defaultdict(list)
            for i in range(n_s):
                r0, r1 = i * slice_h, (i + 1) * slice_h if i < n_s - 1 else H
                sl = img[r0:r1]
                col_m = sl.mean(axis=0)
                mu, sd = np.mean(col_m), np.std(col_m)
                th_p, th_v = mu + const * sd, const * sd - mu
                pk, pprop = find_peaks(col_m, height=th_p, distance=20)
                vl, vprop = find_peaks(-col_m, height=th_v, distance=20) if valleys else ([], {})
                all_sets.append(set(pk) | set(vl))
                for j, p in enumerate(pk):
                    metric_col[p].append((pprop["peak_heights"][j] - mu) / sd)
                for j, v in enumerate(vl):
                    metric_col[v].append((vprop["peak_heights"][j] + mu) / sd)
            union = set.union(*all_sets) if all_sets else set()
            nonf = set.intersection(*all_sets) if all_sets else set()
            frag = union - nonf
            return nonf, frag, metric_col

        def _update(_=None):
            try:
                frame_idx = int(s_frame.val)
                vmin_pct = float(t_vmin.text)
                vmax_pct = float(t_vmax.text)
                splits = int(s_split.val)
                const = s_const.val
                valleys = chk_val.get_status()[0]
            except (ValueError, IndexError):
                return

            # --- INTELLIGENT CACHING LOGIC ---

            # 1. Load raw data only if frame changes
            if frame_idx != state['last_frame_idx']:
                state['raw_data'] = self.h5_file_handle[all_frame_keys[frame_idx]][:]
                state['last_frame_idx'] = frame_idx
                state['last_clip_params'] = None # Invalidate caches
                state['last_analysis_params'] = None

            # 2. Re-clip image only if vmin/vmax changes
            clip_params = (vmin_pct, vmax_pct)
            if clip_params != state['last_clip_params']:
                # This is the first heavy operation
                raw_img = state['raw_data']
                state['clipped_image'] = np.clip(raw_img, *np.percentile(raw_img, [vmin_pct, vmax_pct]))
                state['last_clip_params'] = clip_params
                state['last_analysis_params'] = None # Invalidate analysis cache

            # 3. Re-run analysis only if params change
            analysis_params = (splits, const, valleys)
            if analysis_params != state['last_analysis_params']:
                 # This is the second heavy operation
                state['analysis_result'] = _process_peaks(state['clipped_image'], splits, const, valleys)
                state['last_analysis_params'] = analysis_params

            # --- PLOTTING (now uses cached data) ---
            img, (nonf_now, frag_now, met_now) = state['clipped_image'], state['analysis_result']
            vmin_plot, vmax_plot = img.min(), img.max()

            for ax in (ax1, ax2, ax3, ax4): ax.clear()

            ax1.imshow(img, cmap="gray", vmin=vmin_plot, vmax=vmax_plot)
            ax1.set_title("Original (Stretched)"); ax1.axis("off")
            ax2.imshow(img, cmap="gray", vmin=vmin_plot, vmax=vmax_plot)
            ax2.set_title("Labeled"); ax2.axis("off")
            
            W = img.shape[1]
            for p in nonf_now:
                y = np.mean(met_now.get(p, [0]))
                ax2.axvline(p, color="red")
                ax3.vlines(p, 0, y, color="red")
                ax3.text(p, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
            for p in frag_now:
                y = np.mean(met_now.get(p, [0]))
                ax2.axvline(p, color="blue", linestyle="--")
                ax4.vlines(p, 0, y, color="blue")
                ax4.text(p, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

            ax3.set_title(f"Non-Fragmented peaks – frame {frame_idx}")
            ax4.set_title(f"Fragmented peaks – frame {frame_idx}")
            for a in (ax3, ax4):
                a.set_xlim(0, W); a.set_ylim(0, 20); a.grid(True, ls='--', alpha=0.6)

            canvas.draw_idle()

        s_frame.on_changed(_update)
        s_split.on_changed(_update)
        s_const.on_changed(_update)
        t_vmin.on_submit(_update)
        t_vmax.on_submit(_update)
        chk_val.on_clicked(_update)
        _update()

    # ---------- BATCH MODE (Unchanged) ------------------------------------
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

if __name__ == "__main__":
    app = RAFTApp()
    app.mainloop()
