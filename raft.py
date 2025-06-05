#!/usr/bin/env python3
# ============================================================================
#  RAFT - Rapid Analysis of Faulty (noisy) columns in image sTacks
#  --------------------------------------------------------------------------
#  A simple Tk-based launcher that lets you
#  (1) explore a single HDF-5 stack with an interactive matplotlib GUI, or
#  (2) run the batch analyser over a whole directory of *.h5 files.
#  Everything lives in **one** file for easy distribution.
# ============================================================================
from __future__ import annotations
import argparse, csv, pathlib, sys, threading
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Set, Tuple

import h5py, numpy as np
from scipy.signal import find_peaks

# --- Tk / GUI ---------------------------------------------------------------
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")                  # interactive in param-mode
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, CheckButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ======================= interactive (parameter-setting) GUI =======================
def run_interactive(file_path:str, on_close_callback:callable) -> None:
    """
    Replicates the original slider-based exploratory program, but optimized
    to run on a single frame at a time for performance. The main application
    window will be hidden and will reappear when this plot window is closed.
    """
    # 1. helpers ----------------------------------------------------------------
    def _analyse_frame_for_peaks(img, *, n_splits, const, include_valleys):
        """
        Processes a single image frame to find non-fragmented and fragmented
        noisy columns based on the provided parameters.
        """
        H, W   = img.shape
        slice_h= H // n_splits
        all_sets, metric_col = [], defaultdict(list)

        for i in range(n_splits):
            r0, r1 = i*slice_h, (i+1)*slice_h if i<n_splits-1 else H
            sl     = img[r0:r1]
            col_m  = sl.mean(axis=0)
            mu, sd = np.mean(col_m), np.std(col_m)

            th_pk, th_val = mu + const*sd, const*sd - mu
            pk, pprop = find_peaks(col_m,       height=th_pk)
            vl, vprop = find_peaks(-col_m, height=th_val) if include_valleys else ([], {})

            slice_peaks = set()
            if len(pk) > 0: slice_peaks.update(pk)
            if include_valleys and len(vl) > 0: slice_peaks.update(vl)
            all_sets.append(slice_peaks)

            # Calculate metric (height in terms of stddevs) for each peak/valley
            for j,p in enumerate(pk):
                metric_col[p].append((pprop["peak_heights"][j]-mu)/sd)
            for j,v in enumerate(vl):
                metric_col[v].append((vprop["peak_heights"][j]+mu)/sd)

        union_peaks = set.union(*all_sets) if all_sets else set()
        nonfrag_peaks = set.intersection(*all_sets) if all_sets else set()
        frag_peaks = union_peaks - nonfrag_peaks

        return nonfrag_peaks, frag_peaks, metric_col

    # 2. prepare once per file --------------------------------------------------
    with h5py.File(file_path, "r") as f:
        all_frames = sorted(f.keys(), key=lambda x:int(x.split()[-1]))
    num_frames = len(all_frames)

    # defaults
    p_vmax_default, p_vmin_default = 99, 1
    n_splits_default, const_default = 4, 4.0

    # 3. build figure with Matplotlib widgets -----------------------------------
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(
        2, 2, figsize=(20,10), gridspec_kw={"width_ratios":[3,1]}
    )
    plt.subplots_adjust(bottom=0.25)

    ax_sf = plt.axes([0.25, 0.10, 0.65, 0.03])
    slider_frame = Slider(ax_sf, "Frame #", 0, num_frames-1,
                          valinit=0, valstep=1)
    ax_ss = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider_split = Slider(ax_ss, "Number of Slices", 1, 10,
                          valinit=n_splits_default, valstep=1)
    ax_sc = plt.axes([0.25, 0.01, 0.65, 0.03])
    slider_const = Slider(ax_sc, "Threshold", 0.0, 20.0,
                          valinit=const_default, valstep=0.1)

    ax_vmin_box = plt.axes([0.05, 0.10, 0.05, 0.03])
    ax_vmax_box = plt.axes([0.05, 0.05, 0.05, 0.03])
    textbox_vmin = TextBox(ax_vmin_box, "Vmin (%)", initial=str(p_vmin_default))
    textbox_vmax = TextBox(ax_vmax_box, "Vmax (%)", initial=str(p_vmax_default))

    ax_chk = plt.axes([0.05, 0.01, 0.1, 0.03])
    chk_valleys = CheckButtons(ax_chk, ["Show Valleys"], [False])

    def _update(_=None):
        frame_idx = int(slider_frame.val)
        n_splits_val = int(slider_split.val)
        const_val = slider_const.val
        p_vmin_val = float(textbox_vmin.text)
        p_vmax_val = float(textbox_vmax.text)
        valleys_val = chk_valleys.get_status()[0]

        frame_num = frame_idx + 1
        with h5py.File(file_path,"r") as f:
            img = f[f"Image {frame_num:04d}"][:]

        vmin, vmax = np.percentile(img, [p_vmin_val, p_vmax_val])
        img = np.clip(img, vmin, vmax)

        # OPTIMIZATION: Analyse only the current frame, not the whole stack
        nonfrag_now, frag_now, metric_now = _analyse_frame_for_peaks(
            img,
            n_splits=n_splits_val,
            const=const_val,
            include_valleys=valleys_val
        )

        for a in (ax1, ax2, ax3, ax4): a.clear()

        ax1.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax1.set_title("Original"); ax1.axis("off")

        ax2.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax2.set_title("Labeled"); ax2.axis("off")

        W = img.shape[1]

        for p in nonfrag_now:
            y = np.mean(metric_now.get(p,[0]))
            ax2.axvline(p, color="red")
            ax3.vlines(p, 0, y, color="red")
            ax3.text(p, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
        for p in frag_now:
            y = np.mean(metric_now.get(p,[0]))
            ax2.axvline(p, color="blue", linestyle="--")
            ax4.vlines(p, 0, y, color="blue")
            ax4.text(p, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

        ax3.set_title(f"Non-Fragmented peaks - frame {frame_num}")
        ax4.set_title(f"Fragmented peaks - frame {frame_num}")
        for a in (ax3, ax4):
            a.set_xlim(0,W); a.set_ylim(0,20); a.grid(True)

        fig.canvas.draw_idle()

    slider_frame.on_changed(_update)
    slider_split.on_changed(_update)
    slider_const.on_changed(_update)
    textbox_vmin.on_submit(_update)
    textbox_vmax.on_submit(_update)
    chk_valleys.on_clicked(_update)

    _update()
    plt.show(block=True)  # blocks until figure is closed

    # UI/UX: Signal to the main app that this window has closed.
    if on_close_callback:
        on_close_callback()

# =========================== batch-mode backend ================================
def detect_peaks(img:np.ndarray, *, n_splits:int, const:float,
                 include_valleys:bool)->Tuple[Set[int],Set[int]]:
    H,_ = img.shape
    slice_h = H // n_splits
    per_slice : List[Set[int]] = []
    for i in range(n_splits):
        r0,r1 = i*slice_h, (i+1)*slice_h if i<n_splits-1 else H
        sl = img[r0:r1]
        col_mean = sl.mean(axis=0)
        mu, sd = np.mean(col_mean), np.std(col_mean)
        pk,_   = find_peaks( col_mean, height=mu+const*sd )
        vl,_   = find_peaks(-col_mean, height=const*sd-mu) if include_valleys else ([],{})
        per_slice.append(set(pk)|set(vl))
    union = set.union(*per_slice) if per_slice else set()
    non_frag = set.intersection(*per_slice) if per_slice else set()
    frag = union - non_frag
    return non_frag, frag


def analyse_stack(path:pathlib.Path, *, const:float, vmin:float, vmax:float,
                  splits:int, valleys:bool)->Dict:
   
    frame_names, nonfrag_pf, frag_pf, union_pf = [], [], [], []
    with h5py.File(path,"r") as f:
        # Sort frames numerically to ensure correct temporal order
        for fname in sorted(f, key=lambda x:int(x.split()[-1])):
            frame_names.append(fname)
            img = f[fname][:]
            
            # Clip image intensity to the specified percentile range to improve contrast
            img = np.clip(img, *np.percentile(img,[vmin, vmax]))
            
            # Detect peaks for the current frame
            nonf, frag = detect_peaks(img, n_splits=splits,
                                      const=const, include_valleys=valleys)
            
            # Store results for the current frame
            nonfrag_pf.append(nonf)
            frag_pf.append(frag)
            union_pf.append(nonf|frag)

    # Aggregate results across all frames
    stable     = set.intersection(*union_pf) if union_pf else set()
    blinking = set.union(*union_pf) - stable if union_pf else set()
    union_nf = set.union(*nonfrag_pf) if nonfrag_pf else set()
    union_frag = set.union(*frag_pf) if frag_pf else set()

    # ========================== CORRECTED SECTION ==========================
    # This section now correctly populates the 'noisy_frames' dictionary.
    # It iterates through 'union_pf', which contains ALL noisy columns
    # (both non-fragmented and fragmented) for each frame. This ensures
    # the final CSV report will list the corresponding frames for every
    # identified noisy column, regardless of its category.
    #
    noisy_frames = defaultdict(list)
    for fn, all_noisy_in_frame in zip(frame_names, union_pf):
        for c in all_noisy_in_frame:
            noisy_frames[c].append(fn)
    # ======================== END OF CORRECTED SECTION =======================

    # Return a dictionary with all the analysis results
    return dict(frame_names=frame_names,
                nonfrag_pf=nonfrag_pf, 
                frag_pf=frag_pf, 
                union_pf=union_pf,
                stable=stable, 
                blinking=blinking,
                union_nf=union_nf, 
                union_frag=union_frag,
                noisy_frames=noisy_frames)


# -- CSV writers & bar plots (unchanged except path param) -------------------
def write_frame_summary(path:pathlib.Path, *, frame_names:List[str],
                        frag_pf:List[Set[int]], union_pf:List[Set[int]])->None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Frame Name","Fragmented Columns","Non Fragmented Columns"])
        for fn, frag, uni in zip(frame_names, frag_pf, union_pf):
            non_frag = uni - frag
            w.writerow([fn,
                        ";".join(map(str,sorted(frag))) or "",
                        ";".join(map(str,sorted(non_frag))) or ""])

def write_column_summary(path:pathlib.Path, ana:Dict)->None:
    cnf, cf, bnf, bf = set(), set(), set(), set()
    for col in ana["union_nf"]|ana["union_frag"]|ana["blinking"]:
        continuous  = col in ana["stable"]
        fragmented  = col in ana["union_frag"]
        if  continuous and not fragmented: cnf.add(col)
        elif continuous and fragmented:     cf.add(col)
        elif (not continuous) and (not fragmented): bnf.add(col)
        else: bf.add(col)
    def cat(col:int)->str:
        if col in cnf: return "Continuous Non Fragmented"
        if col in cf:  return "Continuous Fragmented"
        if col in bnf: return "Blinking Non Fragmented"
        return         "Blinking Fragmented"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Column Number","Categorization","Noisy Frames"])
        for col in sorted(cnf|cf|bnf|bf):
            w.writerow([col, cat(col),
                        ";".join(ana["noisy_frames"].get(col,[]))])

import matplotlib.pyplot as _plt
def _bar_top(path_png:pathlib.Path, counter:Counter[int],
             title:str, colour:str)->None:
    if not counter: return
    top = counter.most_common(25)
    cols, freqs = zip(*top)
    fig, ax = _plt.subplots(figsize=(12,4))
    ax.bar(range(len(cols)), freqs, color=colour)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_ylabel("# Frames")
    ax.set_title(title)
    ax.grid(axis="y", ls="--", alpha=.4)
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    _plt.close(fig)

def _bar_category_counts(path_png:pathlib.Path, *, cnf:set, cf:set,
                         bnf:set, bf:set)->None:
    cats = ["Continuous\nNon Frag.","Continuous\nFrag.",
            "Blinking\nNon Frag.","Blinking\nFrag."]
    counts= [len(cnf), len(cf), len(bnf), len(bf)]
    colors= ["#1f77b4","#d62728","#ff7f0e","#9467bd"]
    fig, ax = _plt.subplots(figsize=(7,4))
    ax.bar(cats, counts, color=colors)
    for i,v in enumerate(counts): ax.text(i, v+0.5, str(v), ha="center")
    ax.set_ylabel("# Columns")
    ax.set_title("Global category counts")
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    _plt.close(fig)


def batch_analyse(folder:str, *, threshold:float, vmin:float,
                  vmax:float, splits:int, valleys:bool) -> None:
    src = pathlib.Path(folder).expanduser().resolve()
    h5_files = sorted(src.glob("*.h5"))
    if not h5_files:
        messagebox.showerror("RAFT - Batch mode",
                             f"No .h5 files found in {src}")
        return
    results_root = pathlib.Path(__file__).with_name("results")
    results_root.mkdir(exist_ok=True)

    for h5 in h5_files:
        ana = analyse_stack(h5, const=threshold, vmin=vmin, vmax=vmax,
                            splits=splits, valleys=valleys)
        out = results_root / h5.stem
        out.mkdir(parents=True, exist_ok=True)
        write_frame_summary(out / "frame_fragmentation_summary.csv",
                            frame_names=ana["frame_names"],
                            frag_pf=ana["frag_pf"], union_pf=ana["union_pf"])
        write_column_summary(out / "column_category_summary.csv", ana=ana)
        cnf = {c for c in ana["union_nf"] if c in ana["stable"]
                                         and c not in ana["union_frag"]}
        cf  = {c for c in ana["union_frag"] if c in ana["stable"]}
        bnf = {c for c in ana["union_nf"] if c not in ana["stable"]}
        bf  = {c for c in ana["union_frag"] if c not in ana["stable"]}
        cnt_frag, cnt_nf = Counter(), Counter()
        for s in ana["frag_pf"]: cnt_frag.update(s)
        for s in ana["nonfrag_pf"]: cnt_nf.update(s)
        _bar_top(out / "top_fragmented_columns.png",
                 cnt_frag, "Top 25 fragmented columns", "#9467bd")
        _bar_top(out / "top_nonfragmented_columns.png",
                 cnt_nf, "Top 25 non-fragmented columns", "#1f77b4")
        _bar_category_counts(out / "category_counts.png",
                             cnf=cnf, cf=cf, bnf=bnf, bf=bf)
    messagebox.showinfo("RAFT - Batch mode",
                        f"Done!  Results were written to:\n{results_root}")


# =============================== Tk front-end ==================================
class RAFTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAFT - Rapid Analysis of Faulty columns in image sTacks")
        self.geometry("420x230")
        self._build_home()

    # ---------------- home screen ------------------
    def _build_home(self):
        for w in self.winfo_children(): w.destroy()
        tk.Label(self, text="RAFT", font=("Segoe UI", 24, "bold")).pack(pady=10)

        tk.Button(self, text="Parameter setting mode", width=30,
                  command=self._param_mode).pack(pady=5)
        tk.Button(self, text="Batch run mode",         width=30,
                  command=self._batch_mode).pack(pady=5)
        tk.Button(self, text="Exit",                   width=30,
                  command=self.destroy).pack(pady=5)

    def _return_to_home(self):
        """Helper to rebuild the home screen and show the main window."""
        self._build_home()
        self.deiconify()

    # ---------------- parameter mode ---------------
    def _param_mode(self):
        file_path = filedialog.askopenfilename(
            title="Select .h5 stack",
            filetypes=[("HDF-5 files","*.h5"), ("All files","*.*")]
        )
        if not file_path: return                  # user cancelled

        # UI/UX: Hide main window and define callback to restore it later.
        self.withdraw()
        def _schedule_return_home():
            self.after(0, self._return_to_home)

        # Launch in another thread so Tk keeps responding
        threading.Thread(target=run_interactive,
                         args=(file_path, _schedule_return_home),
                         daemon=True).start()

    # ---------------- batch mode -------------------
    def _batch_mode(self):
        dir_path = filedialog.askdirectory(
            title="Select directory containing .h5 files")
        if not dir_path: return
        top = tk.Toplevel(self)
        top.title("RAFT - Batch parameters")
        tk.Label(top, text="Set processing parameters").grid(row=0,column=0,
                                                             columnspan=2,pady=8)
        # entries
        entries = {}
        defs = dict(threshold="4.0", vmin="1", vmax="99", splits="4")
        for i,(k,default) in enumerate(defs.items(), start=1):
            tk.Label(top, text=f"{k}:").grid(row=i,column=0, sticky="e", padx=5)
            e = tk.Entry(top, width=10)
            e.insert(0, default)
            e.grid(row=i,column=1, sticky="w")
            entries[k]=e
        valleys_var = tk.IntVar(value=0)
        tk.Checkbutton(top, text="Include valleys", variable=valleys_var
                       ).grid(row=len(defs)+1, column=0, columnspan=2, pady=2)

        # buttons
        def _run_batch():
            try:
                thr  = float(entries["threshold"].get())
                vmin = float(entries["vmin"].get())
                vmax = float(entries["vmax"].get())
                splits = int(entries["splits"].get())
            except ValueError:
                messagebox.showerror("RAFT","Invalid numeric value.")
                return
            top.destroy()
            # heavy work in thread to keep GUI responsive
            threading.Thread(target=batch_analyse, daemon=True,
                             args=(dir_path,),
                             kwargs=dict(threshold=thr, vmin=vmin, vmax=vmax,
                                         splits=splits, valleys=bool(valleys_var.get())),
                             ).start()

        tk.Button(top, text="Run", width=12, command=_run_batch
                  ).grid(row=len(defs)+2,column=0,pady=10)
        tk.Button(top, text="Home", width=12, command=top.destroy
                  ).grid(row=len(defs)+2,column=1,pady=10)


# =========================== main entry point ==================================
if __name__ == "__main__":
    RAFTApp().mainloop()
