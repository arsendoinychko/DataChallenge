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
import csv, pathlib, threading, tkinter as tk, os
from collections import Counter, defaultdict
from functools import lru_cache
from tkinter import filedialog, messagebox
from typing import Dict, List, Set, Tuple
from datetime import datetime

import h5py, numpy as np
from scipy.signal import find_peaks

# ────────── Matplotlib – embed with TkAgg ──────────────────────────────
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, TextBox, CheckButtons


# ═════════════════════ low-level helpers (For Batch Mode) ═══════════════════════════════
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
        pk, _ = find_peaks(col_m,       height=mu + const*sd, distance=20)
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
        # Filter for keys that match the expected "Image XXXX" format
        valid_keys = sorted([k for k in f if k.startswith("Image ")], key=lambda x: int(x.split()[-1]))
        for fname in valid_keys:
            frame_names.append(fname)
            img = f[fname][:]
            img = np.clip(img, *np.percentile(img, [vmin, vmax]))
            nonf, frag = detect_peaks(img, n_splits=splits, const=const,
                                      include_valleys=valleys)
            nonfrag_pf.append(nonf)
            frag_pf.append(frag)
            union_pf.append(nonf | frag)

    stable      = set.intersection(*union_pf) if union_pf else set()
    blinking = set.union(*union_pf) - stable if union_pf else set()
    union_nf = set.union(*nonfrag_pf) if nonfrag_pf else set()
    union_fr = set.union(*frag_pf) if frag_pf else set()

    noisy_frames = defaultdict(list)
    # Note: original code associated noisy_frames with non-fragmented peaks.
    # We will use the union of all detected peaks per frame for this mapping.
    for fn, uni in zip(frame_names, union_pf):
        for c in uni:
            noisy_frames[c].append(fn)

    return dict(frame_names=frame_names,
                nonfrag_pf=nonfrag_pf, frag_pf=frag_pf, union_pf=union_pf,
                stable=stable, blinking=blinking,
                union_nf=union_nf, union_fr=union_fr,
                noisy_frames=noisy_frames)


# ────────── CSV & plotting utilities (For Batch Mode) ────────────
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
    all_cols = ana["union_nf"] | ana["union_fr"]
    cnf, cf, bnf, bf = set(), set(), set(), set()
    for col in all_cols:
        is_stable = col in ana["stable"]
        is_frag = col in ana["union_fr"] # A column is fragmented if it ever appears as fragmented.
        
        if is_stable and not is_frag: cnf.add(col)
        elif is_stable and is_frag: cf.add(col)
        elif not is_stable and not is_frag: bnf.add(col)
        elif not is_stable and is_frag: bf.add(col)

    def cat(col: int) -> str:
        if col in cnf: return "Continuous Non Fragmented"
        if col in cf:  return "Continuous Fragmented"
        if col in bnf: return "Blinking Non Fragmented"
        if col in bf: return "Blinking Fragmented"
        return "Uncategorized"

    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Column Number", "Categorization", "Noisy Frames"])
        for col in sorted(all_cols):
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
    cats    = ["Continuous\nNon Frag.", "Continuous\nFrag.",
               "Blinking\nNon Frag.",   "Blinking\nFrag."]
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
        
        # === State variables for Parameter Mode, adapted from script 2 ===
        self.param_all_union_peaks = []
        self.param_noisy_peaks = []
        self.param_fragmented_peaks = []
        self.param_noise_avg_col = defaultdict(list)
        self.param_prev_frame_idx = -1
        self.param_prev_temporal_noisy_peaks = set()
        self.param_prev_temporal_fragmented_peaks = set()


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

    # ---------- PARAMETER MODE (NEW BACKEND) -----------------------------
    def _choose_param_file(self):
        fp = filedialog.askopenfilename(title="Select .h5 stack",
                                         filetypes=[("HDF-5 files", "*.h5"),
                                                    ("All files", "*.*")])
        if fp:
            self.show_parameter_mode(fp)

    # --- Parameter Mode: Backend logic adapted from script 2 ---
    def _param_export_to_csv(self, cats: Dict, params: Dict):
        export_dir = "param_mode_results"
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(export_dir, f"peak_data_{timestamp}.csv")
        
        export_data = []
        
        def add_peak_data(peak, peak_type):
            intensity_values = self.param_noise_avg_col.get(peak, [])
            avg_intensity = np.mean(intensity_values) if intensity_values else 0
            max_intensity = max(intensity_values) if intensity_values else 0
            min_intensity = min(intensity_values) if intensity_values else 0
            std_intensity = np.std(intensity_values) if len(intensity_values) > 1 else 0
            
            export_data.append({
                'Position': peak,
                'Type': peak_type,
                'Average_Intensity': f"{avg_intensity:.2f}",
                'Max_Intensity': f"{max_intensity:.2f}",
                'Min_Intensity': f"{min_intensity:.2f}",
                'Std_Intensity': f"{std_intensity:.2f}",
                'Occurrence_Count': len(intensity_values)
            })

        cat_map = {
            'Stable_Noisy': cats['stable_noisy'],
            'Blinking_Noisy': cats['blinking_noisy'],
            'Blinking_Fragmented': cats['blinking_frag'],
            'Stable_Fragmented': cats['stable_frag'],
        }

        for cat_name, peak_set in cat_map.items():
            for peak in sorted(list(peak_set)):
                add_peak_data(peak, cat_name)
        
        if not export_data:
            print("No peak data to export")
            return

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Position', 'Type', 'Average_Intensity', 'Max_Intensity', 
                          'Min_Intensity', 'Std_Intensity', 'Occurrence_Count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(export_data)
            
        print(f"Parameter mode peak data exported to {filename}")

        info_filename = os.path.join(export_dir, f"peak_data_info_{timestamp}.txt")
        with open(info_filename, 'w') as info_file:
            info_file.write(f"Source file: {params['file_path']}\n")
            info_file.write(f"Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            info_file.write(f"Number of slices: {params['splits']}\n")
            info_file.write(f"Threshold value: {params['const']}\n")
            info_file.write(f"Vmax percentile: {params['vmax_p']}\n")
            info_file.write(f"Vmin percentile: {params['vmin_p']}\n")
            info_file.write(f"Valleys detection: {'Enabled' if params['valleys'] else 'Disabled'}\n")
            info_file.write(f"Total frames processed: {params['n_frames']}\n")
            info_file.write(f"Total unique peaks found: {len(export_data)}\n")
            for cat_name, peak_set in cat_map.items():
                info_file.write(f"{cat_name}: {len(peak_set)}\n")
        
        messagebox.showinfo("Export Complete", f"Results exported to:\n{os.path.abspath(export_dir)}")

    def _param_get_max_peak(self, peaks):
        return max(peaks, key=lambda p: np.mean(self.param_noise_avg_col[p]) if p in self.param_noise_avg_col else 0, default=None)

    def _param_process_frame(self, frame_data, n_splits, const, valleys, distance=10):
        metric_col = defaultdict(list)
        height, _ = frame_data.shape
        split_height = height // n_splits
        all_peaks_of_frame = []

        for i in range(n_splits):
            start_row = i * split_height
            end_row = (i + 1) * split_height if i < n_splits - 1 else height
            split_frame = frame_data[start_row:end_row, :]
            
            column_averages = split_frame.mean(axis=0)
            mean_intensity = np.mean(column_averages)
            std_intensity = np.std(column_averages)
            
            height_threshold_peaks = mean_intensity + const * std_intensity
            peaks, peak_properties = find_peaks(column_averages, height=height_threshold_peaks, distance=distance)
            
            current_split_peaks = set(peaks)
            for j, peak in enumerate(peaks):
                metric_col[peak].append((peak_properties['peak_heights'][j] - mean_intensity) / std_intensity)

            if valleys:
                height_threshold_valley = const * std_intensity - mean_intensity
                vals, valley_properties = find_peaks(-column_averages, height=height_threshold_valley, distance=distance)
                current_split_peaks.update(vals)
                for j, valley in enumerate(vals):
                    metric_col[valley].append((valley_properties['peak_heights'][j] + mean_intensity) / std_intensity)

            all_peaks_of_frame.append(current_split_peaks)

        union_peaks = set.union(*all_peaks_of_frame) if all_peaks_of_frame else set()
        self.param_all_union_peaks.append(union_peaks)
        
        temporal_noisy_peaks = set.intersection(*all_peaks_of_frame) if all_peaks_of_frame else set()
        temporal_fragmented_peaks = union_peaks - temporal_noisy_peaks

        if temporal_noisy_peaks: self.param_noisy_peaks.append(temporal_noisy_peaks)
        if temporal_fragmented_peaks: self.param_fragmented_peaks.append(temporal_fragmented_peaks)

        return temporal_noisy_peaks, temporal_fragmented_peaks, metric_col

    def _param_draw_summary_lines(self, ax_main, ax_noise, ax_fragmented, W, params):
        stable_peaks = set.intersection(*self.param_all_union_peaks) if self.param_all_union_peaks else set()
        blinking_peaks = set.union(*self.param_all_union_peaks) - stable_peaks if self.param_all_union_peaks else set()
        
        all_noisy_peaks = set.union(*self.param_noisy_peaks) if self.param_noisy_peaks else set()
        all_frag_peaks = set.union(*self.param_fragmented_peaks) if self.param_fragmented_peaks else set()
        
        stable_noisy = stable_peaks.intersection(all_noisy_peaks) - all_frag_peaks
        blinking_noisy = blinking_peaks.intersection(all_noisy_peaks) - all_frag_peaks
        blinking_frag = blinking_peaks.intersection(all_frag_peaks)
        stable_frag = stable_peaks.intersection(all_frag_peaks)
        
        categories = {
            "stable_noisy": stable_noisy, "blinking_noisy": blinking_noisy,
            "blinking_frag": blinking_frag, "stable_frag": stable_frag
        }
        
        # Call export before clearing data
        self._param_export_to_csv(categories, params)

        ax_noise.clear()
        ax_fragmented.clear()

        mapping = [
            ("Stable Noisy", stable_noisy, ax_noise, 'black', '-', ax_main, '-'),
            ("Blinking Noisy", blinking_noisy, ax_noise, 'gold', '-', ax_main, '-'),
            ("Stable Fragmented", stable_frag, ax_fragmented, 'green', '-', ax_main, '--'),
            ("Blinking Fragmented", blinking_frag, ax_fragmented, 'purple', '-', ax_main, '--')
        ]

        for label, peaks, ax_sub, color, ls_sub, ax_l, ls_l in mapping:
            if not peaks: continue
            max_peak = self._param_get_max_peak(peaks)
            for i, p in enumerate(sorted(list(peaks))):
                y = np.mean(self.param_noise_avg_col.get(p, [0]))
                is_first = (i == 0)
                ax_sub.vlines(p, 0, y, color=color, linestyle=ls_sub, label=label if is_first else "")
                ax_sub.text(p, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8, color='red')
                ax_l.axvline(p, color=color, linestyle=ls_l, label=label if is_first else "")
                if p == max_peak:
                    ax_sub.plot(p, y, 'ro') # Mark max peak in sub-plot
                    # Find a suitable y-position for the marker in the main plot
                    main_y = ax_l.get_ylim()[0] + 0.95 * (ax_l.get_ylim()[1] - ax_l.get_ylim()[0])
                    ax_l.plot(p, main_y, 'ro')
        
        ax_noise.set_title(f"Non-Fragmented Types ({len(stable_noisy | blinking_noisy)})")
        ax_fragmented.set_title(f"Fragmented Types ({len(stable_frag | blinking_frag)})")
        
        for ax in [ax_noise, ax_fragmented]:
            ax.set_xlim(0, W); ax.set_ylim(0, 20); ax.grid(True); ax.legend()
        
        ax_main.set_title("Labeled Frame (Final Summary)")
        ax_main.legend()
        
    def show_parameter_mode(self, file_path: str):
        self._clear()
        tk.Button(self.content, text="< Back to Home", font=("Segoe UI", 12),
                  command=self.show_home).pack(anchor="w", padx=6, pady=6)

        frm_plot = tk.Frame(self.content)
        frm_plot.pack(fill="both", expand=True)
        
        # --- Reset State ---
        self.param_all_union_peaks.clear()
        self.param_noisy_peaks.clear()
        self.param_fragmented_peaks.clear()
        self.param_noise_avg_col.clear()
        self.param_prev_frame_idx = -1
        self.param_prev_temporal_noisy_peaks = set()
        self.param_prev_temporal_fragmented_peaks = set()

        # ---- Prepare data ----
        with h5py.File(file_path, "r") as f:
            all_frames = sorted([k for k in f.keys() if k.startswith("Image ")], key=lambda x: int(x.split()[-1]))
        n_frames = len(all_frames)
        
        # ---- Build Matplotlib figure ----
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(
            2, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [3, 1]}
        )
        plt.subplots_adjust(bottom=0.25, hspace=0.3, wspace=0.2)
        canvas = FigureCanvasTkAgg(fig, master=frm_plot)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # ---- Widgets inside figure ----
        ax_sf = plt.axes([0.25, 0.10, 0.65, 0.03])
        s_frame = Slider(ax_sf, "Frame #", 0, n_frames - 1, valinit=0, valstep=1)
        ax_ss = plt.axes([0.25, 0.05, 0.65, 0.03])
        s_split = Slider(ax_ss, "Number of Slices", 1, 10, valinit=4, valstep=1)
        ax_sc = plt.axes([0.25, 0.01, 0.65, 0.03])
        s_const = Slider(ax_sc, "Threshold", 0.0, 20.0, valinit=4.0, valstep=0.1)

        ax_vmin = plt.axes([0.05, 0.10, 0.05, 0.03])
        t_vmin = TextBox(ax_vmin, "Vmin (%)", initial="1")
        ax_vmax = plt.axes([0.05, 0.05, 0.05, 0.03])
        t_vmax = TextBox(ax_vmax, "Vmax (%)", initial="99")

        ax_chk = plt.axes([0.05, 0.01, 0.1, 0.03])
        chk_val = CheckButtons(ax_chk, ["Show Valleys"], [False])

        # ---- Update Callback ----
        def _update(_=None):
            frame_idx = int(s_frame.val)
            
            # If frame index is the same but other params changed, pop previous data
            if frame_idx == self.param_prev_frame_idx:
                if self.param_all_union_peaks: self.param_all_union_peaks.pop()
                if self.param_noisy_peaks: self.param_noisy_peaks.pop()
                if self.param_fragmented_peaks: self.param_fragmented_peaks.pop()
                for peak in self.param_prev_temporal_noisy_peaks:
                    if self.param_noise_avg_col[peak]: self.param_noise_avg_col[peak].pop()
                for peak in self.param_prev_temporal_fragmented_peaks:
                    if self.param_noise_avg_col[peak]: self.param_noise_avg_col[peak].pop()

            n_s = int(s_split.val)
            const = s_const.val
            try:
                pvmin = float(t_vmin.text)
                pvmax = float(t_vmax.text)
            except ValueError:
                pvmin, pvmax = 1, 99 # Fallback
            valleys = chk_val.get_status()[0]

            for a in (ax1, ax2, ax3, ax4): a.clear()

            frame_name = all_frames[frame_idx]
            with h5py.File(file_path, 'r') as f:
                frame_data = f[frame_name][:]
            
            vmin_val, vmax_val = np.percentile(frame_data, [pvmin, pvmax])
            img_clipped = np.clip(frame_data, vmin_val, vmax_val)
            W = img_clipped.shape[1]

            nf_now, fr_now, met_now = self._param_process_frame(img_clipped, n_s, const, valleys)
            self.param_prev_temporal_noisy_peaks = nf_now.copy()
            self.param_prev_temporal_fragmented_peaks = fr_now.copy()
            
            ax1.imshow(img_clipped, cmap="gray", vmin=vmin_val, vmax=vmax_val, aspect='auto')
            ax1.set_title(f"Original Frame: {frame_name}"); ax1.axis("off")
            ax2.imshow(img_clipped, cmap="gray", vmin=vmin_val, vmax=vmax_val, aspect='auto')
            ax2.set_title("Labeled Frame"); ax2.axis("off")

            # Plot current frame's peaks
            for p in nf_now:
                y = np.mean(met_now.get(p, [0])); self.param_noise_avg_col[p].append(y)
                ax2.axvline(p, color="red")
                ax3.vlines(p, 0, y, color="red")
                ax3.text(p, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
            for p in fr_now:
                y = np.mean(met_now.get(p, [0])); self.param_noise_avg_col[p].append(y)
                ax2.axvline(p, color="blue", linestyle="--")
                ax4.vlines(p, 0, y, color="blue")
                ax4.text(p, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

            ax3.set_title(f"Non-Frag. peaks ({len(nf_now)})")
            ax4.set_title(f"Frag. peaks ({len(fr_now)})")
            for a in (ax3, ax4): a.set_xlim(0, W); a.set_ylim(0, 20); a.grid(True)
            
            # On the last frame, run the final analysis
            if frame_idx == n_frames - 1:
                params_dict = {
                    'file_path': file_path, 'splits': n_s, 'const': const,
                    'vmin_p': pvmin, 'vmax_p': pvmax, 'valleys': valleys, 'n_frames': n_frames
                }
                self._param_draw_summary_lines(ax2, ax3, ax4, W, params_dict)
                # Clear state for a potential re-run by moving slider back
                self.param_all_union_peaks.clear()
                self.param_noisy_peaks.clear()
                self.param_fragmented_peaks.clear()
                self.param_noise_avg_col.clear()

            self.param_prev_frame_idx = frame_idx
            canvas.draw_idle()

        # --- Initial call and widget connections ---
        s_frame.on_changed(_update)
        s_split.on_changed(_update)
        s_const.on_changed(_update)
        t_vmin.on_submit(_update)
        t_vmax.on_submit(_update)
        chk_val.on_clicked(_update)
        _update()

    # ---------- BATCH MODE (UNCHANGED) -----------------------------------
    def _choose_batch_folder(self):
        dp = filedialog.askdirectory(title="Select directory with .h5 files")
        if dp:
            self.show_batch_mode(dp)

    def show_batch_mode(self, dir_path: str):
        self._clear()
        tk.Button(self.content, text="< Back to Home", font=("Segoe UI", 12),
                  command=self.show_home).pack(anchor="w", padx=6, pady=6)

        frm = tk.Frame(self.content)
        frm.pack(pady=20)

        defaults = dict(threshold="4.0", vmin="1", vmax="99", splits="4")
        entries = {}
        for r, (lbl, default) in enumerate(defaults.items()):
            tk.Label(frm, text=f"{lbl.capitalize()}:", font=("Segoe UI", 14)
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

        out_root = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "batch_results"
        out_root.mkdir(exist_ok=True)

        for h5 in h5s:
            ana = analyse_stack(h5, const=thr, vmin=vmin, vmax=vmax,
                                splits=splits, valleys=valleys)
            if not ana["frame_names"]: continue # Skip empty or invalid H5 files

            out = out_root / h5.stem
            out.mkdir(parents=True, exist_ok=True)
            write_frame_summary(out/"frame_fragmentation_summary.csv",
                                frame_names=ana["frame_names"],
                                frag_pf=ana["frag_pf"], union_pf=ana["union_pf"])
            write_column_summary(out/"column_category_summary.csv", ana)

            all_noisy = ana["union_nf"] | ana["union_fr"]
            
            cnf = all_noisy.intersection(ana["stable"]) - ana["union_fr"]
            cf  = ana["union_fr"].intersection(ana["stable"])
            bnf = all_noisy.intersection(ana["blinking"]) - ana["union_fr"]
            bf  = ana["union_fr"].intersection(ana["blinking"])

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
    app = RAFTApp()
    app.mainloop()
