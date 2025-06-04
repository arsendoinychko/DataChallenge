import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import matplotlib.widgets as widgets
from collections import defaultdict
import argparse
import csv
from datetime import datetime
import os

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Analysis Tool')
    parser.add_argument('--file', '-f', type=str, 
                        default=r"E:\Data challenge\BRUNNER\REF_N1.h5",
                        help='Path to the H5 file')
    return parser.parse_args()

# Get arguments
args = parse_arguments()
file_path = args.file

# Global variables
__all_union_of_peaks_of_every_frame = []
noisy_peaks = list()
fragmented_peaks = list()
percentage_of_vmax = 99
percentage_of_vmin = 1
n_splits = 4
const = 4
noise_avg_col = defaultdict(list)
previous_frame_idx = -1
prev_temporal_noisy_peaks = set() 
prev_temporal_fragmented_peaks = set()


# Add this function to export the peak data to CSV
def export_to_csv(
    stable_noisy_peaks,
    blinking_noisy_peaks,
    blinking_fragmented_peaks,
    stable_fragmented_peaks
):
    # Create a directory for exports if it doesn't exist
    export_dir = "peak_exports"
    os.makedirs(export_dir, exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(export_dir, f"peak_data_{timestamp}.csv")
    
    
    # Prepare data for export
    export_data = []
    
    # Helper function to process each peak
    def add_peak_data(peak, peak_type):
        intensity_values = noise_avg_col.get(peak, [])
        avg_intensity = np.mean(intensity_values) if intensity_values else 0
        max_intensity = max(intensity_values) if intensity_values else 0
        min_intensity = min(intensity_values) if intensity_values else 0
        std_intensity = np.std(intensity_values) if len(intensity_values) > 1 else 0
        
        export_data.append({
            'Position': peak,
            'Type': peak_type,
            'Average_Intensity': avg_intensity,
            'Max_Intensity': max_intensity,
            'Min_Intensity': min_intensity,
            'Std_Intensity': std_intensity,
            'Occurrence_Count': len(intensity_values)
        })
    
    # Add all peak types to the export data
    for peak in stable_noisy_peaks:
        add_peak_data(peak, 'Stable_Noisy')
    
    for peak in blinking_noisy_peaks:
        add_peak_data(peak, 'Blinking_Noisy')
    
    for peak in blinking_fragmented_peaks:
        add_peak_data(peak, 'Blinking_Fragmented')
    
    for peak in stable_fragmented_peaks:
        add_peak_data(peak, 'Stable_Fragmented')
    
    # Write to CSV
    if export_data:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Position', 'Type', 'Average_Intensity', 'Max_Intensity', 
                         'Min_Intensity', 'Std_Intensity', 'Occurrence_Count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header and file info
            writer.writeheader()
            
            # Write all the peak data
            for row in export_data:
                writer.writerow(row)
            
        print(f"Peak data exported to {filename}")
        
        # Also export additional information file
        info_filename = os.path.join(export_dir, f"peak_data_info_{timestamp}.txt")
        with open(info_filename, 'w') as info_file:
            info_file.write(f"Source file: {file_path}\n")
            info_file.write(f"Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            info_file.write(f"Number of slices: {n_splits}\n")
            info_file.write(f"Threshold value: {const}\n")
            info_file.write(f"Vmax percentile: {percentage_of_vmax}\n")
            info_file.write(f"Vmin percentile: {percentage_of_vmin}\n")
            valleys_status = 'Enabled' if checkbox_valleys.get_status()[0] else 'Disabled'
            info_file.write(f"Valleys detection: {valleys_status}\n")
            info_file.write(f"Total frames processed: {max_frame - 1}\n")
            info_file.write(f"Total peaks found: {len(export_data)}\n")
            info_file.write(f"Stable noisy peaks: {len(stable_noisy_peaks)}\n")
            info_file.write(f"Blinking noisy peaks: {len(blinking_noisy_peaks)}\n")
            info_file.write(f"Stable fragmented peaks: {len(stable_fragmented_peaks)}\n")
            info_file.write(f"Blinking fragmented peaks: {len(blinking_fragmented_peaks)}\n")
    else:
        print("No peak data to export")
# Function to process peaks for a given frame

# Find max peaks
def get_max_peak(peaks):
    return max(peaks, key=lambda p: np.mean(noise_avg_col[p]) if p in noise_avg_col else 0, default=None)

def process_frame_peaks(frame_data, checkbox_valleys_state):
    global __all_union_of_peaks_of_every_frame, noisy_peaks, fragmented_peaks

    metric_col = defaultdict(list)  # Store metrics for each column

    height, width = frame_data.shape
    split_height = height // n_splits

    all_peaks_of_frame = []
    all_column_averages = []  # Save column averages per split

    for i in range(n_splits):
        start_row = i * split_height
        end_row = (i + 1) * split_height if i < n_splits - 1 else height
        split_frame = frame_data[start_row:end_row, :]

        column_averages = split_frame.mean(axis=0)
        all_column_averages.append(column_averages)

        mean_intensity = np.mean(column_averages)
        std_intensity = np.std(column_averages)
        height_threshold_peaks = mean_intensity + const * std_intensity
        height_threshold_valley = const * std_intensity - mean_intensity  # TODO: Adjust this threshold as needed
        # prominence_threshold = const * std_intensity

        # Detect peaks (positive peaks)
        peaks, peak_properties = find_peaks(column_averages, height=height_threshold_peaks)
        # Detect valleys (negative peaks)
        if checkbox_valleys_state:
            valleys, valley_properties = find_peaks(-column_averages, height = height_threshold_valley)
            all_peaks_of_frame.append(set(np.concatenate([peaks, valleys])))  # Store peaks and valleys as a set
            # print(f"Split {i+1}: Found {len(peaks)} peaks and {len(valleys) if checkbox_valleys_state else 0} valleys")
            # Store metrics for each peak
            for i, peak in enumerate(peaks):
                metric_col[peak].append((peak_properties['peak_heights'][i]-mean_intensity)/std_intensity)  # Store peak heights for peaks
            # Store metrics for valleys if valleys are detected
            for i, valley in enumerate(valleys):
                metric_col[valley].append((valley_properties['peak_heights'][i] + mean_intensity)/std_intensity)  # Store valley heights for valleys


        else:
            all_peaks_of_frame.append(set(peaks))
            # Store metrics for each peak
            for i, peak in enumerate(peaks):
                metric_col[peak].append((peak_properties['peak_heights'][i]-mean_intensity)/std_intensity)
        # print(f"Split {i+1}: Found {len(peaks)} peaks")
        

    # Union and intersection of peaks
    union_peaks = set().union(*all_peaks_of_frame)
    __all_union_of_peaks_of_every_frame.append(union_peaks)
    temporal_noisy_peaks = set.intersection(*all_peaks_of_frame) if all_peaks_of_frame else set()
    temporal_fragmented_peaks = union_peaks - temporal_noisy_peaks

    # Update global noisy and fragmented peaks
    if temporal_noisy_peaks:
        noisy_peaks.append(temporal_noisy_peaks)
    if temporal_fragmented_peaks:
        fragmented_peaks.append(temporal_fragmented_peaks)
    # print(fragmented_peaks)



    return temporal_noisy_peaks, temporal_fragmented_peaks, metric_col

# Function to update plot based on slider value
def update(val):
    global n_splits, const, previous_frame_idx, prev_temporal_noisy_peaks, prev_temporal_fragmented_peaks, frame_width
    frame_idx = int(slider.val)
    if frame_idx == previous_frame_idx:
        if __all_union_of_peaks_of_every_frame:
           __all_union_of_peaks_of_every_frame.pop()  # Remove the last entry if the frame index hasn't changed
        if noisy_peaks:
            noisy_peaks.pop()
        if fragmented_peaks:
            fragmented_peaks.pop() # Remove the last entry if the frame index hasn't changed
        for peak in prev_temporal_noisy_peaks:
            noise_avg_col[peak].pop()  # Remove the last entry if the frame index hasn't changed
        for peak in prev_temporal_fragmented_peaks:
            noise_avg_col[peak].pop()
        # print(fragmented_peaks)
    n_splits = int(slider_n_splits.val)
    const = slider_const.val
    percentage_of_vmax = float(textbox_vmax.text)
    percentage_of_vmin = int(textbox_vmin.text)
    checkbox_valleys_state = checkbox_valleys.get_status()[0]  # Get the state of the checkbox
    # print(f"Frame: {frame_idx}, n_splits: {n_splits}, const: {const}")
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    frame_name = f"Image {frame_idx:04d}"
    with h5py.File(file_path, 'r') as f:
        if frame_name in f:
            frame_data = f[frame_name][:]
            vmin = np.percentile(frame_data, percentage_of_vmin)  # Set vmin to 1st percentile
            vmax = np.percentile(frame_data, percentage_of_vmax)
            frame_clipped = np.clip(frame_data, vmin, vmax)
            # frame_clipped = frame_data.copy()  # Use a copy to avoid modifying the original data
            frame_width = frame_clipped.shape[1]
            
            # Process peaks for this frame
            temporal_noisy_peaks, temporal_fragmented_peaks, metric_col = process_frame_peaks(frame_clipped, checkbox_valleys_state)
            prev_temporal_noisy_peaks = temporal_noisy_peaks.copy()  # Store for the next frame
            prev_temporal_fragmented_peaks = temporal_fragmented_peaks.copy()
            
            # Plot original frame without lines (top left)
            ax1.imshow(frame_clipped, cmap="gray", vmin = vmin , vmax=vmax)
            ax1.axis('off')
            ax1.set_title("Original Frame")
            
            # Plot original frame without lines (bottom left)
            ax2.imshow(frame_clipped, cmap="gray", vmin = vmin , vmax=vmax)
            ax2.axis('off')
            ax2.set_title("Labeled Frame")


            
            # Plot noisy peaks (top right)
            if frame_idx < max_frame - 1:
                if temporal_noisy_peaks:
                    for i, peak in enumerate(temporal_noisy_peaks):
                        ax2.axvline(x=peak, color='red', linestyle='-', linewidth=1, label='Noisy Peaks')
                        y_value = np.mean(metric_col[peak]) if peak in metric_col else 0  # Ensure y_value is non-negative
                        noise_avg_col[peak].append(y_value)  # Store average for noisy peaks
                        ax3.vlines(x=peak, ymin=0, ymax=y_value, color='red', linestyle='-', linewidth=1, label="Noisy Peaks" if peak == min(temporal_noisy_peaks, default=None) else "")
                        ax3.vlines(x=peak, ymin=0, ymax=y_value, color='red', linestyle='-', linewidth=1, label='Noisy Peaks')
                        ax3.text(peak, y_value, f'{y_value:.1f}', fontsize=8, ha='center', va='bottom', color='red')
                    ax3.legend()
                ax3.set_title("Noisy Peaks")
                ax3.set_ylim( 0 , 20)
                ax3.set_xlim(0, frame_clipped.shape[1])
                ax3.grid(True)
                
                # Plot fragmented peaks (bottom right)
                if temporal_fragmented_peaks:
                    for i, peak in enumerate(temporal_fragmented_peaks):
                        ax2.axvline(x=peak, color='blue', linestyle='--', linewidth=1, label='Fragmented Peaks')
                        y_value = np.mean(metric_col[peak]) if peak in metric_col else 0  # Ensure y_value is non-negative
                        noise_avg_col[peak].append(y_value)
                        ax4.vlines(x=peak, ymin=0, ymax=y_value, color='blue', linestyle='-', linewidth=1, label='Fragmented Peaks' if peak == min(temporal_fragmented_peaks, default=None) else "")
                        ax4.text(peak, y_value, f'{y_value:.1f}', fontsize=8, ha='center', va='bottom', color='blue')
                    ax4.legend()
                ax4.set_title("Fragmented Peaks")
                ax4.set_ylim( 0 , 20)
                ax4.set_xlim(0, frame_clipped.shape[1])
                ax4.grid(True)
 
            
            # Draw lines on the bottom left plot only if we've reached the last frame
            if frame_idx == max_frame - 1:
                if __all_union_of_peaks_of_every_frame:
                    draw_lines(ax2, ax3, ax4)
                    # Clear global variables for the next run
                    noisy_peaks.clear()
                    fragmented_peaks.clear()
                    __all_union_of_peaks_of_every_frame.clear()
                    noise_avg_col.clear()
            
            previous_frame_idx = frame_idx     
            fig.canvas.draw_idle()

# Function to draw lines on the bottom left plot
def draw_lines(ax_main, ax_noise, ax_fragmented):
    # Compute peak categories
    stable_peaks = set.intersection(*__all_union_of_peaks_of_every_frame) if __all_union_of_peaks_of_every_frame else set()
    blinking_peaks = set().union(*__all_union_of_peaks_of_every_frame) - stable_peaks if __all_union_of_peaks_of_every_frame else set()
    stable_noisy_peaks = set.intersection(stable_peaks, set().union(*noisy_peaks))
    blinking_noisy_peaks = set.intersection(blinking_peaks, set().union(*noisy_peaks))
    blinking_fragmented_peaks = set.intersection(blinking_peaks, set().union(*fragmented_peaks))
    stable_fragmented_peaks = set.intersection(stable_peaks, set().union(*fragmented_peaks))
    
    # Clear existing lines (if any)
    # for ax in [ax_main, ax_noise, ax_fragmented]:
    #     for artist in ax.lines + ax.texts:
    #         artist.remove()

    # Export peak data to CSV
    export_to_csv(stable_noisy_peaks, blinking_noisy_peaks, blinking_fragmented_peaks, stable_fragmented_peaks)


    max_stable_noisy = get_max_peak(stable_noisy_peaks)
    max_blinking_noisy = get_max_peak(blinking_noisy_peaks)
    max_blinking_fragmented = get_max_peak(blinking_fragmented_peaks)
    max_stable_fragmented = get_max_peak(stable_fragmented_peaks)

    
    # Draw lines with labels
    for peak in stable_noisy_peaks:
        y = np.mean(noise_avg_col[peak]) if peak in noise_avg_col else 0
        ax_noise.vlines(x=peak, ymin=0, ymax=y, color='black', linestyle='-', linewidth=1,
                        label='Stable Noisy' if peak == min(stable_noisy_peaks, default=None) else "")
        ax_noise.text(peak, y, f'{y:.1f}', fontsize=8, ha='center', va='bottom', color='red')
        ax_noise.set_ylim( 0 , 20)
        ax_noise.set_xlim(0, frame_width)
        if peak == max_stable_noisy:
            ax_noise.plot(peak, y, 'ro')
    

    for peak in blinking_noisy_peaks:
        y = np.mean(noise_avg_col[peak]) if peak in noise_avg_col else 0
        ax_noise.vlines(x=peak, ymin=0, ymax=y, color='yellow', linestyle='-', linewidth=1,
                        label='Blinking Noisy' if peak == min(blinking_noisy_peaks, default=None) else "")
        ax_noise.text(peak, y, f'{y:.1f}', fontsize=8, ha='center', va='bottom', color='red')
        if peak == max_blinking_noisy:
            ax_noise.plot(peak, y, 'ro')

    ax_noise.set_title(f"Number of Noisy Columns Appeared: {sum(map(len, noisy_peaks))}")
    ax_noise.legend()



    
    # Draw lines with labels
    for peak in blinking_fragmented_peaks:
        y = np.mean(noise_avg_col[peak]) if peak in noise_avg_col else 0
        ax_fragmented.vlines(x=peak, ymin=0, ymax=y, color='purple', linestyle='-', linewidth=1,
                            label='Blinking Fragmented' if peak == min(blinking_fragmented_peaks, default=None) else "")
        ax_fragmented.text(peak, y, f'{y:.1f}', fontsize=8, ha='center', va='bottom', color='red')
        ax_fragmented.set_ylim( 0 , 20)
        ax_fragmented.set_xlim(0, frame_width)
        if peak == max_blinking_fragmented:
            ax_fragmented.plot(peak, y, 'ro')

    for peak in stable_fragmented_peaks:
        y = np.mean(noise_avg_col[peak]) if peak in noise_avg_col else 0
        ax_fragmented.vlines(x=peak, ymin=0, ymax=y, color='green', linestyle='-', linewidth=1,
                            label='Stable Fragmented' if peak == min(stable_fragmented_peaks, default=None) else "")
        ax_fragmented.text(peak, y, f'{y:.1f}', fontsize=8, ha='center', va='bottom', color='red')
        if peak == max_stable_fragmented:
            ax_fragmented.plot(peak, y, 'ro')

    ax_fragmented.set_title(f"Number of Fragmented Columns Appeared: {sum(map(len, fragmented_peaks))}")
    ax_fragmented.legend()



    # Draw lines with labels
    for peak in stable_noisy_peaks:
        ax_main.axvline(x=peak, color='black', linestyle='-', linewidth=1,
                        label='Stable Noisy' if peak == min(stable_noisy_peaks, default=None) else "")
        if peak == max_stable_noisy:
            y = np.mean(noise_avg_col[peak]) if peak in noise_avg_col else 0
            ax_main.plot(peak, y, 'ro')

    for peak in blinking_noisy_peaks:
        ax_main.axvline(x=peak, color='yellow', linestyle='-', linewidth=1,
                        label='Blinking Noisy' if peak == min(blinking_noisy_peaks, default=None) else "")
        if peak == max_blinking_noisy:
            y = np.mean(noise_avg_col[peak]) if peak in noise_avg_col else 0
            ax_main.plot(peak, y, 'ro')

    for peak in blinking_fragmented_peaks:
        ax_main.axvline(x=peak, color='purple', linestyle='--', linewidth=1,
                        label='Blinking Fragmented' if peak == min(blinking_fragmented_peaks, default=None) else "")
        if peak == max_blinking_fragmented:
            y = np.mean(noise_avg_col[peak]) if peak in noise_avg_col else 0
            ax_main.plot(peak, y, 'ro')

    for peak in stable_fragmented_peaks:
        ax_main.axvline(x=peak, color='green', linestyle='--', linewidth=1,
                        label='Stable Fragmented' if peak == min(stable_fragmented_peaks, default=None) else "")
        if peak == max_stable_fragmented:
            y = np.mean(noise_avg_col[peak]) if peak in noise_avg_col else 0
            ax_main.plot(peak, y, 'ro')


    ax_main.legend()



# Set up the plot
with h5py.File(file_path, 'r') as f:
    frame_names = [name for name in f.keys() if name.startswith("Image")]
    max_frame = int(max(frame_names, key=lambda x: int(x.split()[-1]))[6:]) + 1

fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
plt.subplots_adjust(bottom=0.25)

# Add slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = widgets.Slider(ax_slider, 'Frame', 1, max_frame - 1, valinit=0, valstep=1)

# Slider for n_splits
ax_slider_n_splits = plt.axes([0.25, 0.05, 0.65, 0.03])
slider_n_splits = widgets.Slider(ax_slider_n_splits, 'Number of Slices', 1, 10, valinit=n_splits, valstep=1)

# Slider for const
ax_slider_const = plt.axes([0.25, 0.01, 0.65, 0.03])
slider_const = widgets.Slider(ax_slider_const, 'Threshold', 0.0, 20.0, valinit=const, valstep=0.1)

# Input percentage of vmax and vmin on the plot
ax_vmax = plt.axes([0.05, 0.1, 0.05, 0.03])
ax_vmin = plt.axes([0.05, 0.05, 0.05, 0.03])

textbox_vmin = widgets.TextBox(ax_vmax, 'Vmin (%)', initial=str(percentage_of_vmin))
textbox_vmax = widgets.TextBox(ax_vmin, 'Vmax (%)', initial=str(percentage_of_vmax))

# Tick box for peaks or valleys
ax_valleys = plt.axes([0.05, 0.01, 0.1, 0.03])
checkbox_valleys = widgets.CheckButtons(ax_valleys, ['Show Valleys'], [False])



# Connect the slider to the update function
slider.on_changed(update)
slider_n_splits.on_changed(update)
slider_const.on_changed(update)
textbox_vmax.on_submit(update)
textbox_vmin.on_submit(update)
checkbox_valleys.on_clicked(update)


# Initial plot
update(0)

plt.show()