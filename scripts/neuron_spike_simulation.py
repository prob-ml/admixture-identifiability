#!/usr/bin/env python3
"""
Plot overlapping neuron spikes from real neural data.
This script loads actual spike waveforms and plots them with overlapping times.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure plots directory exists
plots_dir = Path("./plots")
plots_dir.mkdir(exist_ok=True)

def load_spike_data(npz_path='vendor/ultra_choices.npz'):
    """Load spike waveforms from npz file."""
    data = np.load(npz_path, allow_pickle=True)
    waveforms = data['waveforms']  # shape (4, 82)
    acronyms = data['acronyms']    # shape (4,)
    sampling_rate = data['sampling_rate']  # scalar (30000 Hz)
    return waveforms, acronyms, sampling_rate

def main():
    """Main plotting function."""
    print("Loading real neuron spike data...")

    # Load real spike waveforms
    waveforms, acronyms, sampling_rate = load_spike_data()
    n_neurons, n_samples = waveforms.shape

    # Create time vector in ms
    dt_ms = 1000.0 / sampling_rate  # time step in milliseconds
    duration_ms = n_samples * dt_ms
    time = np.arange(n_samples) * dt_ms

    # Define spike times (ms) for overlapping spikes
    spike_times = [9.0, 10.3, 10.6, 13.0]  # overlapping

    # Create combined signal by time-shifting individual waveforms
    total_duration = spike_times[-1] + duration_ms + 2.0  # extra padding
    total_samples = int(total_duration / dt_ms)
    combined_signal = np.zeros(total_samples)
    time_full = np.arange(total_samples) * dt_ms

    # Add each spike at its designated time
    individual_signals = []
    for i in range(n_neurons):
        spike_signal = np.zeros(total_samples)
        start_idx = int(spike_times[i] / dt_ms)
        spike_signal[start_idx:start_idx + n_samples] = waveforms[i]
        individual_signals.append(spike_signal)
        combined_signal += spike_signal

    # Create plot
    print("Creating plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.5), sharey=True)

    # Left plot: Observable signal (sum of all spikes)
    ax1.plot(time_full, combined_signal, '-', color='black', linewidth=1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (µV)')
    ax1.set_title('Observable Signal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([10, 15])

    # Right plot: Individual unit contributions
    colors = ['blue', 'red', 'green', 'orange']
    region_labels = [f'Neuron {i+1}' for i in range(n_neurons)]
    for i in range(n_neurons):
        ax2.plot(time_full, individual_signals[i], '-',
                label=region_labels[i], color=colors[i], alpha=0.8, linewidth=1)

    ax2.set_xlabel('Time (ms)')
    ax2.set_title('Individual Unit Contributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([10, 15])

    plt.tight_layout()

    # Save plot
    output_file = plots_dir / "overlapping_neuron_spikes.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Print summary statistics
    print("\nData Summary:")
    print(f"Sampling rate: {sampling_rate/1000:.1f} kHz")
    print(f"Spike duration: {duration_ms:.2f} ms")
    print(f"Number of neurons: {n_neurons}")
    for i, (acronym, spike_time) in enumerate(zip(acronyms, spike_times)):
        peak_amp = np.max(np.abs(waveforms[i]))
        print(f"{acronym} spike at {spike_time} ms, peak amplitude: {peak_amp:.1f} µV")
    print(f"Combined peak: {np.max(np.abs(combined_signal)):.1f} µV")

if __name__ == "__main__":
    main()