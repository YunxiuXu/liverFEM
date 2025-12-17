import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_anisotropy():
    if not os.path.exists("ani_hard.txt") or not os.path.exists("ani_soft.txt"):
        print("Waiting for data files: ani_hard.txt and ani_soft.txt")
        return

    try:
        # Load data (Time, Force)
        # Skip first row (header) if it's text
        df_hard = pd.read_csv("ani_hard.txt", delim_whitespace=True)
        df_soft = pd.read_csv("ani_soft.txt", delim_whitespace=True)
        
        # Plot Force vs Time (assuming linear displacement with time, Time ~= Displacement)
        plt.figure(figsize=(10, 6))
        plt.plot(df_hard['Time'], df_hard['Force'], label='Hard Axis (E=0.5MPa)', color='red', linewidth=2)
        plt.plot(df_soft['Time'], df_soft['Force'], label='Soft Axis (E=0.1MPa)', color='blue', linewidth=2)
        
        plt.title('Anisotropic Force Response (Automated Experiment)', fontsize=14)
        plt.xlabel('Time / Displacement (s)', fontsize=12)
        plt.ylabel('Reaction Force (N)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('Figure5_Anisotropy.png')
        print("Generated Figure5_Anisotropy.png")
        
    except Exception as e:
        print(f"Error plotting anisotropy: {e}")

def plot_volume():
    if not os.path.exists("vol_stability.txt"):
        print("Waiting for vol_stability.txt")
        return

    try:
        df_vol = pd.read_csv("vol_stability.txt", delim_whitespace=True)
        
        plt.figure(figsize=(10, 6))
        # Convert to percentage
        plt.plot(df_vol['Time'], df_vol['VolumeRatio'] * 100, label='Our Method (GB-cFEM)', color='green', linewidth=2)
        
        # Add a dummy XPBD line for comparison if needed (dashed)
        # plt.plot(df_vol['Time'], np.random.normal(-5, 1, len(df_vol)), label='XPBD (Simulated)', linestyle='--', color='gray')

        plt.title('Volume Preservation under Large Deformation (nu=0.49)', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Volume Change (%)', fontsize=12)
        plt.ylim(-5, 5) # Focus on near-zero
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('Figure4_VolumeStability.png')
        print("Generated Figure4_VolumeStability.png")
        
    except Exception as e:
        print(f"Error plotting volume: {e}")

if __name__ == "__main__":
    plot_anisotropy()
    plot_volume()
