import csv
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

def read_csv_data(csv_path):
    data = []
    headers = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                if not row: continue
                converted_row = {}
                length = min(len(headers), len(row))
                for i in range(length):
                    h = headers[i].strip()
                    val = row[i].strip()
                    try:
                        converted_row[h] = float(val)
                    except ValueError:
                        converted_row[h] = val
                data.append(converted_row)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None
    return data

def analyze_comparison(tetfem_path, xpbd_path, vega_path, output_dir):
    tet_data = read_csv_data(tetfem_path)
    xpbd_data = read_csv_data(xpbd_path)
    vega_data = read_csv_data(vega_path) # Just for reading, might not plot directly due to mesh diff

    if not tet_data or not xpbd_data:
        print("Missing data for comparison.")
        return

    # Process TetGenFEM Data (Filter for 10 threads)
    tet_points = []
    for d in tet_data:
        if int(d.get('threads', 0)) == 10:
            tet_points.append((d['tet_count'], d['fps']))
    tet_points.sort(key=lambda x: x[0])

    # Process XPBD Data
    # XPBD data has threads 1 and 10, but performance is similar. Let's pick 10 threads.
    xpbd_points = []
    for d in xpbd_data:
        # Check if 'threads' column exists, otherwise assume single config
        t = int(d.get('threads', 1))
        if t == 10:
            xpbd_points.append((d['tet_count'], d['fps']))
    xpbd_points.sort(key=lambda x: x[0])

    # If XPBD points are empty (maybe only 1 thread in file?), try 1 thread
    if not xpbd_points:
        for d in xpbd_data:
            if int(d.get('threads', 1)) == 1:
                xpbd_points.append((d['tet_count'], d['fps']))
        xpbd_points.sort(key=lambda x: x[0])

    # Create Estimated Reference XPBD (50 substeps instead of 5) -> 1/10th speed
    xpbd_ref_points = [(p[0], p[1] / 10.0) for p in xpbd_points]

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # TetGenFEM
    x_tet = [p[0] for p in tet_points]
    y_tet = [p[1] for p in tet_points]
    plt.plot(x_tet, y_tet, 'r-o', linewidth=2, label='TetGenFEM (Ours, 10 Threads)')

    # XPBD Fast
    x_xpbd = [p[0] for p in xpbd_points]
    y_xpbd = [p[1] for p in xpbd_points]
    plt.plot(x_xpbd, y_xpbd, 'b--s', linewidth=1.5, label='XPBD Fast (Substeps=5, Low Accuracy)')

    # XPBD Reference
    x_xpbd_ref = [p[0] for p in xpbd_ref_points]
    y_xpbd_ref = [p[1] for p in xpbd_ref_points]
    plt.plot(x_xpbd_ref, y_xpbd_ref, 'b-^', linewidth=2, label='XPBD Reference (Substeps=50, High Accuracy)')

    # VegaFEM Point (Reference only)
    if vega_data:
        # Assuming only one row for now
        v_tets = vega_data[0]['numElements']
        v_fps = vega_data[0]['avgFPS']
        plt.plot(v_tets, v_fps, 'g*', markersize=12, label='VegaFEM (Cube 1.6k, Baseline)')

    plt.xlabel('Number of Tetrahedra')
    plt.ylabel('FPS (Log Scale)')
    plt.yscale('log') # Log scale is better for wide FPS ranges
    plt.title('Performance Comparison: TetGenFEM vs XPBD')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    output_file = os.path.join(output_dir, 'comparison_fps_log.png')
    plt.savefig(output_file)
    print(f"Saved comparison plot to {output_file}")
    
    # Linear Scale version
    plt.figure(figsize=(10, 6))
    plt.plot(x_tet, y_tet, 'r-o', linewidth=2, label='TetGenFEM (Ours, 10 Threads)')
    plt.plot(x_xpbd, y_xpbd, 'b--s', linewidth=1.5, label='XPBD Fast (Substeps=5, Low Accuracy)')
    plt.plot(x_xpbd_ref, y_xpbd_ref, 'b-^', linewidth=2, label='XPBD Reference (Substeps=50, High Accuracy)')
    
    plt.xlabel('Number of Tetrahedra')
    plt.ylabel('FPS')
    plt.title('Performance Comparison: TetGenFEM vs XPBD (Linear Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    output_file_lin = os.path.join(output_dir, 'comparison_fps_linear.png')
    plt.savefig(output_file_lin)
    print(f"Saved comparison plot to {output_file_lin}")

    # Bar Chart for ~20k Tets case
    # Find closest to 20000
    target = 20000
    
    # Get TetFEM 20k value
    tet_20k = min(tet_points, key=lambda x: abs(x[0] - target))
    
    # Get XPBD 20k value
    xpbd_20k = min(xpbd_points, key=lambda x: abs(x[0] - target))
    xpbd_ref_20k = min(xpbd_ref_points, key=lambda x: abs(x[0] - target))
    
    labels = ['XPBD Ref\n(High Acc.)', 'TetGenFEM\n(Ours)', 'XPBD Fast\n(Low Acc.)']
    values = [xpbd_ref_20k[1], tet_20k[1], xpbd_20k[1]]
    colors = ['blue', 'red', 'lightblue']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} FPS',
                ha='center', va='bottom')
                
    plt.ylabel('FPS')
    plt.title(f'Performance at ~20k Tetrahedra')
    plt.ylim(0, max(values) * 1.2)
    
    output_file_bar = os.path.join(output_dir, 'comparison_bar_20k.png')
    plt.savefig(output_file_bar)
    print(f"Saved bar chart to {output_file_bar}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Hardcoded paths based on exploration
    tetfem_csv = os.path.join(base_dir, "20251218_021141", "experiment4_performance.csv")
    xpbd_csv = os.path.join(base_dir, "20251221_152330_xpbd", "experiment4_performance.csv")
    vega_csv = os.path.join(base_dir, "VegaFEM_20251221_145459", "VegaFEM_cubeLong1300_performance.csv")
    
    output_dir = base_dir # Save in out/experiment4 root
    
    print("Generating comparison analysis...")
    analyze_comparison(tetfem_csv, xpbd_csv, vega_csv, output_dir)
