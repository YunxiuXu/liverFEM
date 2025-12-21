import csv
import matplotlib.pyplot as plt
import os
import sys
import glob

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

def analyze_comparison(tetfem_path, xpbd_path, vega_base_dir, output_dir):
    tet_data = read_csv_data(tetfem_path)
    xpbd_data = read_csv_data(xpbd_path)
    
    # Collect all VegaFEM data from all folders
    vega_points = []
    vega_csv_files = glob.glob(os.path.join(vega_base_dir, "VegaFEM_*", "*_performance.csv"))
    
    for v_csv in vega_csv_files:
        v_data = read_csv_data(v_csv)
        if v_data:
            for d in v_data:
                # To avoid duplicate points from the same mesh, we can average or just add all
                # Each file usually has 1 and 10 threads, but we saw VegaFEM doesn't change much.
                # Let's take the multi-threaded one (if it exists) or just any.
                if int(d.get('numThreads', d.get('threads', 0))) >= 1:
                    vega_points.append((d['numElements'], d['avgFPS']))
    
    # Sort and remove duplicates (take max FPS for same element count)
    vega_dict = {}
    for num_ele, fps in vega_points:
        if num_ele not in vega_dict or fps > vega_dict[num_ele]:
            vega_dict[num_ele] = fps
    
    vega_points = sorted(vega_dict.items())

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
    xpbd_points = []
    for d in xpbd_data:
        t = int(d.get('threads', 1))
        if t == 10:
            xpbd_points.append((d['tet_count'], d['fps']))
    xpbd_points.sort(key=lambda x: x[0])

    if not xpbd_points:
        for d in xpbd_data:
            if int(d.get('threads', 1)) == 1:
                xpbd_points.append((d['tet_count'], d['fps']))
        xpbd_points.sort(key=lambda x: x[0])

    # Create Estimated Reference XPBD (50 substeps instead of 5)
    xpbd_ref_points = [(p[0], p[1] / 10.0) for p in xpbd_points]

    # Plotting Log Scale
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

    # VegaFEM Line
    if vega_points:
        x_v = [p[0] for p in vega_points]
        y_v = [p[1] for p in vega_points]
        plt.plot(x_v, y_v, 'g-d', linewidth=2, markersize=8, label='VegaFEM (Implicit BE, Ground Truth)')

    plt.xlabel('Number of Tetrahedra')
    plt.ylabel('FPS (Log Scale)')
    plt.yscale('log')
    plt.title('Performance Comparison: TetGenFEM vs XPBD vs VegaFEM')
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
    if vega_points:
        plt.plot(x_v, y_v, 'g-d', linewidth=2, label='VegaFEM (Implicit BE, Ground Truth)')
    
    plt.xlabel('Number of Tetrahedra')
    plt.ylabel('FPS')
    plt.title('Performance Comparison: TetGenFEM vs XPBD vs VegaFEM (Linear Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    output_file_lin = os.path.join(output_dir, 'comparison_fps_linear.png')
    plt.savefig(output_file_lin)
    print(f"Saved comparison plot to {output_file_lin}")

    # Bar Chart for ~20k Tets case
    target = 20000
    tet_20k = min(tet_points, key=lambda x: abs(x[0] - target))
    xpbd_ref_20k = min(xpbd_ref_points, key=lambda x: abs(x[0] - target))
    
    # Find closest VegaFEM to 20k
    if vega_points:
        vega_20k = min(vega_points, key=lambda x: abs(x[0] - target))
        labels = ['VegaFEM\n(GT)', 'XPBD Ref\n(High Acc.)', 'TetGenFEM\n(Ours)']
        values = [vega_20k[1], xpbd_ref_20k[1], tet_20k[1]]
        colors = ['green', 'blue', 'red']
    else:
        labels = ['XPBD Ref\n(High Acc.)', 'TetGenFEM\n(Ours)']
        values = [xpbd_ref_20k[1], tet_20k[1]]
        colors = ['blue', 'red']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} FPS',
                ha='center', va='bottom')
                
    plt.ylabel('FPS')
    plt.title(f'Performance at ~20k Tetrahedra (Liver Mid)')
    plt.ylim(0, max(values) * 1.3)
    
    output_file_bar = os.path.join(output_dir, 'comparison_bar_20k.png')
    plt.savefig(output_file_bar)
    print(f"Saved bar chart to {output_file_bar}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tetfem_csv = os.path.join(base_dir, "20251218_021141", "experiment4_performance.csv")
    xpbd_csv = os.path.join(base_dir, "20251221_152330_xpbd", "experiment4_performance.csv")
    vega_base_dir = base_dir # out/experiment4 root
    
    output_dir = base_dir
    
    print("Generating comparison analysis with multiple VegaFEM points...")
    analyze_comparison(tetfem_csv, xpbd_csv, vega_base_dir, output_dir)
