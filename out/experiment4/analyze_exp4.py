import csv
import matplotlib.pyplot as plt
import os
import sys

def read_csv_data(csv_path):
    data = []
    headers = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                if not row: continue
                # Convert numeric values
                converted_row = {}
                # Handle potential mismatch in header length vs row length
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
        print(f"Error reading CSV: {e}")
        return None
    return data

def analyze_and_plot(csv_path, output_dir):
    data = read_csv_data(csv_path)
    if not data:
        print("No data found.")
        return

    # Extract unique thread counts and sort them
    try:
        thread_counts = sorted(list(set(d['threads'] for d in data)))
    except KeyError:
        print("Error: 'threads' column not found in data.")
        print("Available keys:", data[0].keys() if data else "None")
        return
    
    # 1. Performance vs Mesh Size (FPS)
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    
    for i, t in enumerate(thread_counts):
        # Filter and sort by tet_count
        subset = [d for d in data if d['threads'] == t]
        subset.sort(key=lambda x: x['tet_count'])
        
        x = [d['tet_count'] for d in subset]
        y = [d['fps'] for d in subset]
        
        plt.plot(x, y, marker=markers[i % len(markers)], label=f'{int(t)} Threads')

    plt.xlabel('Number of Tetrahedra')
    plt.ylabel('FPS (Frames Per Second)')
    plt.title('Simulation Performance: FPS vs Mesh Size')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    output_path1 = os.path.join(output_dir, 'fps_vs_mesh_size.png')
    plt.savefig(output_path1)
    plt.close()

    # 2. Computational Time vs Mesh Size
    plt.figure(figsize=(10, 6))
    for i, t in enumerate(thread_counts):
        subset = [d for d in data if d['threads'] == t]
        subset.sort(key=lambda x: x['tet_count'])
        
        x = [d['tet_count'] for d in subset]
        y = [d['ms_mean'] for d in subset] # using ms_mean
        
        plt.plot(x, y, marker=markers[i % len(markers)], label=f'{int(t)} Threads')

    plt.xlabel('Number of Tetrahedra')
    plt.ylabel('Frame Time (ms)')
    plt.title('Computation Time vs Mesh Size')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    output_path2 = os.path.join(output_dir, 'time_vs_mesh_size.png')
    plt.savefig(output_path2)
    plt.close()

    # 3. Speedup Analysis
    multi_thread_data = [d for d in data if d['threads'] > 1]
    
    if multi_thread_data:
        plt.figure(figsize=(10, 6))
        multi_thread_counts = sorted(list(set(d['threads'] for d in multi_thread_data)))
        
        for i, t in enumerate(multi_thread_counts):
            subset = [d for d in multi_thread_data if d['threads'] == t]
            subset.sort(key=lambda x: x['tet_count'])
            
            x = [d['tet_count'] for d in subset]
            y = [d['speedup'] for d in subset]
            
            plt.plot(x, y, marker=markers[i % len(markers)], label=f'{int(t)} Threads Speedup')
        
        plt.xlabel('Number of Tetrahedra')
        plt.ylabel('Speedup (vs 1 Thread)')
        plt.title('Parallel Speedup vs Mesh Size')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.ylim(bottom=0)
        output_path3 = os.path.join(output_dir, 'speedup_vs_mesh_size.png')
        plt.savefig(output_path3)
        plt.close()

    print(f"Analysis plots generated in {output_dir}")

if __name__ == "__main__":
    # Path to the specific valid experiment data
    # Default to relative path if running from project root context
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "20251218_021141", "experiment4_performance.csv")
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    output_directory = os.path.dirname(csv_file)
    
    if os.path.exists(csv_file):
        print(f"Analyzing {csv_file}...")
        analyze_and_plot(csv_file, output_directory)
    else:
        print(f"File not found: {csv_file}")

