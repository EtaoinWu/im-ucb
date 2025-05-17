import math

def weight_convert(w):
  return w**0.618

def process_file(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # header line: num_nodes num_edges
    header = lines[0].strip()
    num_nodes, num_edges = map(int, header.split())

    edges = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        u, v = int(parts[0]), int(parts[1])
        w = float(parts[2])
        new_w = weight_convert(w)
        edges.append((u, v, new_w))

    with open(output_path, 'w') as f:
        f.write(f"{num_nodes} {num_edges}\n")
        for u, v, new_w in edges:
            f.write(f"{u} {v} {new_w:.10f}\n")

process_file("congress_orig.txt", "congress.txt")
