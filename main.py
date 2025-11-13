import csv
import heapq
import matplotlib.pyplot as plt
import networkx as nx
from math import radians, cos, sin, sqrt, atan2
import time

# ========================================
# 1. Haversine Distance (in km)
# ========================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def dist_sq(stations, u, v):
    lat1, lon1 = stations[u]['lat'], stations[u]['lon']
    lat2, lon2 = stations[v]['lat'], stations[v]['lon']
    return (lat1 - lat2)**2 + (lon1 - lon2)**2

# ========================================
# 2. Load Stations
# ========================================
def load_stations(filename="DelhiMetro.csv"):
    stations = []
    station_map = {}
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Station Names'].strip()
            try:
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
            except:
                continue
            idx = len(stations)
            stations.append({'id': idx, 'name': name, 'lat': lat, 'lon': lon})
            station_map[name.lower()] = idx
    return stations, station_map

# ========================================
# 3. Prim’s Algorithm
# ========================================
def prim_mst(stations, start_idx=0):
    V = len(stations)
    in_mst = [False] * V
    min_edge = [float('inf')] * V
    parent = [-1] * V
    pq = []
    min_edge[start_idx] = 0
    heapq.heappush(pq, (0, start_idx))

    total_cost = 0.0
    edges = []

    while pq:
        _, u = heapq.heappop(pq)
        if in_mst[u]: continue
        in_mst[u] = True

        if parent[u] != -1:
            d = haversine(
                stations[parent[u]]['lat'], stations[parent[u]]['lon'],
                stations[u]['lat'], stations[u]['lon']
            )
            total_cost += d
            edges.append((parent[u], u, d))

        for v in range(V):
            if not in_mst[v]:
                w = dist_sq(stations, u, v)
                if w < min_edge[v]:
                    min_edge[v] = w
                    parent[v] = u
                    heapq.heappush(pq, (w, v))

    return edges, total_cost

# ========================================
# 4. Kruskal’s Algorithm
# ========================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        return True

def kruskal_mst(stations):
    V = len(stations)
    edges = []
    for i in range(V):
        for j in range(i + 1, V):
            d = haversine(stations[i]['lat'], stations[i]['lon'],
                          stations[j]['lat'], stations[j]['lon'])
            edges.append((d, i, j))
    edges.sort()
    uf = UnionFind(V)
    mst_edges = []
    total = 0.0
    for d, u, v in edges:
        if uf.union(u, v):
            mst_edges.append((u, v, d))
            total += d
            if len(mst_edges) == V - 1:
                break
    return mst_edges, total

# ========================================
# 5. Graph Building
# ========================================
def build_graph(stations, edges):
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(stations[u]['name'], stations[v]['name'], weight=d)
    return G

# ========================================
# 6. Shortest Path in MST
# ========================================
def find_path(G, start, end):
    try:
        path = nx.shortest_path(G, start, end, weight='weight')
        length = nx.shortest_path_length(G, start, end, weight='weight')
        return path, length
    except:
        return None, 0

# ========================================
# 7. Plot Path with Edge Costs
# ========================================
def plot_path_only(stations, path, title="", start=None, end=None):
    plt.figure(figsize=(8, 6))
    G = nx.Graph()
    pos = {}

    for name in path:
        idx = next(i for i, s in enumerate(stations) if s['name'] == name)
        G.add_node(name)
        pos[name] = (stations[idx]['lon'], stations[idx]['lat'])

    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='white', edgecolors='black', linewidths=1.5)
    if start:
        nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='lightblue', node_size=1000)
    if end:
        nx.draw_networkx_nodes(G, pos, nodelist=[end], node_color='orange', node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    # Edge cost labels
    edge_labels = {}
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        u_idx = next(s['id'] for s in stations if s['name'] == u)
        v_idx = next(s['id'] for s in stations if s['name'] == v)
        d = haversine(stations[u_idx]['lat'], stations[u_idx]['lon'],
                      stations[v_idx]['lat'], stations[v_idx]['lon'])
        edge_labels[(u, v)] = f"{d:.2f} "

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='green')

    plt.title(title, fontsize=12)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ========================================
# 8. MAIN
# ========================================
def main():
    stations, station_map = load_stations("DelhiMetro.csv")

    print("\n" + "="*60)
    print("DELHI METRO – Enter Start & End Station")
    print("="*60)
    start_input = input("Enter START station: ").strip()
    end_input = input("Enter END station: ").strip()

    start_lower, end_lower = start_input.lower(), end_input.lower()
    if start_lower not in station_map or end_lower not in station_map:
        print("❌ Invalid station name. Please check your input.")
        return

    sidx = station_map[start_lower]
    start_name = stations[sidx]['name']
    end_name = stations[station_map[end_lower]]['name']

    # --- Prim’s Algorithm ---
    t1 = time.time()
    prim_edges, _ = prim_mst(stations, sidx)
    prim_time = time.time() - t1
    Gp = build_graph(stations, prim_edges)
    path_p, len_p = find_path(Gp, start_name, end_name)

    print("\n" + "-"*70)
    print("PRIM’S MST RESULTS")
    print(f"Execution Time: {prim_time:.4f} s")
    print(f"Total Path Cost: {len_p:.2f} km")
    if path_p:
        print("Traversal Path:")
        print("  " + " -> ".join(path_p))
        print("\nEdge Distances:")
        for i in range(len(path_p) - 1):
            u, v = path_p[i], path_p[i + 1]
            u_idx = station_map[u.lower()]
            v_idx = station_map[v.lower()]
            d = haversine(stations[u_idx]['lat'], stations[u_idx]['lon'],
                          stations[v_idx]['lat'], stations[v_idx]['lon'])
            print(f"  {u} → {v}: {d:.2f} km")
        plot_path_only(stations, path_p, f"Prim’s Path ({len_p:.2f} km)", start_name, end_name)
    else:
        print("No path found using Prim’s Algorithm")

    # --- Kruskal’s Algorithm ---
    t2 = time.time()
    kruskal_edges, _ = kruskal_mst(stations)
    kruskal_time = time.time() - t2
    Gk = build_graph(stations, kruskal_edges)
    path_k, len_k = find_path(Gk, start_name, end_name)

    print("\n" + "-"*70)
    print("KRUSKAL’S MST RESULTS")
    print(f"Execution Time: {kruskal_time:.4f} s")
    print(f"Total Path Cost: {len_k:.2f} km")
    if path_k:
        print("Traversal Path:")
        print("  " + " -> ".join(path_k))
        print("\nEdge Distances:")
        for i in range(len(path_k) - 1):
            u, v = path_k[i], path_k[i + 1]
            u_idx = station_map[u.lower()]
            v_idx = station_map[v.lower()]
            d = haversine(stations[u_idx]['lat'], stations[u_idx]['lon'],
                          stations[v_idx]['lat'], stations[v_idx]['lon'])
            print(f"  {u} → {v}: {d:.2f} km")
        plot_path_only(stations, path_k, f"Kruskal’s Path ({len_k:.2f} km)", start_name, end_name)
    else:
        print("No path found using Kruskal’s Algorithm")

    # --- Summary ---
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*70)
    print(f"Prim’s Total Cost:    {len_p:.2f} km | Time: {prim_time:.4f} s")
    print(f"Kruskal’s Total Cost: {len_k:.2f} km | Time: {kruskal_time:.4f} s")
    if prim_time < kruskal_time:
        print("\n✅ Prim’s Algorithm executed faster.")
    elif kruskal_time < prim_time:
        print("\n✅ Kruskal’s Algorithm executed faster.")
    else:
        print("\n⚖️ Both took nearly the same time.")

# ========================================
# Run
# ========================================
if __name__ == "__main__":
    main()
