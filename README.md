# Metro-Network-Optimization-Using-Minimum-Spanning-Tree-MST-Algorithms

A **Minimum Spanning Tree (MST)**-based approach to optimize the **Metro network** using real-world station coordinates. This project compares **Prim’s** and **Kruskal’s** algorithms to connect 30 metro stations with **minimum total distance**, simulating cost-efficient infrastructure planning.

---

## Features

- Real geospatial data of **30 Delhi Metro stations** (Latitude, Longitude)
- Accurate distance calculation using the **Haversine formula**
- Implementation of **Prim’s** and **Kruskal’s** MST algorithms
- Interactive pathfinding between any two stations
- Visualizations of MST and shortest paths using **Matplotlib** and **NetworkX**
- Performance comparison (execution time & path cost)

---

## Dataset

`DelhiMetro.csv` contains:
- Station ID, Name, Line, Distance from first station, Opening year, Layout, Coordinates

---

View:
MST construction
Shortest path in MST
Edge distances
Interactive plots



Sample Output:

PRIM’S MST RESULTS
Execution Time: 0.0021 s
Total Path Cost: 45.32 km
Traversal Path:
  Kashmere Gate -> Rajiv Chowk -> New Delhi -> Dwarka

Edge Distances:
  Kashmere Gate → Rajiv Chowk: 10.45 km
  Rajiv Chowk → New Delhi: 1.20 km
  New Delhi → Dwarka: 33.67 km

KRUSKAL’S MST RESULTS
Execution Time: 0.0018 s
Total Path Cost: 48.91 km
...

Key Findings:

Both algorithms produce a valid MST

Prim’s often gives shorter user paths (better for passengers)

Kruskal’s is slightly faster on small datasets

Total MST cost: ~150–160 km (minimum infrastructure)


Future Scope:

Hybrid Prim-Kruskal algorithm

Multi-objective optimization (cost, demand, land price)

Real-time route suggestions with live data


References:

Kim, H.-C. et al. (2017). Schematic Transit Network Design using Minimum Spanning Tree Algorithm. Journal of the Eastern Asia Society for Transportation Studies.
https://doi.org/10.11175/easts.12.1299

Zeng, X. et al. (2019). An Improved Prim Algorithm for Connection Scheme of Last Train in Urban Mass Transit Network. Symmetry.
https://doi.org/10.3390/sym11050681
