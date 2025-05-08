import heapq
import math
import random
import time
import scipy.spatial
from collections import defaultdict, namedtuple

# ----------------------------------
# Data structures
# ----------------------------------

# edge is a named tuple with two fields: to (destination vertex ID) and weight (edge weight)
Edge = namedtuple("Edge", ["to", "weight"])

# undirected graph with vertex coordinates and adjacency list
class Graph:
    def __init__(self):
        self.xy = {}                    # id -> (x, y)
        self.adj = defaultdict(list)    # id -> list[Edge]

    # add vertex with given ID and coordinates
    def add_vertex(self, vid, x, y):
        self.xy[vid] = (x, y)

    # get euclidean distance between two vertices
    def _euclidean(self, u, v):
        x1, y1 = self.xy[u]
        x2, y2 = self.xy[v]
        return math.hypot(x1 - x2, y1 - y2)

    # add undirected edge between two vertices
    def add_edge(self, u, v):
        w = self._euclidean(u, v)
        self.adj[u].append(Edge(v, w))
        self.adj[v].append(Edge(u, w))

# ----------------------------------
# Dijkstra
# ----------------------------------
def dijkstra(g: Graph, post_offices):
    # set up distance and nearest office dictionaries
    dist = {v: math.inf for v in g.xy}
    nearest = {v: None for v in g.xy}

    # generate one distinct color per post office
    # using a nice pastel color palette
    color_palette = {}
    rng = random.Random(15)         # my lucky number
    for p in post_offices:
        color_palette[p] = (rng.random()*0.5 + 0.4,   # no dark colors
                            rng.random()*0.5 + 0.4,
                            rng.random()*0.5 + 0.4)

    # initialize distances to post offices and set up priority queue
    pq = []                         # heap elements: (distance, vertex_id)
    for p in post_offices:
        dist[p] = 0.0
        nearest[p] = p
        heapq.heappush(pq, (0.0, p))

    # dijkstra's algorithm
    visited = set()
    while pq:
        d_u, u = heapq.heappop(pq)  # get vertex with smallest distance
        if u in visited:        
            continue
        visited.add(u)

        for edge in g.adj[u]:       # for each neighbor
            v = edge.to
            if v in visited:        # already settled with final distance
                continue
            alt = d_u + edge.weight # alternative distance
            if alt < dist[v]:       # found a shorter path
                dist[v] = alt
                nearest[v] = nearest[u]      # inherit office label
                heapq.heappush(pq, (alt, v))

    # assign colors
    color = {v: color_palette[nearest[v]] for v in g.xy}

    return dist, nearest, color

# ----------------------------------
# Tests
# ----------------------------------
if __name__ == "__main__":
    # build a graph: N random points in the unit square,
    # connect each vertex to its k nearest neighbors
    # p% of the vertices are post offices
    n = 1000
    k = 10
    p = 5
    time_alg = True      # set to True for testing
    vis = True           # set to True for visualization

    random.seed(15)
    pts = [(random.random(), random.random()) for _ in range(n)]

    g = Graph()
    for vid, (x, y) in enumerate(pts):
        g.add_vertex(vid, x, y)

    # k‑nearest‑neighbor edges using a KD‑tree
    # using the scipy.spatial KDTree implementation
    tree = scipy.spatial.KDTree(pts)
    for vid, (x, y) in enumerate(pts):
        _, idxs = tree.query((x, y), k + 1)   # k+1 because 1st closest is itself
        for v in idxs[1:]:
            g.add_edge(vid, int(v))

    # choose p% of vertices as post offices
    offices = random.sample(range(n), max(1, n * p // 100))

    # time dijkstra's algorithm 5 times and average the time
    if time_alg:
        times = []
        for _ in range(5):
            start_time = time.time()
            dijkstra(g, offices)
            end_time = time.time()
            times.append(end_time - start_time)
        avg_time = sum(times) / len(times)
        print(f"n: {n}, k: {k}, p: {p}, avg_time: {avg_time:.4f}s")
    
    # run dijkstra's algorithm once more to get the distances and colors
    # (this is the final version, so we don't need to time it)
    dist, nearest, color = dijkstra(g, offices)
    

    # -----------------------------------
    # visualization with matplotlib (comment out for final version)
    # -----------------------------------
    if vis:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        xs, ys = zip(*pts)
        cs = [color[i] for i in range(n)]

        fig, ax = plt.subplots(figsize=(8, 8))

        # draw edges
        lines = []
        for u in range(n):
            for edge in g.adj[u]:
                v = edge.to
                if u < v:       # avoid drawing both directions
                    lines.append([pts[u], pts[v]])
        lc = LineCollection(lines, colors='lightgray', linewidths=0.7, alpha=0.5)
        ax.add_collection(lc)

        # draw nodes
        sc = ax.scatter(xs, ys, c=cs, s=40, edgecolors='k', linewidths=0.7, zorder=2)

        # draw post offices
        office_x = [pts[i][0] for i in offices]
        office_y = [pts[i][1] for i in offices]
        ax.scatter(office_x, office_y, c='none', edgecolors='black', marker='^', s=180, linewidths=2, label='Post Office', zorder=3)

        ax.set_title("Post Office Map")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

        # save the figure
        fig.savefig("post_office_map.png", dpi=300, bbox_inches='tight')