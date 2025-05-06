import heapq
import math
import random
import matplotlib.pyplot as plt
import itertools, scipy.spatial
from matplotlib.collections import LineCollection
from collections import defaultdict, namedtuple

# ----------------------------------
# Simple data structures
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
    dist = {v: math.inf for v in g.xy}
    nearest = {v: None for v in g.xy}

    # generate one distinct color per post office
    # using a nice pastel color palette
    color_palette = {}
    rng = random.Random(42)         # always use 42 :)
    for p in post_offices:
        color_palette[p] = (rng.random()*0.5 + 0.4,   # avoid dark colors
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
    # build a graph: 100 random points in the unit square,
    # connect each vertex to its 6 nearest neighbors
    N = 100
    k = 6
    pts = [(random.random(), random.random()) for _ in range(N)]

    g = Graph()
    for vid, (x, y) in enumerate(pts):
        g.add_vertex(vid, x, y)

    # k‑nearest‑neighbor edges using a KD‑tree
    # this is much faster than brute force (O(N^2) vs O(N log N))
    tree = scipy.spatial.KDTree(pts)
    for vid, (x, y) in enumerate(pts):
        _, idxs = tree.query((x, y), k + 1)   # k+1 because the first is itself
        for v in idxs[1:]:
            g.add_edge(vid, int(v))

    # choose 5% of vertices as post offices
    offices = random.sample(range(N), max(1, N * 5 // 100))

    dist, nearest, color = dijkstra(g, offices)

    # -----------------------------------
    # visualization
    xs, ys = zip(*pts)
    cs = [color[i] for i in range(N)]

    fig, ax = plt.subplots(figsize=(8, 8))

    # draw edges
    lines = []
    for u in range(N):
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