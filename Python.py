import heapq
import random
import sys

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = {i: [] for i in range(vertices)}

    def add_edge(self, u, v, w):
        self.graph[u].append((v, w))
        self.graph[v].append((u, w))

    def dijkstra(self, src):
        dist = [float('inf')] * self.V
        dist[src] = 0
        visited = [False] * self.V
        heap = [(0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            for v, w in self.graph[u]:
                if not visited[v] and dist[v] > d + w:
                    dist[v] = d + w
                    heapq.heappush(heap, (dist[v], v))
        return dist

def generate_random_graph(v, e):
    g = Graph(v)
    edges = set()
    while len(edges) < e:
        u = random.randint(0, v - 1)
        v2 = random.randint(0, v - 1)
        if u != v2:
            w = random.randint(1, 100)
            if (u, v2) not in edges and (v2, u) not in edges:
                edges.add((u, v2))
                g.add_edge(u, v2, w)
    return g

def floyd_warshall(graph):
    V = graph.V
    dist = [[float('inf')] * V for _ in range(V)]
    for i in range(V):
        dist[i][i] = 0
    for u in graph.graph:
        for v, w in graph.graph[u]:
            dist[u][v] = min(dist[u][v], w)
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

def prim_mst(graph):
    V = graph.V
    visited = [False] * V
    min_heap = [(0, 0)]
    total_weight = 0
    while min_heap:
        weight, u = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        total_weight += weight
        for v, w in graph.graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (w, v))
    return total_weight

def kruskal_mst(graph):
    parent = list(range(graph.V))
    rank = [0] * graph.V
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            elif rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_v] = root_u
                rank[root_u] += 1
            return True
        return False
    edges = []
    for u in graph.graph:
        for v, w in graph.graph[u]:
            if u < v:
                edges.append((w, u, v))
    edges.sort()
    total_weight = 0
    for w, u, v in edges:
        if union(u, v):
            total_weight += w
    return total_weight

def bfs(graph, start):
    visited = [False] * graph.V
    queue = [start]
    visited[start] = True
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v, _ in graph.graph[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    return order

def dfs_util(graph, u, visited, order):
    visited[u] = True
    order.append(u)
    for v, _ in graph.graph[u]:
        if not visited[v]:
            dfs_util(graph, v, visited, order)

def dfs(graph, start):
    visited = [False] * graph.V
    order = []
    dfs_util(graph, start, visited, order)
    return order

def topological_sort_util(graph, v, visited, stack):
    visited[v] = True
    for u, _ in graph.graph[v]:
        if not visited[u]:
            topological_sort_util(graph, u, visited, stack)
    stack.insert(0, v)

def topological_sort(graph):
    visited = [False] * graph.V
    stack = []
    for i in range(graph.V):
        if not visited[i]:
            topological_sort_util(graph, i, visited, stack)
    return stack

def bellman_ford(graph, src):
    dist = [float('inf')] * graph.V
    dist[src] = 0
    edges = []
    for u in graph.graph:
        for v, w in graph.graph[u]:
            edges.append((u, v, w))
    for _ in range(graph.V - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[v] > dist[u] + w:
            return None
    return dist

def main():
    v = 10
    e = 20
    g = generate_random_graph(v, e)
    print("Dijkstra:", g.dijkstra(0))
    print("Floyd-Warshall:", floyd_warshall(g))
    print("Prim MST:", prim_mst(g))
    print("Kruskal MST:", kruskal_mst(g))
    print("BFS:", bfs(g, 0))
    print("DFS:", dfs(g, 0))
    print("Topological Sort:", topological_sort(g))
    print("Bellman-Ford:", bellman_ford(g, 0))

if __name__ == "__main__":
    main()