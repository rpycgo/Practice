# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:57:57 2021

@author: MJH
"""



visited = []
graph = []
cnt = 0


def bfs(edge: int) -> None:
    global visited, graph, cnt
    
    queue = [edge]
    while queue:
        for next_ in graph[queue.pop()]:
            if not visited[next_]:
                visited[next_]=True
                queue.append(next_)
                cnt += 1
  
def main():
    global visited, graph, cnt
    node = int(input())
    edge = int(input())
    
    visited = [False for _ in range(node+1)]
    graph = [[] for _ in range(node+1)]
    
    
    for _ in range(edge):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    bfs(1)
    print(cnt-1)


if __name__ == '__main__':
    main()        