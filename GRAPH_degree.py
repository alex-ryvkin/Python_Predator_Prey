import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create a graph object
GG = nx.Graph()

r=6
c=10
#
nn =np.arange(r*c)
Nodes = np.zeros(r*c)
# print(Nodes)
#
# # Add nodes
#
GG.add_nodes_from(nn)
#
# # Add edges
# G.add_edge(1, 2)
# G.add_edges_from([(1, 3), (2, 3)])
#
# # Plot the graph
# nx.draw(G, with_labels=True, node_color='skyblue', node_size=2000, font_size=12)
# plt.show()

def check(n):
    f = True
    if G[n][0] == 1:
        f = not f
    return f


# T = [[0 for j in range(6)] for i in range(10)]
p = 0.2 # obs/r*c
obs = round(r*c*p)
#err = [3,8,16,22,32,39,42,46,57]
err = np.random.randint(0, r*c-1, size=obs)
print(obs, err)
Nodes[err]=1

# count = 1
# for i in range(len(T)):
#     for j in range(len(T[i])):
#         if count in err:
#             T[i][j] = 1
#         count += 1
#
# G = {}
# rowlen = 10
# columnlen = 6

count = 1
Ed = []

for i in range(c*r):
    if(((i+1)%c)!=0):
        if (Nodes[i]!=1 and Nodes[i+1]!=1  ):
            Ed.append([i,i+1])
    if(i<c*(r-1)):
        if (Nodes[i]!=1 and Nodes[i+c]!=1  ):
            Ed.append([i,i+c])
GG.add_edges_from(Ed)
print(Ed)
print(GG)




# for r in range(rowlen):
#     for c in range(columnlen):
#         G[count] = [T[r][c], []]
#
#         count += 1
#
#
# num = 0
#
#
#
# for r in range(rowlen):
#     for c in range(columnlen):
#         num += 1
#         if not check(num):
#             continue
#         vert = {'right': num + 1, 'down': num + columnlen, 'left': num - 1, 'up': num - columnlen}
#         if r == 0:
#             del vert['up']
#         if r == rowlen - 1:
#             del vert['down']
#         if c == 0:
#             del vert['left']
#         if c == columnlen - 1:
#             del vert['right']
#
#         for k in vert:
#             v = vert[k]
#             if check(v):
#                 G[num][1].append(v)
#                 Ed.append()
shortest_path = nx.shortest_path(GG, source=0, target=59)  # Assuming edges have weights
print(shortest_path)
nx.draw(GG, with_labels=True, node_color='skyblue', node_size=100, font_size=12)
plt.show()