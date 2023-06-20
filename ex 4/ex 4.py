# Ex 4. Lost city
import string
import numpy as np
import math
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import heapq


def dijkstra(graph, start):
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


data_dict = {}
with open('map.txt') as f:
    for line in f:
        if 'Coord' in line or '---' in line:
            continue
        (key, val) = line.strip().split("            ")
        data_dict[key] = val
lst = list(data_dict.items())
coord = list(data_dict.keys())
value = list(data_dict.values())
for i in range(len(value)):
    if value[i] != "NaN":
        value[i] = float(value[i])
    else:
        value[i] = np.nan
coord2 = []
for i in coord:
    coord2.append(i.translate(str.maketrans('', '', string.punctuation)))
x = []
y = []
for i in range(0, 6):
    x.append(i)
for j in range(0, 3):
    y.append(j)
coord_final = []
for j in y:
    for i in x:
        coord_final.append((i, j))
Coords = np.array(coord_final)
Value = np.array(value)

Known_Values = Value[~np.isnan(Value)]
Known_Coords = Coords[~np.isnan(Value)]

rbf = Rbf(Known_Coords[:, 0], Known_Coords[:, 1], Known_Values)
x_meshgrid, y_meshgrid = np.meshgrid(np.arange(6), np.arange(3))
interpolated_elevations = rbf(x_meshgrid, y_meshgrid)

print('Interpolated Values by Radial Basis Function:')
iterat = 0
for j in y:
    for i in x:
        print(f"({i}, {j})   {float(interpolated_elevations[j][i])}")

ax = plt.axes(projection='3d')
surface = ax.plot_surface(x_meshgrid, y_meshgrid, interpolated_elevations, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Elevation, meters')
plt.colorbar(surface)
ax.set_title('Surface of the Map')
plt.show()
