import numpy as np
i = [[ 1,2,3],
     [4,5,6],
     [7,8,9]]

j = [[9,8,7],
     [6,5,4],
     [3,2,1]]

p1 = [[0,0,0],
      [0,0,0],
      [0,0,0]]

for x in range(len(i[0])):
    for y in range(len(j[0])):
        p1[x][y] = i[x][y] + j[x][y]
for r in p1:
    print("the two matrices added = " + r)
    
