import numpy as np
i = [[ 1,2,3],
     [4,5,6],
     [7,8,9]]

j = [[5,9,9],
     [4,7,2],
     [1,8,1]]

p1 = [[0,0,0],
      [0,0,0],
      [0,0,0]]

maxint = 0
maxrow = 0

for x in range(len(i[0])):
    for y in range(len(j[0])):
        p1[x][y] = i[x][y] + j[x][y]

print("the two matrices added = ")
for r in p1:
    print(r)

x = 1
y = 0

for x in range(len(p1[0])):
    #//if p1[x][y] > maxrow:
    # maxrow = p1[x][y]
    #print("Max number in row", x ," is ", maxrow)
    for y in range(len(p1[0])):
        if (p1[x][y] > maxint):
            maxint = p1[x][y]
print("Max number in matix is ", maxint)
