# ----------------------------------------------------------------------
# Rubik's cube simulator
# Numpy is used for face representation and operation
# Matplotlib only for plotting
# Written by Jorge Scharfhausen (2017)
# The aim of this code is to give a simple rubik cube simulator to
# test Discrete Planning Techniques.


import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import colors
import copy

'''
Face State order as it is internally represented
    | 4 |
| 0 | 1 | 2 | 3 |
    | 5 |
Each face is represented by state matrix (NxN) and each cell is an integuer (0-5). 
Row and columns are disposed with the origin at the upper left corner, 
with faces disposed as the unfolded cube states. 

Rotations are referred to axis relative faces.
The outward-pointing normal of face 1 is the X axis.
The outward-pointing normal of face 2 is the Y axis.
The outward-pointing normal of face 4 is the Z axis.
 
Rotations are considered positive if they are ccw around the axis (math positive rotation)
The  cube slices are considered as layers. The upper layer (faces 1, 2 or 4) have index 0, while de 
backward layers (3,0,5) have index N-1 (N is the cube dimension)

Initial colors have the same index than their respective faces
'''

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

class RubCube:
    # face + rotation, face -, lateral faces (index, [tuple 1] [tuple2) tomando como base la gira +
    # giro X
    F_axis = {'front': 1, 'back': 3, 'faces': ((2, (0, 1), (-1, 0)),
                                               (4, (-1, 0), (0, -1)),
                                               (0, (0, -1), (1, 0)),
                                               (5, (1, 0), (0,
                                                            1)))}  # giro F realizado en la cara 1  capa i afecta a la i*[0,i], (0...N)*[-i 0]
    # giro Y
    R_axis = {'front': 2, 'back': 0, 'faces': ((3, (0, 1), (-1, 0)),
                                               (4, (0, -1), (1, 0)),
                                               (1, (0, -1), (1, 0)),
                                               (5, (0, -1), (1,
                                                             0)))}  # giro R realizado en la cara 2  capa i afecta a la i*[0,i], (0...N)*[-i 0]
    # giro Z
    U_axis = {'front': 4, 'back': 5, 'faces': ((0, (1, 0), (0, 1)),
                                               (1, (1, 0), (0, 1)),
                                               (2, (1, 0), (0, 1)),
                                               (3, (1, 0), (0,
                                                            1)))}  # giro U realizado en la cara 4  capa i afecta a la i*[0,i], (0...N)*[-i 0]
    axis_dict = {'x': F_axis, 'y': R_axis, 'z': U_axis}

    def __init__(self, N=3):
        self._N = N
        self.reset()

    def rotate_90(self, axis_name='x', n=0, n_rot=1):
        '''rotates 90*n_rot around one axis ('x','y','z') the layer n'''
        if axis_name not in self.axis_dict:
            return
        axis = self.axis_dict[axis_name]
        if n == 0:  # rotate the front face
            self._state[axis['front']] = np.rot90(self._state[axis['front']], k=n_rot)
        if n == self._N - 1:
            self._state[axis['back']] = np.rot90(self._state[axis['back']], k=n_rot)
        aux = []
        for f in axis['faces']:
            if f[1][0] > 0:  # row +
                r = self._state[f[0]][n, ::f[2][1]]
            elif f[1][0] < 0:  # row -
                r = self._state[f[0]][-(n + 1), ::f[2][1]]
            elif f[1][1] > 0:  # column +
                r = self._state[f[0]][::f[2][0], n]
            else:
                r = self._state[f[0]][::f[2][0], -(n + 1)]
            aux.append(r)
        raux = np.roll(np.array(aux), (self._N) * n_rot)
        
        for i,f in enumerate(axis['faces']):
            r = raux[i]
            if f[1][0] > 0:  # row +
                self._state[f[0]][n, ::f[2][1]] = r
            elif f[1][0] < 0:  # row -
                self._state[f[0]][-(n + 1), ::f[2][1]] = r
            elif f[1][1] > 0:  # column +
                self._state[f[0]][::f[2][0], n] = r
            else:
                self._state[f[0]][::f[2][0], -(n + 1)] = r

    def set_State(self, state):
        self._state = np.array(state)

    def get_State(self):
        return totuple(self._state)

    def plot(self, block=True):
        plot_list = ((1, 4), (4, 0), (5, 1), (6, 2), (7, 3), (9, 5))
        color_map = colors.ListedColormap(['#00008f', '#cf0000', '#009f0f', '#ff6f00', 'w', '#ffcf00'], 6)
        fig = plt.figure(1, (8., 8.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(3, 4),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )
        for p in plot_list:
            grid[p[0]].matshow(self._state[p[1]], vmin=0, vmax=5, cmap=color_map)
        plt.show(block=block)

    def reset(self):
        self._state = []
        for i in range(6):
            self._state.append(i * np.ones((self._N, self._N), dtype=np.int8))
    def randomMoves(self, num):
        moves=[]
        for i in range(num):
            x = random.choice(('x','y','z'))
            num = random.choice([0, self._N - 1])
            n_rot = random.randint(-1,2)
            self.rotate_90(x,num,n_rot)
            moves.append((x,num,n_rot))
        print(moves)
        return moves


class Node():
    def __init__(self, stateNode = None, parent=None, move = 'x00'):
        self.parent = parent
        self.state = stateNode
        self.move = move

        self.G = 0
        self.H = 0
        self.F = 0

    def getMove(self):
        if self.move[2] == '2':
            return 2
        else: return 1


    def getH(self):                 #TODO: Heuristic may not be the best. Think about something better
        for i in range(6):
            for x in range(3):
                for y in range(3):
                    if self.state._state[i][x][y] != i:
                        self.H+=1
        return self.H




def AStarAlgorithm(start):

    iterNumber=0
    # Start and end node, with the state given by randomMove
    startNode = Node(start, None)
    startNode.G = startNode.H = startNode.F = 0

    endNode = Node(RubCube(3),None)

    # Initialize open and closed list
    openList = []
    closedList = []

    # Add the start node to the open list to start iterating
    openList.append(startNode)

    # While there's a node in the list, keep going
    while len(openList) > 0:

        # Get the current node
        currentNode = openList[0]
        currentIndex = 0
        for index, item in enumerate(openList):
            if item.F < currentNode.F:
                currentNode = item
                currentIndex = index

        # Take out the node from open list, add to closed list
        openList.pop(currentIndex)
        closedList.append(currentNode)

        # Found the goal //TODO: CHANGE HOW IT CHECKS IF IT'S THE END
        if currentNode.state.get_State() == endNode.state.get_State():
            path = []
            current = currentNode

            while current is not None:
                path.append(current.move)
                current = current.parent

            print(path[-2::-1]) #Return inverse path, taking out the first default move 'x00', which doesn't mean anyting
            print("Cube solved!")
            return currentNode.state

        # Generate children
        children = []
        auxState = copy.deepcopy(currentNode.state)  #If assigned with '=', it references the
                                                     #instance of that class, so every change in currentNode affects auxState

        for letter in 'xyz':
                for slice in '02':      #To maintain convention that the center of each face stays in its place
                        for rots in '123':      #3 rotations = -1 rotation: Cost of rotation is 2 for rots=2, 1 otherwise
                            auxState.rotate_90(letter,int(slice),int(rots))
                            children.append(Node(auxState, currentNode, letter+slice+rots))
                            auxState = copy.deepcopy(currentNode.state)

        # For every child in children
        for child in children:

           
            for closedChild in closedList:
                if child == closedChild:
                    continue

            # Create the f, g, and h values
            child.G = currentNode.G + currentNode.getMove()

            #if child.G>20:      #God's number
             #   children.pop(child)
              #  continue

            child.H = child.getH()
            child.F = child.G + child.H

            # Child is already in the open list
            for openNode in openList:
                if child == openNode and child.G > openNode.G:
                    continue

            # Add the child to the open list
            openList.append(child)

        iterNumber=iterNumber+1
        print("End of iteration {}. The current node F is {}".format(iterNumber, currentNode.F))


if __name__ == '__main__':
    import sys

    try:
        N = int(sys.argv[1])
    except:
        N = 3

    a = RubCube(N)      
    m=a.randomMoves(4)
    a.plot()
    sol=AStarAlgorithm(a)
  






