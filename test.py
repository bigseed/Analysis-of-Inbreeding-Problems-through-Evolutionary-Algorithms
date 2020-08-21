import matplotlib.pyplot as plt
import numpy as np
import operator
import random


def mul(point, k):
    return tuple(map(lambda x: tuple(map(lambda y: k*y, x)), point))


def makeGrid(n):
    basicPoint = ((0, 1), (round(np.sqrt(3)/2, 3), 0.5), (round(np.sqrt(3)/2, 3), -0.5),
                  (0, -1), (-round(np.sqrt(3)/2, 3), -0.5), (-round(np.sqrt(3)/2, 3), 0.5))
    point = []
    for k in range(n):
        point.append(mul(basicPoint, k+1))

    return tuple(point)


def drawGrid(n):
    todraw = makeGrid(n)
    for points in todraw:
        for i in range(len(points)):
            plt.plot((points[i-1][0], points[i][0]),
                     (points[i-1][1], points[i][1]), c='#bdc3c7', linestyle='--')

    for i in range(3):
        plt.plot((todraw[-1][i][0], todraw[-1][i+3][0]),
                 (todraw[-1][i][1], todraw[-1][i+3][1]), c='#bdc3c7', linestyle='--')


def drawPoint(todraw):
    for i in range(len(todraw)):
        plt.fill((todraw[i-1][0], todraw[i][0], 0),
                 (todraw[i-1][1], todraw[i][1], 0), c='#e74c3c')
        plt.plot((todraw[i-1][0], todraw[i][0]),
                 (todraw[i-1][1], todraw[i][1]), c='#e74c3c', marker='o', linewidth=3)


def toPoint(chromosome):
    point = []
    step = np.pi / 3
    for i, genes in enumerate(chromosome):
        point.append(
            (round(genes*np.sin(step*i), 3), round(genes*np.cos(step*i), 3)))
    return tuple(point)


def surface(chromosome):
    s = 0
    for i in range(len(chromosome)):
        s += chromosome[i] * chromosome[i-1] * np.sin(np.pi/3) / 2
    return round(s, 3)


def offspring(p1, p2):
    baby = []
    for i in range(6):
        baby.append(round(random.uniform(p1[i], p2[i]), 3))

    if random.uniform(0, 100) < 1:
        baby = mutation(baby)
    return baby


def breeding(mother):
    n = len(mother)
    child = []
    for i in range(len(mother)):
        for j in range(i):
            child.append(offspring(mother[i], mother[j]))

    InS = {}
    total = 0
    for i, j in enumerate(child):
        InS[i] = surface(j)
        total += InS[i]

    p = []
    for i in range(len(child)):
        p.append(InS[i] / total)

    next_index = np.random.choice(range(len(child)), n, replace=False, p=p)
    next_generation = list(map(lambda x: child[x], next_index))

    maxS = surface(next_generation[0])

    for i in range(1, n):
        temp = surface(next_generation[i])

        if temp > maxS:
            maxS = temp
            next_generation[0], next_generation[i] = next_generation[i], next_generation[0]

    return (next_generation, maxS)


def mutation(chromosome):
    if random.choice([True, False]):
        return cross(chromosome)
    else:
        return reset(chromosome)


def cross(chromosome):
    n = random.randint(1, 4)
    index = (0, 1, 2, 3, 4, 5)

    for _ in range(n):
        toSwitch = random.sample(index, 2)
        chromosome[toSwitch[0]], chromosome[toSwitch[1]
                                            ] = chromosome[toSwitch[1]], chromosome[toSwitch[0]]

    return chromosome


def reset(chromosome):
    global maxP
    index = (0, 1, 2, 3, 4, 5)
    sample = random.sample(index, random.randint(1, 5))
    for i in sample:
        chromosome[i] = random.uniform(0.1, maxP-1)
    return chromosome


maxP = 15

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set_aspect('equal')

drawGrid(maxP)
p = toPoint(np.random.randint(3, 15, 6))
drawPoint(p)
ax1.set_title('example_1')

ax2 = fig.add_subplot(122)
ax2.set_aspect('equal')

drawGrid(maxP)
p = toPoint(np.random.randint(3, 15, 6))
drawPoint(p)
ax2.set_title('example_2')

plt.show()
