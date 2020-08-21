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


fig, ax = plt.subplots(figsize=(15, 25))
maxP = 15
n = 6
iterNum = 30

superior = {'surf': 0,
            'point': 1}
tenth_generation = []
maxS = []

F = np.random.uniform(0.1, maxP/2, (n, 6))
F_1 = F
for genes in F:
    temp = surface(genes)
    if temp > superior['surf']:
        superior['surf'] = temp
        superior['point'] = genes

maxS.append(superior['surf'])

for iteration in range(iterNum):
    F, temp = breeding(F)

    maxS.append(temp)
    if temp > superior['surf']:
        superior['surf'] = temp
        superior['point'] = F[0]

    if iteration % (iterNum / 10) == (iterNum/10-1):
        tenth_generation.append(F[0])

location = ((0, 0), (0, 2), (2, 0), (2, 2), (4, 0), (4, 2))
for i in range(6):
    p = toPoint(F_1[i])
    plt.subplot2grid((14, 10), location[i], colspan=2, rowspan=2)
    plt.tick_params(labelbottom=False, labelleft=False,
                    bottom=False, left=False)
    plt.xlim(-(maxP+0.5), maxP+0.5)
    plt.ylim(-(maxP+0.5), maxP+0.5)
    plt.title('F1_{}'.format(i+1))
    drawGrid(maxP)
    drawPoint(p)

location = ((6, 0), (6, 2), (6, 4), (6, 6), (6, 8),
            (8, 0), (8, 2), (8, 4), (8, 6), (8, 8))
for i in range(10):
    p = toPoint(tenth_generation[i])
    plt.subplot2grid((14, 10), location[i], colspan=2, rowspan=2)
    plt.tick_params(labelbottom=False, labelleft=False,
                    bottom=False, left=False)
    plt.xlim(-(maxP+0.5), maxP+0.5)
    plt.ylim(-(maxP+0.5), maxP+0.5)
    plt.title('Tenth_{}'.format(i+1))
    drawGrid(maxP)
    drawPoint(p)

great = toPoint(superior['point'])
plt.subplot2grid((14, 10), (0, 4), colspan=6, rowspan=6)
plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
plt.xlim(-(maxP+0.5), maxP+0.5)
plt.ylim(-(maxP+0.5), maxP+0.5)
plt.title("Most Superior")
drawGrid(maxP)
drawPoint(great)

graph = plt.subplot2grid((14, 10), (10, 0), colspan=10, rowspan=4)
graph.tick_params(labelleft=False, left=False)
plt.xlabel('Numbers of Generation')
plt.ylabel('Size of Surface')
graph.xaxis.set_tick_params(labelsize=20)
plt.title('Variation of Max-Surface of each Generation')
plt.plot(range(len(maxS)), maxS, linewidth=4)

plt.savefig('result1.png')
