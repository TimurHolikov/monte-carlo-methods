#%%
from random import choice
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')

DIRS = [(1,0),(0,1),(-1,0),(0,-1)]

def self_avoiding_walk_path():
    x,y = 0,0
    visited = {(x,y)}
    prev = None
    path = [(x,y)]

    while True:
        allowed = []
        for dx,dy in DIRS:
            if prev and (dx,dy)==(-prev[0],-prev[1]):
                continue

            nx,ny = x+dx,y+dy
            if (nx,ny) in visited:
                continue
            
            allowed.append((dx,dy))

        if not allowed:
            break

        dx,dy = choice(allowed)

        x += dx
        y += dy

        visited.add((x,y))
        path.append((x,y))

        prev = (dx,dy)

    return path

def sample_lengths(M):
    lengths = []
    for _ in range(M):
        lengths.append(len(self_avoiding_walk_path())-1)

    return lengths

def plot_walk(path):
    xs,ys = zip(*path)

    plt.figure(figsize=(5,5))
    
    plt.plot(xs,ys,lw=1.5)
    plt.scatter(xs[0],ys[0],s=70,label="Start")
    plt.scatter(xs[-1],ys[-1],s=70,label="End")

    plt.axis("equal")
    plt.title(f"Self-avoiding walk (length={len(path)-1})")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    path = self_avoiding_walk_path()
    plot_walk(path)

    M = 20000
    lengths = sample_lengths(M)

    print("mean length:", sum(lengths)/len(lengths))

    plt.figure(figsize=(7,5))
    plt.hist(lengths,bins=50,density=True)
    plt.xlabel("Length of self-avoiding walk")
    plt.ylabel("Probability density")
    plt.title("Distribution of walk lengths")
    plt.show()
#%%