import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from collections import defaultdict
import matplotlib.pyplot as plt


def keanes_bump_ga(x, w=1e10, adaptive_mean=None, penalty=True):
    '''
    Keane's Bump function to be maximised with
    elitist GA
    '''
    n = len(x)
    cos_4_sum = np.sum(np.cos(x)**4)
    cos_2_prod = np.prod(np.cos(x)**2)

    num = cos_4_sum - 2 * cos_2_prod
    denom = np.sqrt(np.sum([(i+1) * x[i]**2 for i in range(n)]))
    cost = np.abs(num/denom)

    # Add penalties for broken constraints
    pen = 0
    coeff = w

    if adaptive_mean is not None:
        if not valid(x) and cost > adaptive_mean:
            cost = adaptive_mean
    
    if penalty:
        for i in range(n):
            pen -= coeff * max(max(0, -x[i]), max(0, x[i]-10)) # applies penalty for violating 0 <= x_i <= 10
        
        pen -= coeff * (max(0, np.sum(x) - 15*n/2)) # applies penalty for violating sum(x) < 15n/2
        pen -= coeff * (max(0, -np.prod(x) + 0.75)) # applies penalty for violating prod(x) > 0.75

    return -1 * (cost + pen)


def keanes_bump_ts(x, w=1e8, penalty=True):
    '''
    Keane's Bump function to be maximised with 
    Tabu Search
    '''
    n = len(x)
    cos_4_sum = np.sum(np.cos(x)**4)
    cos_2_prod = np.prod(np.cos(x)**2)

    num = cos_4_sum - 2 * cos_2_prod
    denom = np.sqrt(np.sum([(i+1) * x[i]**2 for i in range(n)]))
    cost = np.abs(num/denom)

    # Add penalties for broken constraints
    pen = 0
    
    if penalty:
        for i in range(n):
            pen -= w * max(max(0, -x[i]), max(0, x[i]-10)) # applies penalty for violating 0 <= x_i <= 10
        
        pen -= w * (max(0, np.sum(x) - 15*n/2)) # applies penalty for violating sum(x) < 15n/2
        pen -= w * (max(0, -np.prod(x) + 0.75)) # applies penalty for violating prod(x) > 0.75

    return (cost + pen)


def contour_xyz(x_range, y_range, res=0.1):
    '''
    Takes a range of x and y values and returns x, y
    and z in the form to be taken by the matplotlib
    contourf function.
    '''
    x_contour = np.arange(x_range[0], x_range[1], res)
    y_contour = np.arange(y_range[0], y_range[1], res)
    x_mesh, y_mesh = np.meshgrid(x_contour, y_contour)
    xy_mesh = np.stack([x_mesh, y_mesh], axis=-1)
    z = np.zeros(x_mesh.shape)
    for i in range(xy_mesh.shape[0]):
        for j in range(xy_mesh.shape[1]):
            if not valid(xy_mesh[i][j]):
                z[i][j] = -0.5
            else:
                z[i][j] = keanes_bump_ts(xy_mesh[i][j])
    return x_mesh, y_mesh, z


def plot_region_distribution(solutions, n_cells=3, algo_name='Tabu Search'):
    '''
    Function that displays a frequency plot of the regions visited
    by an algorithm
    '''
    region_counts = defaultdict(int)

    for x in solutions:
        grid_coord = np.zeros(len(x))
        for i, var in enumerate(x):
            start, stop = 0, 10
            grid_spacing = np.linspace(start, stop, n_cells+1)
            grid_coord[i] = np.searchsorted(grid_spacing, var)
        region_counts[tuple(grid_coord)] += 1
    
    counts = sorted(region_counts.values())
    fig, ax = plt.subplots()
    x_pos = np.linspace(0, len(counts), len(counts))
    ax.bar(x_pos,counts, width=0.9)
    ax.set_yscale('log')
    ax.set_title('{}: {}/{} cells used for parent selection'.format(algo_name,
                                                               len(counts), 
                                                               int(n_cells**8)))
    ax.set_ylabel('Frequency of Parent Selections')
    plt.show()
    

def valid(x):
    for i in range(len(x)):
        if x[i] > 10 or x[i] < 0:
            return False

    if np.sum(x) >= 15*len(x)/2:
        return False
    
    if np.prod(x) <= 0.75:
        return False
    
    return True


if __name__ == '__main__':
    n = 100
    x1_arr = np.linspace(0, 10, n)
    x2_arr = np.linspace(0, 10, n)
    x1, x2 = np.meshgrid(x1_arr, x2_arr)
    contour_xyz(keanes_bump_ts, [0, 10], [0, 10])
    inp = np.stack([x1, x2], axis=-1)

    z = np.zeros((n, n))
    z_andy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            z[i][j] = keanes_bump_ts(inp[i][j])
            if not valid(inp[i][j]):
                z[i][j] = -0.5
    
    print(np.max(z))
    
    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    

    