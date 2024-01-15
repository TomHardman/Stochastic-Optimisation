import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt


def keanes_bump_ga(x, n_iter=1, w=1, k=2, penalty=True):
    '''
    Keane's Bump function to be maximised
    '''
    n = len(x)
    cos_4_sum = np.sum(np.cos(x)**4)
    cos_2_prod = np.prod(np.cos(x)**2)

    num = cos_4_sum - 2 * cos_2_prod
    denom = np.sqrt(np.sum([(i+1) * x[i]**2 for i in range(n)]))
    cost = np.abs(num/denom)

    # Add penalties for broken constraints
    pen = 0
    coeff = n_iter ** k * w
    
    if penalty:
        for i in range(n):
            pen -= coeff * max(max(0, -x[i]), max(0, x[i]-10)) # applies penalty for violating 0 <= x_i <= 10
        
        pen -= coeff * (max(0, np.sum(x) - 15*n/2)) # applies penalty for violating sum(x) < 15n/2
        pen -= coeff * (max(0, -np.prod(x) + 0.75)) # applies penalty for violating prod(x) > 0.75

    return -1 * (cost + pen)


def keanes_bump_ts(x, w=1e8, penalty=True):
    '''
    Keane's Bump function to be maximised
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


def contour_xyz(f, x_range, y_range, res=0.1):
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
                z[i][j] = f(xy_mesh[i][j])
    return x_mesh, y_mesh, z


def contour_plot_gs(x_mesh, y_mesh, means):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    cp = ax.contourf(x_mesh, y_mesh, means, levels=10)
    best_mean = np.max(means)
    x_i, y_i = np.where(means == best_mean)
    ax.scatter  


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

    

    