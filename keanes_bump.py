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


def keanes_bump_ts(x, w=1000, penalty=True):
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

def valid_ga(x):
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
    inp = np.stack([x1, x2], axis=-1)

    z = np.zeros((n, n))
    z_andy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            z[i][j] = keanes_bump_ts(inp[i][j])
            if not valid_ga(inp[i][j]):
                z[i][j] = -1
    
    print(np.max(z))
    
    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    

    