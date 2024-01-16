from collections import deque
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from util import *
import random


class STM():
    '''
    Class that implements the short term memory for the Tabu Search

    Parameters
    ------------
    N: no. of points to be stored in memory
    D: no. of dimensions of optimisation problem

    Methods
    ------------
    update(self, x):
        Updates the STM by removing oldest solution and adding
        newest
    
    __contains__(self, x):
        Returns True if point x is in short term memory. False
        Otherwise
    '''
    def __init__(self, N, D):
        self.N = N
        self.STM_queue = deque([np.full(D, np.inf) for _ in range(self.N)])
    
    def update(self, x):
        self.STM_queue.popleft() # remove oldest solution
        self.STM_queue.append(x) # add new solution to queue
    
    def __contains__(self, x):
        for point in self.STM_queue:
           if (x == point).all():
              return True
        return False
    

class MTM():
    '''
    Class that implements the medium term memory for the Tabu Search

    Parameters
    ------------
    M: no. of points to be stored in memory

    Methods
    ------------
    update(self, x):
        Updates the STM by removing M-th best solution and adding
        current if current better solution is
    
    get_mean(x):
        Returns the mean location of all the points currently
        in the medium term memory. Used during search diversification
    
    contains(self, x):
        Returns True if point x is in medium term memory. False
        Otherwise
    '''
    def __init__(self, M):
        self.M = M
        self.MTM_arr =[]
    
    def update(self, x, obj):
        if len(self.MTM_arr) < self.M:
            self.MTM_arr.append((obj, x))
        
        else:
            # if solution being considered better than M-th best
            self.MTM_arr = sorted(self.MTM_arr, reverse=True, key=lambda x:x[0])
            if obj > self.MTM_arr[-1][0]:
                # add solution to MTM and remove worst solution in MTM
                self.MTM_arr.pop()
                self.MTM_arr.append((obj, x))
    
    def get_mean(self):
        x_arr = np.array([pair[1] for pair in self.MTM_arr])
        mean_x = np.mean(x_arr, axis=0)
        return mean_x
    
    def contains(self, x):
        for pair in self.MTM_arr:
           if (x == pair[1]).all():
              return True
        return False


class LTM():
    '''
    Class that implements the long term memory for the Tabu Search

    Parameters
    ------------
    varbound: array like, 2D array of bounds for each variable that
              defines the search space of form: 
              [[start_x1, stop_x1], [start_x2, stop_x2]...]

    n_cells: the number of discrete cells to split each axis into
             for discretisation of the search space

    Methods
    ------------
    global_to_grid(self, x): 
        converts global co-ordinates to grid co-ordinates
    
    grid_to_random_global(self, x):
        converts grid co-ordinates to a global co-ordinate randomly
        and uniformly sampled from within the bounds of the grid cell
    
    update(self, x):
        updates the LTM for a successfully visited point x
    
    generate_new_solution(self ,x):
        randomly choose a grid cell from set of unvisited regions
        or least visited regions in case all regions have been visited
        at least once
    
    generate_all_solutions(self):
        generates a set of grid co-ordinates for all grid cells in
        the search space
    '''
    def __init__(self, varbound, n_cells):
        self.varbound = varbound
        self.D = len(varbound)
        self.n_cells = n_cells
        self.unvisited_cells = self.generate_all_coords() # set of all possible grid co-ords
        self.tallies = {} # tallies of no. times each grid region is visited
         
    def global_to_grid(self, x): # converts global co-ordinates to grid-coordinates
        grid_coord = np.zeros(len(x))

        for i, var in enumerate(x):
            start, stop = self.varbound[i]
            grid_spacing = np.linspace(start, stop, self.n_cells+1)
            grid_coord[i] = np.searchsorted(grid_spacing, var)
        
        return grid_coord
    
    def grid_to_random_global(self, grid_coord):
        global_coord = np.zeros(len(grid_coord))

        for i in range(self.D):
            start, stop = self.varbound[i]
            grid_spacing = np.linspace(start, stop, self.n_cells+1)
            lower = grid_spacing[grid_coord[i] - 1]
            upper = grid_spacing[grid_coord[i]]
            global_coord[i] = np.random.uniform(lower, upper)
        
        return global_coord

    def update(self, x):
        grid = self.global_to_grid(x)
        grid = tuple(grid.astype(int))
        
        if grid in self.unvisited_cells:
            self.unvisited_cells.discard(grid)
            self.tallies[grid] = 1

        else:
            self.tallies[grid] += 1
    
    def generate_new_solution(self):
        # try to randomly generate solution from set of unvisited regions
        if self.unvisited_cells:
            grid_coord = np.array(random.choice(list(self.unvisited_cells)))
        
        # if all regions have been visited
        # chose least-visited region
        else:
            sorted_regions = sorted(self.tallies, key=self.tallies.get)
            grid_coord=np.array(sorted_regions[0])
        
        x = self.grid_to_random_global(grid_coord)
        return x

    def generate_all_coords(self):
        possible_vals = range(1, self.n_cells+1)
        all_coords = set(product(possible_vals, repeat=self.D))
        return all_coords


class TabuSearch():
    '''
    Class for implementing a function maximisation via Tabu Search

    Parameters
    -------------
    f: function to be optimised

    D: no. dimensions of problem

    var_bound: array like, 2D array of bounds for each variable that
              defines the search space of form: 
              [[start_x1, stop_x1], [start_x2, stop_x2]...]

    random_seed: Random seed for seeded runs
    
    algo_param - dictionary to contain the following parameters
        N: no. points to store in short term memory

        M: no. points to store in medium term memory

        i_count: Threshold that needs to be reached for intensification

        d_count: Threshold that needs to be reached for diversification

        r_count: Threshold that needs to be reached for step reduction

        subset_size: Size of subset used during random subset, successive
                     random subset or variable prioritisation search

        initial_step_size: Initial step size for the search

        reduce_factor: factor to reduce step size by during step reduction

        n_cells : number of discrete cells along each axis used when
                  discretising the search space in the LTM
    
    External Methods
    ----------------------------
    run(self):
        runs the model
    
    plot_best_evolution(self):
        displays plot of the evolution of the best solution 
        against no. iterations
    
    map_accepted_solutions(self, levels=30, best_evo=True, save_fig=False):
        For use with 2D optimisation only. Display contour map of function 
        with accepted solutions and evolutions of best solution if best_evo
        is set to true
    '''
    def __init__(self, f, D, var_bound, random_seed=None,
                 algo_param = {'search_method': 'exhaustive', 
                              'N': 7,
                              'M': 4, 
                              'i_count': 10,
                              'd_count': 15,
                              'r_count': 25,
                              'subset_size': 4,
                              'prioritisation_period': 10,
                              'initial_step_size': 1,
                              'reduce_factor': 0.8,
                              'n_cells': 3}):
        
        # check valid search method has been provided
        search_method = algo_param['search_method']
        valid_methods = ['exhaustive', 'random_subset', 'successive_random_subset',
                         'variable_prioritisation']
        assert(search_method in valid_methods),\
            '{} is not a valid search method, valid methods are: {}'.format(search_method, 
                                                                            valid_methods)
        # check no. dimensions is consistent
        assert(len(var_bound) == D),\
            'var_bound must be of length D, D={}'.format(D)
        
        # set random_seed
        if random_seed:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        self.N = algo_param['N']
        self.M = algo_param['M']
        self.D = D
        self.i_count=algo_param['i_count']
        self.d_count=algo_param['d_count']
        self.r_count=algo_param['r_count']
        self.var_bound = var_bound
        self.f = f

        self.subset_size = algo_param['subset_size']
        self.search_method = search_method
        self.prioritisation_period = algo_param['prioritisation_period']
        self.reduce_factor = algo_param['reduce_factor']
        self.n_cells = algo_param['n_cells']

        self.MTM = MTM(self.M)
        self.STM = STM(self.N, D)
        self.LTM = LTM(var_bound, self.n_cells)
        self.solutions = []
        self.best_solutions = []

        self.eval_count = 0
        self.n_iter = 0
        self.step_size = algo_param['initial_step_size']
        self.best_obj = -np.inf

    def run(self):
        start = self.initialise_start_point() # random start point
        base = start # set first base
        current_obj = self.f(base) # evaluate function at base
        self.eval_count += 1
        counter = 0

        # Record start point
        self.update_memories(base, current_obj)
        self.best_obj = current_obj
        self.best_x = base
        self.best_solutions.append((self.best_x, self.best_obj))
        self.solutions.append((base, current_obj))

        while self.eval_count < 10000:
            next_base, next_obj = self.local_search(base, current_obj)
            self.n_iter += 1
            self.update_memories(next_base, next_obj)
            
            if next_obj > current_obj: # if best move improves objective function
                # find successive base with pattern move
                pattern = next_base - base
                suc_base = next_base + pattern 
                
                if suc_base not in self.STM and self.valid(suc_base): # if move valid
                    suc_obj = self.f(suc_base)
                    self.eval_count += 1

                    # if pattern move doesn't improve objective function
                    if suc_obj < next_obj: # reject pattern move
                        base = next_base.copy()
                        current_obj = next_obj
                
                    else: # accept pattern move
                        base = suc_base.copy()
                        current_obj = suc_obj
                        self.update_memories(suc_base, suc_obj)
                
                else:
                    base = next_base.copy()
                    current_obj = next_obj
            
            else:
                base = next_base.copy()
                current_obj = next_obj

            #if new best not found increase count
            if current_obj < self.best_obj:
                counter += 1

                if counter == self.i_count:
                    base = self.MTM.get_mean().copy() # intensify search
                    current_obj = self.f(base)
                    self.eval_count += 1
                    self.update_memories(base, current_obj)
                
                if counter == self.d_count:
                    base = self.LTM.generate_new_solution().copy() # diversify search
                    current_obj = self.f(base)
                    self.eval_count += 1
                    self.update_memories(base, current_obj)
                
                if counter == self.r_count:
                    self.step_size = self.step_size*self.reduce_factor # reduce step size
                    counter = 0
            
            else:
                self.best_obj = current_obj
                self.best_x = base.copy()
            
            # Report best solution
            self.best_solutions.append((self.best_x, self.best_obj))
            self.solutions.append((base, current_obj))
 
    def update_memories(self, x, obj):
        self.STM.update(x)
        self.MTM.update(x, obj)
        self.LTM.update(x)
        self.solutions.append((x, obj))
    
    def valid(self, x):
        return all([x[i] >= self.var_bound[i][0] and x[i] <= self.var_bound[i][1] 
                    for i in range(self.D)])
    
    def local_search(self, base, current_obj):
        if self.search_method == 'exhaustive':
            next_base, next_obj = self.exhaustive_search(base)
        elif self.search_method == 'random_subset':
            next_base, next_obj = self.random_subset_search(base)
        elif self.search_method == 'successive_random_subset':
            next_base, next_obj = self.successive_rss(base, current_obj)
        else:
            next_base, next_obj = self.variable_prioritisation_search(base)
        return next_base, next_obj

    def search_along_axes(self, base, axes):
        # searches along axes in 'axes' and returns new base and cost
        best_obj = -np.inf
        best_x = base
        for ax in axes:
            next_x = base.copy()
            for step in [self.step_size, -2*self.step_size]:
                next_x[ax] += step

                if next_x not in self.STM and self.valid(next_x): # if valid move
                    next_obj = self.f(next_x)
                    self.eval_count += 1
                    
                    if next_obj > best_obj:
                        best_x = next_x.copy()
                        best_obj = next_obj
    
        return best_x, best_obj

    def exhaustive_search(self, base):
        axes = range(0, self.D)
        best_x, best_obj = self.search_along_axes(base, axes)
        return best_x, best_obj
    
    def random_subset_search(self, base):
        axes = random.sample(range(self.D), self.subset_size)
        best_x, best_obj = self.search_along_axes(base, axes)
        return best_x, best_obj

    def successive_rss(self, base, current_obj):
        # successive random subset search
        axes_set = set(range(0, self.D))
        best_obj = -np.inf
        
        # while move does not improve cost and we have remaining axes to search
        while current_obj - best_obj >= 0 and axes_set:
            if len(axes_set) >= self.subset_size:
                axes = random.sample(list(axes_set), self.subset_size)
            else:
                axes = list(axes_set)
            
            best_x_iter, best_obj_iter = self.search_along_axes(base, axes)
            if best_obj_iter > best_obj:
                best_obj = best_obj_iter
                best_x = best_x_iter
        
            for ax in axes:
                axes_set.discard(ax)
        
        return best_x, best_obj
            
    def variable_prioritisation_search(self, base):
        if self.n_iter % self.prioritisation_period == 0:
            axes = range(0, self.D)
            sensitivities = []
            for ax in axes:
                delta = np.zeros(self.D)
                delta[ax] += self.step_size
                s_i = np.linalg.norm((self.f(base+delta) - self.f(base-delta))
                                     /(2 * self.step_size))
                sensitivities.append(s_i)
            
            sens_ranking = np.argsort(sensitivities)
            self.vp_axes = sens_ranking[:self.subset_size]
        
        best_x, best_obj = self.search_along_axes(base, self.vp_axes)
        return best_x, best_obj
    
    def initialise_start_point(self):
        # randomly initialise start point
        start_point = np.zeros(self.D)
        for i in range(self.D):
            lower = self.var_bound[i][0]
            upper = self.var_bound[i][1]
            start_point[i] = np.random.uniform(lower, upper)
        return start_point

    def plot_best_evolution(self):
        plt.plot([sol[1] for sol in self.best_solutions])
        plt.xlabel('No. accepted solutions')
        plt.ylabel('Cost')
        plt.show()
    
    def map_accepted_solutions(self, levels=30, best_evo=True, save_fig=False):
        if self.D != 2:
            raise ValueError('Can only map solutions for 2D problems')
        
        x = [s[0][0] for s in self.solutions]
        y = [s[0][1] for s in self.solutions]
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))  
        x_c, y_c, z_c = contour_xyz(self.var_bound[0], self.var_bound[1]+0.01,
                                    res=0.05)
        cp = ax.contourf(x_c, y_c, z_c, levels=levels)
        fig.colorbar(cp)
        ax.scatter(x, y, marker='.', c="orange", label="Accepted solutions")
        ax.set_title("2-D tabu search on Keane's Bump function", fontsize=18)
        if best_evo:
            x_b = [s[0][0] for s in self.best_solutions]
            y_b = [s[0][1] for s in self.best_solutions]
            ax.plot(x_b, y_b, color='r', label='Best solution evolution')
            ax.plot(x_b[0], y_b[0], color='lime', marker='s', 
                       label='Start')
            ax.plot(x_b[-1], y_b[-1], color='cyan', marker='*', 
                       label='Best')
        ax.legend(loc='upper right')
        if save_fig:
            plt.savefig('{}_{}cells.png'.format(str(self.best_obj)[:6], 
                        self.n_cells))
        plt.show()



if __name__ == '__main__':
    D=8
    varbound=np.array([[0,10]]*D)
    f = keanes_bump_ts
    algo_param  = {'search_method': 'random_subset', 
                    'N': 7,
                    'M': 4, 
                    'i_count': 10,
                    'd_count': 15,
                    'r_count': 25,
                    'subset_size': 3,
                    'prioritisation_period': 5,
                    'initial_step_size': 3,
                    'reduce_factor': 0.94,
                    'n_cells': 2}
    tabu_optimiser = TabuSearch(f, D, varbound, 1, algo_param)
    tabu_optimiser.run()
    tabu_optimiser.plot_best_evolution()
    print(tabu_optimiser.best_obj)
    print(tabu_optimiser.best_x)
    #tabu_optimiser.map_accepted_solutions(save_fig=False)
    plot_region_distribution([sol[0] for sol in tabu_optimiser.solutions], n_cells=3)
    
    