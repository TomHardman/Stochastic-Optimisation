from collections import deque
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import random
from heapq import *
from keanes_bump import keanes_bump_ts


class STM():
    def __init__(self, N):
        self.N = N
        self.STM_queue = deque([None for _ in range(self.N)])
    
    def update(self, x):
        self.STM_queue.popleft() # remove oldest solution
        self.STM_queue.append(x) # add new solution to queue
    
    def __contains__(self, x):
        return x in self.STM_queue
    

class MTM():
    def __init__(self, M):
        self.M = M
        self.MTM_heap = [] # implement min-heap to store solutions
        heapify(self.MTM_heap)
    
    def update(self, x, obj):
        if len(self.MTM_heap) < self.M:
            self.MTM_heap.heappush((obj, x))
        
        else:
            if obj > self.MTM_heap[0][0]: # if solution being considered better than M-th best
                # add solution to MTM and remove worst solution in MTM
                self.MTM_heap.heappop
                self.MTM_heap.heappush((obj, x))
    
    def get_mean(self):
        x_arr = np.array([pair[1] for pair in self.MTM_heap])
        mean_x = np.mean(x_arr, axis=1)
        return mean_x


class LTM():
    def __init__(self, varbound, n_cells):
        self.varbound = varbound
        self.D = len(varbound)
        self.n_cells = n_cells
        self.coords_set = self.generate_all_coords() # set of all possible grid co-ords
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
        if grid in self.coords_set:
            self.coords_set.discard(grid)
            self.tallies[grid] = 1

        else:
            self.tallies[grid] += 1
    
    def generate_new_solution(self):
        # try to randomly generate solution from set of unvisited regions
        if self.coords_set:
            grid_coord = np.array(random.choice(list(self.coords_set)))
        
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
    def __init__(self, f, search_method, D, var_bound, N=7, M=4, 
                 intensify=10, diversify=15, reduce=25, subset_size=4, 
                 prioritisation_period=10, initial_step_size=0.5, 
                 reduce_factor=2, n_cells=2):
        
        self.N = N
        self.M = M
        self.D = D
        self.i_count=intensify
        self.d_count=diversify
        self.r_count=reduce
        self.var_bound = var_bound
        self.f = f

        self.subset_size = subset_size
        self.search_method = search_method
        self.prioritisation_period = prioritisation_period
        self.reduce_factor = reduce_factor

        self.MTM = MTM(M)
        self.STM = STM(N)
        self.LTM = LTM(var_bound, n_cells)
        self.solutions = []
        self.best_solutions = []

        self.eval_count = 0
        self.n_iter = 0
        self.step_size = initial_step_size
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
        self.best_solutions.append(self.best_x, self.best_obj)
        self.solutions.append(base, current_obj)

        while self.eval_count < 10000:
            next_base, next_obj = self.local_search(base, current_obj)
            self.n_iter += 1
            self.update_memories(next_base, next_obj)
            
            if next_obj > current_obj: # if best move improves objective function
                # find successive base with pattern move
                pattern = next_base - base
                suc_base = next_base + pattern 
                
                if suc_base not in self.STM_set and\
                self.valid(suc_base): # if move valid
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

            #if new best not found increase count
            if current_obj < self.best_obj:
                counter += 1

                if counter == self.i_count:
                    base = self.MTM.get_mean() # intensify search
                    current_obj = self.f(base)
                    self.eval_count += 1
                    self.update_memories(base, current_obj)
                
                if counter == self.d_count:
                    base = self.LTM.generate_new_solution # diversify search
                    current_obj = self.f(base)
                    self.eval_count += 1
                    self.update_memories(base, current_obj)
                
                if counter == self.r_count:
                    self.step_size = self.step_size*self.reduce_factor # reduce step size
                    counter = 0
            
            else:
                self.best_obj = current_obj
                self.best_x = base
            
            # Report best solution
            self.best_solutions.append(self.best_x, self.best_obj)
            self.solutions.append(base, current_obj)
 
    def update_memories(self, x, obj):
        self.STM.update(x)
        self.MTM.update(x, obj)
        self.LTM.update(x)
        self.solutions.append((x, obj))
    
    def valid(self, x):
        return all([x[i] >= self.varbound[i][0] and x[i] <= self.varbound[i][1] 
                    for i in range(self.D)])
    
    def local_search(self, base, current_obj):
        if self.search_method == 'exhaustive':
            next_base, next_obj = self.exhaustive_search(base)
        elif self.search_method == 'random_subset':
            next_base, next_obj = self.random_subset_search(base)
        elif self.search_method == 'successive_random_subset':
            next_base, next_obj = self.successive_rss(base, current_obj)
        else:
            next_base, next_obj = self.variable_prioritsation_search(base)
        return next_base, next_obj

    def search_along_axes(self, base, axes):
        best_obj = -np.inf
        best_x = None

        for ax in axes:
            next_x = base.copy()
            for step in [self.step_size, -2*self.step_size]:
                next_x[ax] += step

                if next_x not in self.STM: # if not tabu
                    next_obj = self.f(next_x)
                    self.eval_count += 1
                    
                    if next_obj > best_obj:
                        best_x = next_x
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
        axes_set = set(range(0, self.D))
        best_obj = -np.inf
        
        while current_obj - best_obj <= 0 and axes_set:
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
            
    def variable_prioritsation_search(self, base):
        if self.n_iter % self.prioritisation_period == 0:
            axes = range(0, self.D)
            sensitivities = []
            for ax in axes:
                delta = np.zeros(self.D)
                delta[ax] += self.step_size
                s_i = np.linalg.norm((self.f(base+delta) - self.f(base-delta))/(2 * self.step_size))
                sensitivities.append(s_i)
            
            sens_ranking = np.argsort(sensitivities)
            self.vp_axes = sens_ranking[:self.subset_size]
        
        best_x, best_obj = self.search_along_axes(base, self.vp_axes)
        return best_x, best_obj
    
    def initialise_start_point(self):
        start_point = np.zeros(self.D)
        for i in range(self.D):
            lower = self.var_bound[i][0]
            upper = self.var_bound[i][1]
            start_point[i] = np.random.uniform(lower, upper)
        return start_point

    def plot_best_evolution(self):
        plt.plot([sol[1] for sol in self.best_solutions])
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()
        

if __name__ == '__main__':
    D=2
    varbound=np.array([[0,10]]*D)
    f = keanes_bump_ts
    tabu_optimiser = TabuSearch(f, 'exhaustive', D, varbound, N=7, M=4, 
                           intensify=10, diversify=15, reduce=25, subset_size=4, 
                           prioritisation_period=10, initial_step_size=0.5, 
                           reduce_factor=2, n_cells=2)
    tabu_optimiser.run()
    tabu_optimiser.plot_best_evolution()

    
    