from TS import TabuSearch
import numpy as np
from util import keanes_bump_ts

def run_multiple_ts(N, f, D, algo_param, log=False):
    '''
    Runs N seeded runs of the tabu search and returns
    an array containing cost of best solution found in
    each case
    '''
    varbound=np.array([[0,10]]*D)
    best_arr = []
    
    for n in range(N):
        model=TabuSearch(f, D, varbound, n, algo_param)
        model.run()
        best = (model.best_x, model.best_obj)
        best_arr.append(best)

        if log:
            print(f'Model finished running for random seed = {n}')
    
    return best_arr


def grid_search_alpha_delta0(alpha_arr, delta0_arr, n_cells, D=8, log=False):
    f = keanes_bump_ts
    means = np.zeros((len(alpha_arr), len(delta0_arr)))

    for i, alpha in enumerate(alpha_arr):
        for j, delta0 in enumerate(delta0_arr):
            if log:
                print('Starting for alpha={}, delta_0={}'.format(alpha, delta0))
            
            algo_param  = {'search_method': 'exhaustive', 
                            'N': 7,
                            'M': 4, 
                            'i_count': 10,
                            'd_count': 15,
                            'r_count': 25,
                            'subset_size': 4,
                            'prioritisation_period': 10,
                            'initial_step_size': delta0,
                            'reduce_factor': alpha,
                            'n_cells': n_cells}
            
            best_arr = run_multiple_ts(50, f, D, algo_param, log=False)
            mean = np.mean([sol[1] for sol in best_arr])
            means[i][j] = mean

    return alpha_arr, delta0_arr, means


def grid_search_alpha_subset(alpha_arr, subset_arr, n_cells, search_method, D=8, log=False):
    f = keanes_bump_ts
    means = np.zeros((len(alpha_arr), len(subset_arr)))

    for i, alpha in enumerate(alpha_arr):
        for j, subset_size in enumerate(subset_arr):
            if log:
                print('Starting for alpha={}, subset_size={}'.format(alpha, subset_size))
            
            algo_param  = {'search_method': search_method, 
                            'N': 7,
                            'M': 4, 
                            'i_count': 10,
                            'd_count': 15,
                            'r_count': 25,
                            'subset_size': subset_size,
                            'prioritisation_period': 10,
                            'initial_step_size': 3,
                            'reduce_factor': alpha,
                            'n_cells': n_cells}
            
            best_arr = run_multiple_ts(50, f, D, algo_param, log=False)
            mean = np.mean([sol[1] for sol in best_arr])
            means[i][j] = mean

    return means

            
def grid_search_alpha_pri_period(alpha_arr, pp_arr, n_cells, search_method, D=8, log=False):
    f = keanes_bump_ts
    means = np.zeros((len(alpha_arr), len(pp_arr)))

    for i, alpha in enumerate(alpha_arr):
        for j, pp in enumerate(pp_arr):
            if log:
                print('Starting for alpha={}, prioritisation_period={}'.format(alpha, pp))
            
            algo_param  = {'search_method': search_method, 
                            'N': 7,
                            'M': 4, 
                            'i_count': 10,
                            'd_count': 15,
                            'r_count': 25,
                            'subset_size': 4,
                            'prioritisation_period': pp,
                            'initial_step_size': 3,
                            'reduce_factor': alpha,
                            'n_cells': n_cells}
            
            best_arr = run_multiple_ts(50, f, D, algo_param, log=False)
            mean = np.mean([sol[1] for sol in best_arr])
            means[i][j] = mean

    return means      