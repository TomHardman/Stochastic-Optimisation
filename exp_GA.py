import numpy as np
from ga import geneticalgorithm
from util import keanes_bump_ga

def run_multiple_ga(N, algo_param, D, log=False):
    varbound=np.array([[0,10]]*D)
    opt_arr = []
    res_arr = []
    
    for n in range(N):
        model=geneticalgorithm(function=keanes_bump_ga, dimension=D, variable_type='real',
                               variable_boundaries=varbound, algorithm_parameters=algo_param, 
                               random_seed=n, progress_bar=False, convergence_curve=False)
        model.run()
        opt = model.output_dict
        res = model.report
        opt_arr.append(opt)
        res_arr.append(res)
        if log:
            print(f'Model finished running for random seed = {n}')
    
    return opt_arr, res_arr


def grid_search_alpha_N(N_arr, alpha_arr, D=8, log=False):
    means = np.zeros((len(N_arr), len(alpha_arr)))

    for i, N in enumerate(N_arr):
        for j, alpha in enumerate(alpha_arr):
            if log:
                print('Starting for N={}, alpha={}'.format(N, alpha))
            
            algo_param={'max_num_iteration': 10000,
                'population_size': N,
                'mutation_probability':0.01,
                'elit_ratio': 0.03,
                'crossover_probability': 0.45,
                'parents_portion': alpha,
                'crossover_type':'uniform',
                'max_iteration_without_improv':None,
                'selection_method': 'ranking_based_srs',
                'selection_pressure': 1.4,
                'offset': 1,
                'penalty_coeff': 1}
            
            opt_arr, res_arr = run_multiple_ga(50, algo_param, 8)
            opt_values = [opt_arr[i]['function'] for i in range(len(opt_arr))]    
            mean = np.mean(opt_values)
            if log:
                print(mean)
            means[i][j] = mean

    return means


def grid_search_mutation_crossover(p_m_arr, p_c_arr, D=8, log=False):
    means = np.zeros((len(p_m_arr), len(p_m_arr)))

    for i, p_m in enumerate(p_m_arr):
        for j, p_c in enumerate(p_c_arr):
            if log:
                print('Starting for P_m={}, P_c={}'.format(p_m, p_c))
            
            algo_param={'max_num_iteration': 10000,
                'population_size': 100,
                'mutation_probability':p_m,
                'elit_ratio': 0.03,
                'crossover_probability': p_c,
                'parents_portion': 0.5,
                'crossover_type':'uniform',
                'max_iteration_without_improv':None,
                'selection_method': 'ranking_based_srs',
                'selection_pressure': 1.4,
                'offset': 1,
                'penalty_coeff': 1}
            
            opt_arr, res_arr = run_multiple_ga(50, algo_param, 8)
            opt_values = [opt_arr[i]['function'] for i in range(len(opt_arr))]    
            mean = np.mean(opt_values)
            if log:
                print(mean)
            means[i][j] = mean

    return means


def line_search_elit(elit_arr, D=8, log=False):
    means = np.zeros(len(elit_arr))

    for i, elit in enumerate(elit_arr):
            if log:
                print('Starting for elit_ratio={}'.format(elit))
            
            algo_param={'max_num_iteration': 10000,
                'population_size': 100,
                'mutation_probability':0.1,
                'elit_ratio': elit,
                'crossover_probability': 0.6,
                'parents_portion': 0.5,
                'crossover_type':'uniform',
                'max_iteration_without_improv':None,
                'selection_method': 'ranking_based_srs',
                'selection_pressure': 1.4,
                'offset': 1,
                'penalty_coeff': 1}
            
            opt_arr, res_arr = run_multiple_ga(50, algo_param, 8)
            opt_values = [opt_arr[i]['function'] for i in range(len(opt_arr))]    
            mean = np.mean(opt_values)
            if log:
                print(mean)
            means[i] = mean

    return means