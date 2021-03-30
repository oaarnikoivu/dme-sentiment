# Find the optimal number of components to maintain from SVD,PCA
def select_n_components(var_ratio, goal_var: float) -> int:
    total_variance = 0.0
    n_components = 0

    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1

        if total_variance >= goal_var:
            break
            
    return n_components