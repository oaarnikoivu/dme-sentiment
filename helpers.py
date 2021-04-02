from textblob import TextBlob

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

# Find the number of adjective POS tags in text 
adj = ['JJ', 'JJR', 'JJS']

def pos_check(x):
    count = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in adj:
                count += 1
    except:
        pass
    return count