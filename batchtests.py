
import itertools
import numpy as np 
import os
import random

from sklearn.metrics import average_precision_score

from iforest import BAD_IForest
import odds_datasets

save_dir = "batch_results/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def run_sim(X, y, batch_size, strategy, seed=0, contamination_factor=0.1, query_multiple=False): 

    save_path = f"{save_dir}/bs_{batch_size}_{strategy}_seed_{seed}_avp.txt"

    np.random.seed(seed)
    random.seed(seed) 

    model = BAD_IForest().fit(X)
    
    scores0 = model.decision_function(X)
    avp_0 = average_precision_score(y, scores0)
    with open(save_path, "a") as f:
        f.write(f"{avp_0}\n")

    iterations = int(np.ceil(X.shape[0] / batch_size))
    queriable = np.ones(X.shape[0], dtype=bool)

    for _ in range(iterations): 

        if query_multiple: 
            queriable = None        # set queriable to None so get_batch_queries knows to return all samples
        
        batch_idxs = model.get_batch_queries(X, batch_size, strategy=strategy, queriable=queriable, contamination_factor=contamination_factor)
        queriable[batch_idxs] = False
        model.update(X[batch_idxs,:], y[batch_idxs])
    
        scores = model.decision_function(X)
        avp = average_precision_score(y, scores)
        with open(save_path, "a") as f:
            f.write(f"{avp}\n")
        
    

def main(): 
    # seeds = [0, 1, 2, 3, 4]
    seeds = [0]
    datasets = ['wine'] #, 'pima', 'cardio', 'annthyroid']
    batch_sizes = [3] #[1, 2, 5, 10]
    strategies = ['wc', 'avg']
    
    configs = list(itertools.product(seeds, datasets, batch_sizes, strategies))

    for seed, dataset, batch_size, strategy in configs:
        data, labels = odds_datasets.load(dataset)
        contamination_factor = np.sum(labels) / len(labels)
        run_sim(data, labels, batch_size, strategy, seed=seed, query_multiple=False, contamination_factor=contamination_factor)

if __name__ == "__main__":
    main()








