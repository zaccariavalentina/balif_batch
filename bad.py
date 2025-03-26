from functools import partial
from typing import Literal, Optional, NamedTuple
from jaxtyping import Float, Int, Shaped

import numpy as np
import copy

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import beta as betaJSP
import equinox as eqx

from pyod.models.base import BaseDetector


class BayesianDetector(BaseDetector):
    @property
    def regions_score(self) -> Float[np.ndarray, "estimators regions"]:
        raise NotImplementedError

    def estimators_apply(
        self, X: Float[np.ndarray, "samples features"]
    ) -> Int[np.ndarray, "samples estimators"]:
        raise NotImplementedError

    def __init__(
        self,
        *args,
        prior_sample_size=0.1,
        aggregation_method="arithmetic_mean",
        reprocess_decision_scores=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_sample_size = prior_sample_size
        self.aggregation_method = aggregation_method
        self.reprocess_decision_scores = reprocess_decision_scores

    def fit(
        self,
        X: Float[np.ndarray, "samples features"],
        y: Optional[Int[np.ndarray, "samples 1"]] = None,
    ):
        super().fit(X, y)
        self.beliefs = EnsembleBeliefs.from_scores(
            regions_score=jnp.asarray(self.regions_score),
            contamination=self.contamination,
            prior_sample_size=self.prior_sample_size,
        )
        if self.reprocess_decision_scores:
            self.decision_scores_ = self.decision_function(X)
            self._process_decision_scores()
        return self

    def decision_function(
        self, X: Float[np.ndarray, "samples features"]
    ) -> Float[np.ndarray, "samples"]:
        regions = jnp.asarray(self.estimators_apply(X))
        scores = self.beliefs.aggregate(regions, self.aggregation_method)
        return np.asarray(scores)

    def update(
        self,
        X: Float[np.ndarray, "samples features"],
        y: Int[np.ndarray, "samples 1"],
        confidence: float | Float[np.ndarray, "#samples"] = 1.0,
    ):
        regions = jnp.asarray(self.estimators_apply(X))
        da = jnp.asarray(confidence * (y >= 1)).flatten()
        db = jnp.asarray(confidence * (y == 0)).flatten()
        self.beliefs = self.beliefs.update(regions, da, db)

    def acquisition_value(self, X: Float[np.ndarray, "samples features"]) -> Float: 
        samples_regions = jnp.asarray(self.estimators_apply(X))                                 # shape (samples, estimators)
        alphas_global, betas_global = self.beliefs.aggregate_as_distribution(samples_regions)     # shape (samples,) or (samples, 2**k) when batch_querying

        # check the number of dimensions of alphas_global and betas_globals
        # if different from 1 (i.e., 2) iterate over the second dimension
        if alphas_global.ndim == 1: 
            modes = jnp.empty_like(alphas_global)
            for i in range(len(alphas_global)):
                mode_sample = BetaDistr.mode(alphas_global[i], betas_global[i])
                modes = modes.at[i].set(mode_sample)
            log_margin = betaJSP.logpdf(modes, alphas_global, betas_global) - betaJSP.logpdf(jnp.full_like(modes, 0.5), alphas_global, betas_global)
            return jnp.exp(-log_margin)
        elif alphas_global.ndim == 2:
            interests = jnp.empty_like(alphas_global)       # shape (samples, 2**k)
            for j in range(alphas_global.shape[1]): 
                modes = jnp.empty(alphas_global.shape[0])
                for i in range(alphas_global.shape[0]):
                    a_i_j_global = alphas_global[i, j]
                    b_i_j_global = betas_global[i, j]
                    mode_sample = BetaDistr.mode(a_i_j_global, b_i_j_global)
                    modes = modes.at[i].set(mode_sample)
                log_margin = betaJSP.logpdf(modes, alphas_global[:, j], betas_global[:, j]) - betaJSP.logpdf(jnp.full_like(modes, 0.5), alphas_global[:, j], betas_global[:, j])
                interests = interests.at[:, j].set(jnp.exp(-log_margin))
            return interests 
        else:
            raise ValueError(f"Unknown shape for alphas_global and betas_global: {alphas_global.shape}")

    def get_batch_queries(self, X: Float[np.ndarray, "samples features"], batch_size: int = 1, strategy:str = 'wc', queriable:np.ndarray = None, contamination_factor:float=None ) -> Float:
        """
        Return indices of samples to query in batch. 
        """
        queriable_copy = copy.deepcopy(queriable)
        def merge_superposition(model_superpos1, model_superpos2): 
            alphas1, betas1 = model_superpos1.beliefs.a, model_superpos1.beliefs.b
            alphas2, betas2 = model_superpos2.beliefs.a, model_superpos2.beliefs.b
            new_alphas = jnp.concatenate([alphas1, alphas2], axis=-1)
            new_betas = jnp.concatenate([betas1, betas2], axis=-1)
            model_superpos1.beliefs = EnsembleBeliefs(a=new_alphas, b=new_betas)
            return model_superpos1
        
        queries_idx = []
        model_superpos = copy.deepcopy(self)
        new_alphas,new_betas = model_superpos.beliefs.a[..., jnp.newaxis], model_superpos.beliefs.b[..., jnp.newaxis]
        model_superpos.beliefs = EnsembleBeliefs(a=new_alphas, b=new_betas)

        for i in range(batch_size): 
            interest = model_superpos.acquisition_value(X)
            if strategy == 'wc': 
                interest = interest.min(axis=-1)
            elif strategy == 'avg':
                if i == 0: 
                    weights = np.ones_like(interest)
                else:
                    weights = np.concatenate([weights*contamination_factor, weights*(1-contamination_factor)], axis=-1)
                interest = jnp.sum(interest * weights, axis=-1) / jnp.sum(weights, axis=-1)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if queriable_copy is not None: 
                query_idx = jnp.where(queriable_copy, interest, -np.inf).argmax()
                queriable_copy[query_idx] = False
            else:
                query_idx = jnp.argmax(interest)

            queries_idx.append(query_idx)

            model_superpos1 = copy.deepcopy(model_superpos)
            model_superpos1.update(X[query_idx,:].reshape(1,-1), 1)
            model_superpos2 = copy.deepcopy(model_superpos)
            model_superpos2.update(X[query_idx,:].reshape(1,-1), 0)
            model_superpos = merge_superposition(model_superpos1, model_superpos2)
        
        return queries_idx


class BetaDistr(eqx.Module):
    a: Float[jax.Array, "..."]
    b: Float[jax.Array, "..."]

    def mean(self):
        return self.a / (self.a + self.b)
    
    @staticmethod
    def mode(a, b):
        return jax.lax.select(jnp.minimum(a, b) > 1, (a -1)/(a+b-2), jax.lax.select(a>b, 1.0, 0.0))
    


class EnsembleBeliefs(BetaDistr):
    a: Float[jax.Array, "estimators regions"]
    b: Float[jax.Array, "estimators regions"]

    @classmethod
    @eqx.filter_jit
    def from_scores(
        cls,
        regions_score: Float[jax.Array, "estimators regions"],
        contamination: float = 0.1,
        prior_sample_size: float = 0.1,
    ):
        # flat prior, matching the contamination
        prior_a = contamination * prior_sample_size
        prior_b = (1 - contamination) * prior_sample_size

        # add positive obs matching the mean to detector scores
        regions_score = jnp.clip(regions_score, 0.01, 0.99)
        a_over_b = regions_score / (1 - regions_score)
        a = jnp.maximum(a_over_b * prior_b, prior_a)
        b = jnp.maximum(prior_a / a_over_b, prior_b)
        return cls(a=a, b=b)

    @eqx.filter_jit
    def update(
        self,
        samples_regions: Int[jax.Array, "samples estimators"],
        da: Float[jax.Array, "samples"],
        db: Float[jax.Array, "samples"],
    ):
        def single_update(beliefs, region, da, db):
            new_a = beliefs.a.at[region].add(da)
            new_b = beliefs.b.at[region].add(db)
            return eqx.tree_at(lambda t: (t.a, t.b), beliefs, (new_a, new_b))

        def scan_fn(beliefs, update_info):
            update_all = jax.vmap(single_update, in_axes=(0, 0, None, None))
            return update_all(beliefs, *update_info), None

        self, _ = jax.lax.scan(scan_fn, self, (samples_regions, da, db))
        return self

    @eqx.filter_jit
    def gather(
        self, samples_regions: Int[jax.Array, "samples estimators"]
    ) -> Shaped[BetaDistr, "samples estimators"]:
        def take(distr, idx):
            return BetaDistr(a=distr.a[idx], b=distr.b[idx])

        take = jax.vmap(take, in_axes=(0, 0))  # map over estimators
        take = jax.vmap(take, in_axes=(None, 0))  # map over samples
        return take(self, samples_regions)

    @eqx.filter_jit
    def aggregate(
        self, samples_regions: Int[jax.Array, "samples estimators"], method: str
    ) -> Float[jax.Array, "samples"]:
        beliefs = self.gather(samples_regions)

        if method == "arithmetic_mean":
            return jnp.mean(beliefs.mean(), axis=-1)
        elif method == "geometric_mean":
            return np.exp(np.mean(np.log(beliefs.mean()), axis=-1))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    @eqx.filter_jit
    def aggregate_distribution(
        self, samples_regions: Int[jax.Array, "samples estimators"]
    ) -> Shaped[Float, "samples"]:
        dist = self.gather(samples_regions)
        gathered_a, gathered_b = dist.a, dist.b     # shape (samples, estimators) or (samples, estimators, 2**k)
        global_a = jnp.sum(gathered_a, axis=1)
        global_b = jnp.sum(gathered_b, axis=1)      # sum over estimators
        return global_a, global_b

    @eqx.filter_jit
    def aggregate_as_distribution(self, samples_regions: Int[jax.Array, "samples estimators"]) -> Shaped[Float, "samples 2**k"]:
        dist = self.gather(samples_regions)
        gathered_a, gathered_b = dist.a, dist.b     # shape (samples, estimators) or (samples, estimators, 2**k)

        global_mean = jnp.mean(gathered_a / (gathered_a + gathered_b), axis=1)      # the global mean is the mean of estimators, shape (samples, ) or (samples, 2**k)
        global_sample_size = jnp.sum(gathered_a + gathered_b, axis=1)               # the global sample size is the sum of the sample sizes of estimators, shape (samples, ) or (samples, 2**k)
        global_a = global_mean * global_sample_size
        global_b = global_sample_size - global_a
        return global_a, global_b