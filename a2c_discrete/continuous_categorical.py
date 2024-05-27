import torch
from torch import nan
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits

__all__ = ["ContinuousCategorical"]

class ContinuousCategorical(Distribution):
    r"""
    Creates a continuous categorical distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both), and a support tensor defining the values of each category.

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
        continuous_support (Tensor): values corresponding to each categorical index
    """
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector, "continuous_support": constraints.real}
    has_enumerate_support = True

    def __init__(self, probs=None, logits=None, continuous_support=None, validate_args=None):
        self._probs_per_dim = continuous_support.size(-1)
        probs = probs.reshape(-1, self._probs_per_dim) if probs is not None else None
        logits = logits.reshape(-1, self._probs_per_dim) if logits is not None else None

        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        
        if continuous_support is None:
            raise ValueError("`continuous_support` must be specified.")
        #if continuous_support.dim() != 2 or continuous_support.size(0) != (probs.size(-1) if probs is not None else logits.size(-1)):
        #    raise ValueError("`continuous_support` must be a 2-dimensional tensor with size equal to the number of categories.")
        
        self._param = self.probs if probs is not None else self.logits
        self.continuous_support = continuous_support.to(self._param.device)
        self._num_events = self.continuous_support.size(-2)
        batch_shape = self._param.size()[:-2] if self._param.ndimension() > 2 else torch.Size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ContinuousCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = new.logits
        new.continuous_support = self.continuous_support
        new._num_events = self._num_events
        super(ContinuousCategorical, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self._num_events - 1)

    @lazy_property
    def continuous_support(self):
        return self.continuous_support
    
    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)

    @property
    def param_shape(self):
        return self._param.size()

    @property
    def mean(self):
        vals = self.probs * self.continuous_support
        vals = vals.sum(-1)
        return self.probs.argmax(-1), vals
        # return torch.full(
        #     self._extended_shape(),
        #     nan,
        #     dtype=self.probs.dtype,
        #     device=self.probs.device,
        # )

    @property
    def mode(self):
        prob_index = self.probs.reshape(-1, self._probs_per_dim).argmax(-1)
        return prob_index, self.continuous_support.gather(-1, prob_index.unsqueeze(-1)).squeeze(-1)

    @property
    def variance(self):
        return torch.full(
            self._extended_shape(),
            nan,
            dtype=self.probs.dtype,
            device=self.probs.device,
        )
    
    def sample_noise(self, target_tensors, low=-0.02, high=0.02):
        # Define the linear decaying PDF and its CDF
        def pdf(x):
            return (high - torch.abs(x)) / (high - low)

        def cdf(x):
            return torch.where(
                x < 0,
                (x - low) * (high + x) / ((high - low) * (high + low)),
                1 - (high - x) * (high - x) / ((high - low) * (high + low))
            )

        def inverse_cdf(u):
            return torch.where(
                u < 0.5,
                low + torch.sqrt(u * (high - low) * (high + low)),
                high - torch.sqrt((1 - u) * (high - low) * (high + low))
            )

        # Sample from a uniform distribution and apply the inverse CDF
        uniform_samples = torch.rand_like(target_tensors)  # Uniformly distributed samples in [0, 1)
        noise_samples = inverse_cdf(uniform_samples)

        return noise_samples
    
    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._probs_per_dim)
        indices_2d = torch.multinomial(probs_2d, sample_shape.numel(), True)
        #indices_2d = torch.multinomial(probs_3d, sample_shape.numel(), True).T
        samples_2d = torch.gather(self.continuous_support, -1, indices_2d)
        indices_2d = indices_2d.squeeze(-1)
        samples_2d = samples_2d.squeeze(-1)
        noise = self.sample_noise(samples_2d)
        #noise = torch.rand_like(samples_2d) * 0.02 - 0.01
        samples_2d = samples_2d + noise
        #indices_2d = indices_2d.reshape(self._extended_shape(sample_shape))
       # samples_2d = samples_2d.reshape(self._extended_shape(sample_shape))
        return indices_2d, samples_2d

    def log_prob(self, indices, values):
        if self._validate_args:
            self._validate_sample(indices)
        indices = indices.long().reshape(-1).unsqueeze(-1)
        logits_2d = self.logits.reshape(-1, self._probs_per_dim)
        logits = logits_2d.gather(-1, indices).squeeze(-1)
        logits = logits.reshape_as(values)
        logits = logits.sum(-1)
        return logits
#        value, log_pmf = torch.broadcast_tensors(indices, self.logits)
#        value = value[..., :1]
#        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    def enumerate_support(self, expand=True):
        num_events = self._num_events
        values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values

# Example usage
# probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
# support = torch.tensor([1.5, 2.5, 3.5, 4.5])
# dist = ContinuousCategorical(probs=probs, continuous_support=support)
# indices, samples = dist.sample((5,))
# print("Indices:", indices)
# print("Samples:", samples)
