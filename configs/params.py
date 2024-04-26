from typing import List


class Params:
    # Default values specified as class attributes for clarity and ease of modification
    dim: int = 2
    data_size: int = 1000
    k: int = 15
    iters: int = 6
    bounds: List[int] = None
    max_dist_mode: str = "disabled"
    max_dist: float = 1000
    min_cluster: int = 0
    min_cluster_alpha: float = 0
    max_cluster: int = 500
    max_cluster_alpha: float = 0
    fixed: bool = True
    num_clients: int = 2
    delay: int = 0
    dp: str = "none"
    eps: float = 0
    seed: int = 1337
    rho: float = 0.225
    assignment: str = "unconstrained"
    init: str = "optimal"
    dp_release: str = "sumcount"
    sum_post: str = "none"
    count_post: str = "none"
    cent_post: str = "none"

    def __init__(self, **kwargs):
        # Set the bounds default here to ensure mutable default isn't shared
        self.bounds = kwargs.pop('bounds', [-1, 1])

        # Set all other parameters, allowing for overrides via kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid parameter of {self.__class__.__name__}")

        self.attributes = [attr for attr in dir(self) if not callable(attr) and not attr.startswith("__")]

    def __getitem__(self, index):
        return getattr(self, self.attributes[index])
