from .benchmark import auc_from_pvalues, evaluate, run_benchmark
from .de import (
    METHODS,
    benjamini_hochberg,
    differential_expression,
    estimate_null_fraction,
    log_cpm,
)
from .ebayes import fit_f_dist, moderated_t_test, squeeze_var, trigamma_inverse
from .normalize import normalize, size_factors
from .simulate import simulate_counts

__all__ = [
    "METHODS",
    "auc_from_pvalues",
    "estimate_null_fraction",
    "fit_f_dist",
    "log_cpm",
    "moderated_t_test",
    "squeeze_var",
    "trigamma_inverse",
    "benjamini_hochberg",
    "differential_expression",
    "evaluate",
    "normalize",
    "run_benchmark",
    "simulate_counts",
    "size_factors",
]
__version__ = "0.2.0"
