# More Powerful Conditional Selective Inference for Generalized Lasso by Parametric Programming

This package implements a more powerful and general conditional Selective Inference (SI) approach for generalized lasso by parametric programming. The main idea is to compute the continuum path of the optimal solutions in the direction of the selected test statistic, and identify the subset of the data space corresponding to the hypothesis selection event by following the solution path. The proposed parametric programming-based method not only avoids all the drawbacks of current SI method for generalized lasso but also improves the performance and practicality of SI in various respects.

See the paper <https://arxiv.org/abs/2105.04920> for more details.

A preliminary short version of this work was presented at the AI & Statistics (AISTATS2021) conference in which we only studied a specific case of vanilla lasso.
The conference paper is available at <https://arxiv.org/abs/2004.09749>.

## Installation & Requirements

This package has the following requirements:

- [cvxpy](https://www.cvxpy.org)
- [numpy](http://numpy.org)
- [mpmath](http://mpmath.org/)
- [matplotlib](https://matplotlib.org/)
- [scipy](https://www.scipy.org)
- [statsmodels](https://www.statsmodels.org/)

We recommend installing or updating anaconda to the latest version and use Python 3
(We used Python 3.8.3).

All commands are run from the terminal.

## Examples for fused lasso

To run the following commands, please first access to 'fused_lasso' directory.

#### (1) Checking the uniformity of the pivot
```
>> python ex1_uniform_pivot.py
```

#### (2) Computing p-value
```
>> python ex2_p_value.py
```

## Examples for vanilla lasso

To run the following commands, please first access to 'vanilla_lasso' directory.

#### (1) Checking the uniformity of the pivot
```
>> python ex1_uniform_pivot.py
```

#### (2) Computing p-value
```
>> python ex2_p_value.py
```

#### (3) Computing confidence interval
```
>> python ex3_confidence_interval.py
```

---
## References

[1] V.N.L. Duy and I. Takeuchi. More Powerful Conditional Selective Inference for Generalized Lasso by Parametric Programming. arXiv preprint arXiv:2105.04920, 2021.

[2] V. N. Le Duy and I. Takeuchi. Parametric Programming Approach for More Powerful and General Lasso Selective Inference. In International Conference on Artificial Intelligence and Statistics, pages 901–909. PMLR, 2021.
