# Medium-Sized Experiments on "Learn-to-Regularize"

The code is based on [BOME](https://github.com/Cranial-XIX/BOME/). 
## Quick Start 

1. Prepare the data.  Please download the [dataset](https://drive.google.com/file/d/14deh-F4YlEH1c_s0P5DSliU042QV39K3/) and unzip it.
For your convenience, we have trained the baseline method without regularization and saved them into `save_l2reg` folder.

2. Compare different bilevel optimization algorithms. 

```
python -u l2reg.py --alg {%algorithm} {%other arguments}
```
where `%algorithm` can be chosen in `['AID_CG', 'AID_FP', 'ITD', 'reverse', 'BOME', 'F2BA', 'AccF2BA', 'PBGD', 'stocBiO', 'VRBO', 'MRBO', 'F2SA', 'F2SA_p']`, and `%other arguments` pass the required hyper-parameters such as learning rates and momentum.
The results after running will be saved in the `save_l2reg` folder.

## Main Results

Medium-sized problems are typically highly smooth. For these problems, our papers suggest the following techniques to improve the performance to better exploit smoothness.

### Deterministic case: use Nesterov acceleration in both upper and lower levels.

Run AccF2BA+ [2] with the command
```
python -u l2reg.py --alg AccF2BA --x_momentum 0.5
```
You may need to tune the momentum parameter for different problems. For example, in our problem, we find `--x_momentum 0.9` is not good while  `--x_momentum 0.5` performs well. 
The same situation holds for the `AccF2BA` method.

### Stochastic case: use high-order finite difference.

Run F2SA-p [1] with the command
```
python -u l2reg.py --alg F2SA_p --w_lr 0.1 --p {%p}
```
In our experiments, we compare p in $\\{2,3,5,8,10\\}$.  The lowel-level probelms are solved in parallel to maximize the utilization of computing resources.

## Reference 

* [1] **Lesi Chen**, Junru Li, El Mahdi Chayti, and Jingzhao Zhang, _Faster Gradient Methods for Highly-smooth Stochastic Bilevel Optimization_ [[ICLR 2026]](https://arxiv.org/abs/2509.02937)
* [2] **Lesi Chen** and Jingzhao Zhang. _On the Condition Number Dependency in Bilevel Optimization_ [[arXiv 2025]](https://arxiv.org/abs/2511.22331)


