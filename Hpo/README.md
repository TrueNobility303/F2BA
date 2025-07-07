# Deterministic F2BA on "Learn-to-Regularize"

The code is based on [BOME](https://github.com/Cranial-XIX/BOME/).

## Quick Start 

1. Prepare the data.  Please download the [dataset](https://drive.google.com/file/d/14deh-F4YlEH1c_s0P5DSliU042QV39K3/) and unzip it under the hpo folder.
For your convenience, we have trained the baseline method without regularization and saved them into `save_l2reg` folder.

2. Compare different bilevel optimization algorithms. 

```
python -u l2reg.py --alg {%algorithm} {%other arguments}
```
where `%algorthm` can be chosen in `['AID_CG', 'AID_FP', 'ITD', 'reverse', 'BOME', 'F2BA', 'AccF2BA', 'PBGD', 'stocBiO', 'VRBO', 'MRBO', 'F2SA', 'F2SA_2']`, and `%other arguments` pass the required hyper-parameters such as learning rates and momentum.
The results after running will be saved at the `save_l2reg` folder.

## Effectness of Fully First-Order Approximation

We suggest **always** use the fully first-order methods rather than Hessian-vector-product (HVP) methods in both the deterministic and stochastic cases.

For the deterministic case, you can compare the results by running
```
python -u l2reg.py --alg F2BA
python -u l2reg.py --alg AID_FP
```
where `F2BA` is our method and `AID_FP` is the best HVP methods we found in all the baselines.

For the stochastic case, you can compare the results by running 
```
python -u l2reg.py --alg F2SA
python -u l2reg.py --alg stocBiO
```

## Further Improvemens on Highly-Smooth Problems

We provide some tricks to improve the performance for highly-smooth problems. 

Note: This part is still under construction.

### Deterministic Case

**Sugestion 1. Use Nesterov accleration in the upper level.**

Switch from `--alg F2BA` to `--alg AccF2BA` and compare their performances. 
```
python -u l2reg.py --alg F2BA
python -u l2reg.py --alg AccF2BA
```
**Suggestion 2. Use Nesterov acceleration in the lower level.**

We have turned on `nesterov=True` in the lower level optimizer if `args.x_momentum != 0.0`, so you only need to compare the result with and without using momentum. 
```
python -u l2reg.py --alg F2BA
python -u l2reg.py --alg F2BA --x_momentum 0.5
```
You may need to tune the momentum parameter for different problems. For example, in our problem we find `--x_momentum 0.9` is not good while  `--x_momentum 0.5` performs well. 
The same situation holds for the `AccF2BA` method.

Finally, you are expected to reproduce the folowing figure by running `python plot_l2reg_det.py`.

![](https://github.com/TrueNobility303/F2BA/blob/main/Hpo/save_l2reg/l2reg_det.png)

In the above figure, the methods with `+` use Nesterov momentum in the lower level.

### Stochastic Case

**Suggestion 1. Normalize the gradient in the upper level.**

Switch from gradient descent to normalized gradient descent. 

* Previous update: $x_{t+1} = x_t - \eta_t g_t$.

* Suggested update: $x_{t+1} = x_t - \eta_t g_t / \Vert g_t \Vert$.

**Suggestion 2. Use symmetric penalty reformulation.**

According to our analysis, it corresponds to using central difference instead of forward difference in the fully first-order approximation for the hyper-gradient.

* Previous reformulation: solve $\min_{x,y} \max_z f(x,y) + \lambda (g(x,y) - g(x,z))$.

* Suggested reformulation: solve $\min_{x,y} \max_z (f(x,y) + f(x,z))/2 + \lambda (g(x,y) - g(x,z))$.

We have implemented the improved method that jointly use the above two tricks in `F2BA_2`. You can see the performance boost by running
```
python -u l2reg.py --alg F2SA
python -u l2reg.py --alg F2SA_2 --w_lr 1.0
```
Note that the optimal learning rates for `F2SA_2` and `F2SA` are different so you have to tuned both of them in a new problem. 

**Suggestion 3. Never use variance reduction.**

In our experiments, we find variance reduction techniques are useless and can even harm the performace. 
It aligns with the conclusion in [BOME](https://github.com/Cranial-XIX/BOME/).
You can run the methos `MRBO` or `VRBO` to see the performances of this class of methods.

When all the runs are compated, you can visualize the results by running `python plot_l2reg_stoc.py`. You are expected to reproduce the following figure.

![](https://github.com/TrueNobility303/F2BA/blob/main/Hpo/save_l2reg/l2reg_stoc.png)


