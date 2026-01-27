# Medium-Sized Experiments on "Learn-to-Regularize"

The code is based on [BOME](https://github.com/Cranial-XIX/BOME/).

## Quick Start 

1. Prepare the data.  Please download the [dataset](https://drive.google.com/file/d/14deh-F4YlEH1c_s0P5DSliU042QV39K3/) and unzip it.
For your convenience, we have trained the baseline method without regularization and saved them into `save_l2reg` folder.

2. Compare different bilevel optimization algorithms. 

```
python -u l2reg.py --alg {%algorithm} {%other arguments}
```
where `%algorthm` can be chosen in `['AID_CG', 'AID_FP', 'ITD', 'reverse', 'BOME', 'F2BA', 'AccF2BA', 'PBGD', 'stocBiO', 'VRBO', 'MRBO', 'F2SA', 'F2SA_p']`, and `%other arguments` pass the required hyper-parameters such as learning rates and momentum.
The results after running will be saved in the `save_l2reg` folder.
```

## Further Improvements on Highly-Smooth Problems

We provide some tricks to improve the performance for highly-smooth problems (still under construction).

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
You may need to tune the momentum parameter for different problems. For example, in our problem, we find `--x_momentum 0.9` is not good while  `--x_momentum 0.5` performs well. 
The same situation holds for the `AccF2BA` method.

In the above figure, the methods with `+` use Nesterov momentum in the lower level.

### Stochastic Case

**Suggestion 1. Normalize the gradient in the upper level.**

Switch from gradient descent to normalized gradient descent, which may sometimes be useful in experiments.

* Previous update: $x_{t+1} = x_t - \eta_t g_t$.

* Suggested update: $x_{t+1} = x_t - \eta_t g_t / \Vert g_t \Vert$.

**Suggestion 2. Use symmetric penalty reformulation.**

In practice, this modification may not being many benifits since the problem remains very similar. But it leads to a very nice theoretical guarantee, which can improves the complexity from $\tilde{\mathcal{O}}(\epsilon^{-6})$ and $\tilde{\mathcal{O}}(\epsilon^{-5})$.

* Previous reformulation: solve $\min_{x,y} \max_z f(x,y) + \lambda (g(x,y) - g(x,z))$.

* Suggested reformulation: solve $\min_{x,y} \max_z (f(x,y) + f(x,z))/2 + \lambda (g(x,y) - g(x,z))$.

We have implemented the improved method that jointly uses the above two tricks in `F2SA_2`. You can see the performance boost by running
```
python -u l2reg.py --alg F2SA
python -u l2reg.py --alg F2SA_2 --w_lr 1.0
```
Note that the optimal learning rates for `F2SA_2` and `F2SA` are different so you have to tuned both of them in a new problem. 



**Suggestion 3. Do not use variance reduction.**

In our experiments, we find variance reduction techniques are useless and can even harm the performace. 
It aligns with the findings in [BOME](https://github.com/Cranial-XIX/BOME/).
You can run the methos `MRBO` or `VRBO` to see the performances of this class of methods.

When all the runs are compated, you can visualize the results by running `python plot_l2reg_stoc.py`. You are expected to reproduce the following figure.

![](https://github.com/TrueNobility303/Highly-Smooth-F2BA/blob/main/Hpo/save_l2reg/l2reg_stoc.png)


