
# Large-Scale F2BA on  LLM Data-Cleaning

The code is based on [ScaleBiO](https://github.com/2003pro/ScaleBiO/).

## Quick Start 

1. Install the required packages.
```
pip install -r requirements.txt
```

2. Prepare the data. We provide an example dataset in the `data_alpaca_0.5` folder. You can also use your own dataset in the same format. For instance, you can modify the `data_alpaca_0.5/process.py` file to change the corruption rate.

3. Login your [wandb](https://wandb.ai/) account and run the experiemnts.

```
bash ./run_F2SA.sh
```
## Takeaway

Although the penalty method is a widely adopted approach for large-scale bilevel optimization and has appeared in the recent works [1–3], our study highlights the critical role of the **two-time-scale stepsize (${\rm lr}_y \ll {\rm lr}_x$)**. This novel design enables us, for the first time, to derive near-optimal complexity guarantees that all prior studies [1–3] using single-time-scale stepsize (${\rm lr}_y = {\rm lr}_x$) failed to achieve. 


We also corroborate this on the GPT-2 data-cleaning task: In the `./example` folder we export `F2SA.csv` from previously finished runs via wandb and plot the curves shown below.

![](https://github.com/TrueNobility303/F2BA/blob/main/LLM/example/GPT2_demo.png)

where $w_{\rm noisy}$  denotes the learned weight of noisy data after training, which is expected to converge to $0$. 

Reproduce the figure via

```
cd example && python plot_demo.py
```

[1] Shen H, Xiao Q, Chen T. On penalty-based bilevel gradient descent method. Mathematical Programming, 2025: 1-51.

[2] Kwon J, Kwon D, Wright S, et al. A fully first-order method for stochastic bilevel optimization. In ICML, 2023.

[3] Liu B, Ye M, Wright S, et al. Bome! bilevel optimization made easy: A simple first-order approach. In NeurIPS, 2022. 


## Citations 

```
@inproceedings{chen2024finding,
  title={On finding small hyper-gradients in bilevel optimization: Hardness results and improved analysis},
  author={Chen, Lesi and Xu, Jing and Zhang, Jingzhao},
  booktitle={COLT},
  year={2024}
}

@article{chen2025near,
  title={Near-optimal nonconvex-strongly-convex bilevel optimization with fully first-order oracles},
  author={Chen, Lesi and Ma, Yaohua and Zhang, Jingzhao},
  journal={JMLR},
  pages={1--56},
  year={2025}
}
```
