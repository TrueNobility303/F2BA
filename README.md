# F2BA

Codes are adapted from [BOME](https://github.com/Cranial-XIX/BOME), which is published in NeurIPS 2022.

Please download the [dataset](https://drive.google.com/file/d/14deh-F4YlEH1c_s0P5DSliU042QV39K3/view?usp=sharing) and unzip it under the data folder specified in `args.dataset`.

## Single Machine Data Hypercleaning

To reproduce our experiment results, run

```
cd ./single_machine
python -u Run_MNIST.py
python -u Run_Fashion.py
```

To plot the figure, further run
```
python -u plot2fig.py
```
Note that our training logs are also contained in the file `plot2fig.py`.

## Distributed Learable Regularization
To reproduce our experiment results, run
```
cd ./distributed
python -u l2reg.py
```
And you can find the training logs and plot the figure in a similar way.
