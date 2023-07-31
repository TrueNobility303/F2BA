# F2BA

Codes are adapted from [BOME](https://github.com/Cranial-XIX/BOME), which is published in NeurIPS 2022.

## Single-Machine Data Hypercleaning

To reproduce our experiment results, please download the dataset `fashion_data_cleaning.pt` and `mnist_data_cleaning.pt`  at [this link](https://drive.google.com/file/d/14deh-F4YlEH1c_s0P5DSliU042QV39K3/view?usp=sharing) and put it under the data folder specified in `args.dataset` in `single_machine/data_cleaning.py`. Then run

```
cd ./single_machine
python -u Run_MNIST.py
python -u Run_Fashion.py
```

To plot the figure, further run
```
python -u plot2fig.py
```
Note that our training logs are also stored as arrays in the file `plot2fig.py`.

## Distributed Learable Regularization
To reproduce our experiment results, please download the dataset `l2reg.pt` at [this link](https://drive.google.com/file/d/14deh-F4YlEH1c_s0P5DSliU042QV39K3/view?usp=sharing) and put it under the data folder specified in `args.dataset` in `distributed/l2reg.py`. Then run
```
cd ./distributed
python -u l2reg.py
```
And you can find the training logs and plot the figure in a similar way.
