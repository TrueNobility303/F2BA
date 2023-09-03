# F2BA

Codes are adapted from [BOME](https://github.com/Cranial-XIX/BOME), which is published in NeurIPS 2022.

## Single-Machine Data Hypercleaning

To reproduce our experiment results, first generate the data by running

```
python -u data_cleaning.py --pretrain 0
```

Then run
```
cd ./single_machine
python -u Run_MNIST.py
python -u Run_Fashion.py
```

To plot the figure, further run
```
python -u plot2fig.py
```


## Distributed Learable Regularization
To reproduce our experiment results, please download the dataset `l2reg.pt` at [this link](https://drive.google.com/file/d/14deh-F4YlEH1c_s0P5DSliU042QV39K3/view?usp=sharing) and put it under the data folder specified in `args.data_path` in `distributed/l2reg.py`.

You can also generate `l2reg.pt` by setting `args.pretrain=True` in `distributed/l2reg.py`.

Then run

```
cd ./distributed
python -u l2reg.py --alg F2BA
python -u l2reg.py --alg AID
python -u l2reg.py --alg ITD
```

And you can find the training logs in the folder `args.log`.


To plot the figure, further run
```
python -u plot2fig.py
```
Note that our training logs are also stored as arrays in the file `plot2fig.py`.
You should change it to your own logs to plot the figure.
