
# Stochastic F2BA on  LLM Data-Cleaning

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
