# cyclingRNN

![image](https://cloud.githubusercontent.com/assets/7425776/22088969/eda6d7fe-ddb5-11e6-8327-cd6e9d5dfc7d.png)

Repo for the "cycling RNN" project. Work in progress.

`cycling_rnn.py` contains most of the project functions.
  - `run_rnn` builds and trains the tensorflow graph
  - `get_generalized_curvature` numerically calculates the (generalized) curvature of an n-dimensional trajectory, specified by an array of shape `(t,n)` where `t` is the number of time points. Returns curvatures and frenet frames. See [curvature of space curves](https://en.wikipedia.org/wiki/Curvature#Curvature_of_space_curves).

`cycling_rnn_wrapper.py` and `cycling_rnn_wrapper_yeti.py` are for specifying hyperparameters and calling `run_rnn` (the latter is for running on Columbia's Yeti cluster, see `rnn_job.sh`).

`cycling_rnn_plotter.py`: after a particular run (set of hyperparameters) is finished, use this script to output `.pdf`s in the same directory that show summary analyses.

-----------

## Geometric analyses

![image](https://cloud.githubusercontent.com/assets/7425776/22089543/d24447ae-ddb9-11e6-8368-d17a19085779.png)

`get_generalized_curvature` estimates the osculating circle (blue) and Frenet frame (black) at a point of an n-dimensional trajectory (green) from possibly noisy samples (blue) by locally fitting an order d polynomial (red) and calculating explicitly from there.

-----------

## Data

Data (not publicly available) is collected from monkeys performing a motor task. Data consists of motor cortex single electrode recordings and muscle recordings (EMG) from the arm. We fit an RNN to the EMG and analyze geometric properties of the model, which we compare to the M1 data. This project is a work in progress.
