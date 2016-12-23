# cyclingRNN

Repo for the "cycling RNN" project.

`cycling_rnn.py` contains most of the project functions.
  - `run_rnn` builds and trains the tensorflow graph
  - `get_generalized_curvature` numerically calculates the (generalized) curvature of an n-dimensional trajectory, specified by an array of shape `(t,n)` where `t` is the number of time points. Returns curvatures and frenet frames. See [curvature of space curves](https://en.wikipedia.org/wiki/Curvature#Curvature_of_space_curves).

`cycling_rnn_wrapper.py` and `cycling_rnn_wrapper_yeti.py` are for specifying hyperparameters and calling `run_rnn` (the latter is for running on Columbia's Yeti cluster, see `rnn_job.sh`).

`cycling_rnn_plotter.py`: after a particular run (set of hyperparameters) is finished, use this script to output `.pdf`s in the same directory that show summary analyses.

-----------

Data (not publicly available) is collected from monkeys performing a motor task. Data consists of motor cortex single electrode recordings and muscle recordings (EMG) from the arm. We fit an RNN to the EMG and analyze geometric properties of the model, which we compare to the M1 data. This project is part of an in-progress paper submission.
