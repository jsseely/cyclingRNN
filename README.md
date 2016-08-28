# cyclingRNN

Repo for the "cycling RNN" project.

`cycling_rnn.py` contains most of the project functions, including `run_rnn` which builds and trains the tensorflow graph.

`cycling_rnn_wrapper.py` and `cycling_rnn_wrapper_yeti.py` are for specifying hyperparameters and calling `run_rnn` (the latter is for running on Columbia's Yeti cluster, see `rnn_job.sh``).

`cycling_rnn_plotter.py`: after a particular run (set of hyperparameters) is finished, use this script to output `.pdf`s in the same directory that show summary analyses.
