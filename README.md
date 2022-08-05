# SpatioTemporalObjectTracking
Graph translation network for modeling spatio-temporal dynamics of household objects for our paper 'Proactive Robot Assistance via Spatio-Temporal Object Modeling'. The model reads in an input graph representing the environment and time, and translates it to a probabilistic output graph representing the environment at the next time step.

<img src="GNNarchitecture.png"
     alt="GNN Architecture"
     style="float: center;" />

### Running this model
To run the model on the existing dataset, you can use the `run.py` with the path to the dataset and config file. e.g. `python3 ./run.py --path=$dataset`. To run a batch using existing configuration with baselines and ablations, use `runRoutines.sh`. 

A processed version of the [HOMER dataset]() used for the results is present in the `data/` directory of this repository, and can be used directly with the model using the above commands. In order to use a different dataset generated using HOMER, first copy the dataset into `data/` directory, and then run `prepRoutines.sh` to run the necessary pre-processing. This needs to be done only once.

If you're curious about the code itself:
- The model and it's helper functions can be found in `GraphTranslatorModule.py`
- The `reader.py` file contains code to process the (HOMER) dataset
- The evaluation functions for our model are in `breakdown_evaluations.py`

### Change Notes
The states are read in the reader.py and passed on as a part of the node feature (class, category, state1, state2,... stateN). '0' means state is absent/irrelevant and the rest are indices. The model is changed accordingly to embed roles and values of states before passing them into a transformer. The evaluation is kept as was before. If we ever pursue this, we'll need to change the evaluation so that transitions like moving cereal bowl while it is still full are penalized.


### Citation
```
Anonymous
```