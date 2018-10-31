## Perfect Match: A Simple Method for Learning Representations For Counterfactual Inference With Neural Networks

Perfect Match (PM) is a method for learning to estimate individual treatment effect (ITE) using neural networks. PM is easy to implement,
compatible with any architecture, does not add computational complexity or hyperparameters, and extends to any number of treatments. This repository contains the source code used to evaluate PM and most of the existing state-of-the-art methods at the time of publication of [our manuscript](https://arxiv.org/abs/1810.00656) (Oct 2018). PM and the presented experiments are described in detail in [our paper](https://arxiv.org/abs/1810.00656). Since we performed one of the most comprehensive evaluations to date with four different datasets with varying characteristics, this repository may serve as a benchmark suite for developing your own methods for estimating causal effects using machine learning methods. In particular, the source code is designed to be easily extensible with (1) new methods and (2) new benchmark datasets.

Author(s): Patrick Schwab, ETH Zurich <patrick.schwab@hest.ethz.ch>, Lorenz Linhardt, ETH Zurich <llorenz@student.ethz.ch> and Walter Karlen <walter.karlen@hest.ethz.ch>

License: MIT, see LICENSE.txt

#### Citation

If you reference or use our methodology, code or results in your work, please consider citing:

    @article{schwab2018perfect,
      title={{Perfect Match: A Simple Method for Learning Representations For Counterfactual Inference With Neural Networks}},
      author={Schwab, Patrick and Linhardt, Lorenz and Karlen, Walter},
      journal={arXiv preprint arXiv:1810.00656},
      year={2018}
    }

#### Usage:

- Runnable scripts are in the perfect_match/apps/ subdirectory.
    - perfect_match/apps/main.py is the main runnable script for running experiments.
    - The available command line parameters for runnable scripts are described in perfect_match/apps/parameters.py
- You can add new baseline methods to the evaluation by subclassing perfect_match/models/baselines/baseline.py
    - See e.g. perfect_match/models/baselines/neural_network.py for an example of how to implement your own baseline methods.
    - You can register new methods for use from the command line by adding a new entry to the get_method_name_map method in perfect_match/apps/main.py
- You can add new benchmarks by implementing the benchmark interface, see e.g. perfect_match/models/benchmarks for examples of how to add your own benchmark to the benchmark suite.
    - You can register new benchmarks for use from the command line by adding a new entry to the get_benchmark_name_map method in perfect_match/apps/evaluate.py

#### Reproduction

- You can use the script perfect_match/apps/run_all_experiments.py to obtain the exact parameters used with main.py to reproduce the experimental results in [our paper](https://arxiv.org/abs/1810.00656).
    - Note that we ran several thousand experiments which can take a while if evaluated sequentially. We therefore suggest to run the experiments in parallel using e.g. a compute cluster.
    - Once you have completed the experiments, you can calculate the summary statistics (mean +- standard deviation) over all the repeated runs using the run_results.sh script.
    - See below for a step-by-step guide for each reported result.
- You can reproduce the figures in our manuscript by running the R-scripts in perfect_match/visualisation/

##### IHDP

- Run perfect_match/apps/run_all_experiments.py