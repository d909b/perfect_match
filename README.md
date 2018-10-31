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

- Runnable scripts are in the `perfect_match/apps/` subdirectory.
    - `perfect_match/apps/main.py` is the main runnable script for running experiments.
    - The available command line parameters for runnable scripts are described in `perfect_match/apps/parameters.py`
- You can add new baseline methods to the evaluation by subclassing `perfect_match/models/baselines/baseline.py`
    - See e.g. `perfect_match/models/baselines/neural_network.py` for an example of how to implement your own baseline methods.
    - You can register new methods for use from the command line by adding a new entry to the `get_method_name_map` method in `perfect_match/apps/main.py`
- You can add new benchmarks by implementing the benchmark interface, see e.g. `perfect_match/models/benchmarks` for examples of how to add your own benchmark to the benchmark suite.
    - You can register new benchmarks for use from the command line by adding a new entry to the `get_benchmark_name_map` method in `perfect_match/apps/evaluate.py`

#### Requirements and dependencies

- This project was designed for use with Python 2.7. We can not guarantee and have not tested compability with Python 3.
- To run the TCGA and News benchmarks you need to download the SQLite databases containing the raw data samples for these benchmarks (`news.db` and `tcga.db`).
    - Note that you need around 10GB of free disk space to store the databases.
    - Save the database files to the `./data` directory relative to this file in order to be compatible with the step-by-step guides below.
- To run BART, Causal Forests and to reproduce the figures you need to have [R](https://www.r-project.org/) installed. See https://www.r-project.org/ for installation instructions.
    - To run BART, you need to have the R-packages `rJava` and `bartMachine` installed. See https://github.com/kapelner/bartMachine for installation instructions. Note that `rJava` requires a working Java installation as well.
    - To run Causal Forests, you need to have the R-package `grf` installed. See https://github.com/grf-labs/grf for installation instructions.
    - To reproduce the paper's figures, you need to have the R-package `latex2exp` installed. See https://cran.r-project.org/web/packages/latex2exp/vignettes/using-latex2exp.html for installation instructions.
- For the python dependencies, see `setup.py`. You can use `pip install .` to install the python dependencies. Note the installation of `rpy2` will fail if you do not have a working R installation on your system (see above).

#### Reproduction

- Make sure you have all the requirements listed above.
- You can use the script `perfect_match/apps/run_all_experiments.py` to obtain the exact parameters used with `main.py` to reproduce the experimental results in [our paper](https://arxiv.org/abs/1810.00656).
    - Note that we ran several thousand experiments which can take a while if evaluated sequentially. We therefore suggest to run the experiments in parallel using e.g. a compute cluster.
    - Once you have completed the experiments, you can calculate the summary statistics (mean +- standard deviation) over all the repeated runs using the `run_results.sh` script.
    - See below for a step-by-step guide for each reported result.
- You can also reproduce the figures in our manuscript by running the R-scripts in `perfect_match/visualisation/`

##### IHDP Step-by-step

- Navigate to the directory containing this file.
- Create a folder to hold the experimental results `mkdir -p results`.
- Run `python ./perfect_match/apps/run_all_experiments.py ./perfect_match/apps ihdp ./data ./results`
    - The script will print all the command line configurations (3000 in total) you need to run to obtain the experimental results to reproduce the IHDP results.
    - Note that we only evaluate PM, + on X, and + MLP on IHDP. All other results are taken from the respective original authors' manuscripts.
- Run the command line configurations from the previous step in your favorite compute environment.
- After the experiments have concluded, use `run_results.sh` to calculate the summary statistics mean +- standard deviation over all repeated runs.
    - Example 1: `run_results.sh ./results/pm_ihdp2a0k_pbm_mse_1 ihdp`, where `ihdp` indicates that you want results for the IHDP dataset, to get the results for "PM" on IHDP.
    - Example 2: `run_results.sh ./results/pm_ihdp2a0k_pbm_mahal_mse_1 ihdp` to get the results for "+ on X" on IHDP.
    - Example 3: `run_results.sh ./results/pm_ihdp2a0k_pbm_no_tarnet_mse_1 ihdp` to get the results for "+ MLP" on IHDP.

##### Jobs Step-by-step

- Navigate to the directory containing this file.
- Create a folder to hold the experimental results `mkdir -p results`.
- Run `python ./perfect_match/apps/run_all_experiments.py ./perfect_match/apps jobs ./data ./results`
    - The script will print all the command line configurations (40 in total) you need to run to obtain the experimental results to reproduce the Jobs results.
    - Note that we only evaluate PM, + on X, + MLP, PSM on Jobs. All other results are taken from the respective original authors' manuscripts.
- Run the command line configurations from the previous step in your favorite compute environment.
- After the experiments have concluded, use `run_results.sh` to calculate the summary statistics mean +- standard deviation over all repeated runs.
    - Example 1: `run_results.sh ./results/pm_jobs2a0k_pbm_mse_1 jobs`, where `jobs` indicates that you want results for the jobs dataset, to get the results for "PM" on Jobs.
    - Example 2: `run_results.sh ./results/pm_jobs2a0k_pbm_mahal_mse_1 jobs` to get the results for "+ on X" on Jobs.
    - Example 3: `run_results.sh ./results/pm_jobs2a0k_pbm_no_tarnet_mse_1 jobs` to get the results for "+ MLP" on Jobs.
    - Example 4: `run_results.sh ./results/pm_jobs2a0k_psm_mse_1 jobs` to get the results for "PSM" on Jobs.

##### News-2/News-4/News-8/News-16 Step-by-step

- Navigate to the directory containing this file.
- Create a folder to hold the experimental results `mkdir -p results`.
- Run `python ./perfect_match/apps/run_all_experiments.py ./perfect_match/apps news ./data ./results`
    - The script will print all the command line configurations (2400 in total) you need to run to obtain the experimental results to reproduce the News results.
    - Note that we evaluate all listed methods on News-2/News-4/News-8/News-16. No results are taken from the respective original authors' manuscripts.
- Run the command line configurations from the previous step in your favorite compute environment.
- After the experiments have concluded, use `run_results.sh` to calculate the summary statistics mean +- standard deviation over all repeated runs.
    - Example 1: `run_results.sh ./results/pm_news2a10k_pbm_mse_1 news-2`, where `news-2` indicates that you want results for the News-2 dataset, to get the results for "PM" on News-2. Note that the folder path must match exactly with the type of dataset requested (news-2 <> news-2), otherwise the shown summary statistics will not be the right metrics.
    - Example 2: `run_results.sh ./results/pm_news4a10k_pbm_mse_1 news-4` to get the results for "PM" on News-4.
    - Example 3: `run_results.sh ./results/pm_news8a10k_pbm_mse_1 news-8` to get the results for "PM" on News-8.
    - Example 4: `run_results.sh ./results/pm_news16a7k_pbm_mse_1 news-16` to get the results for "PM" on News-16.
    - Repeat for all evaluated method / benchmark combinations.

##### TCGA Step-by-step

- Navigate to the directory containing this file.
- Create a folder to hold the experimental results `mkdir -p results`.
- Run `python ./perfect_match/apps/run_all_experiments.py ./perfect_match/apps tcga ./data ./results`
    - The script will print all the command line configurations (180 in total) you need to run to obtain the experimental results to reproduce the TCGA results.
- Run the command line configurations from the previous step in your favorite compute environment.
- After the experiments have concluded, use `run_results.sh` to calculate the summary statistics mean +- standard deviation over all repeated runs.
    - Example 1: `run_results.sh ./results/pm_tcga8a10k18478f_pbm_mse_1 tcga`, where `tcga` indicates that you want results for the TCGA dataset, to get the results for "PM" with 10% hidden confounding on TCGA.
    - Example 2: `run_results.sh ./results/pm_tcga8a10k16425f_pbm_mse_1 tcga` to get the results for "PM" with 20% hidden confounding on TCGA.
    - Example 3: `run_results.sh ./results/pm_tcga8a10k14372f_pbm_mse_1 tcga` to get the results for "PM" with 30% hidden confounding on TCGA.
    - Repeat for all evaluated method / degree of hidden confounding combinations.