#!/bin/sh

# The root folder for which to calculate the summary measures for a given model,
# e.g. /some/directory/pm_news16a7k_cfrnet_mse_1
# The root folder must contain the individual run data (./run_$i/run.txt)
FOLDER_PATH="$1"

# Flag indicating what kind of evaluation measures were used.
# Three options:
#  (1) IS_BINARY="true" = two treatment options,
#                         the outputs are \sqrt{\eps_{PEHE}} mean +- std and \eps_{ATE} mean +- std,
#                         used for News-2, IHDP
#  (2) IS_BINARY="false" = more than two treatment options,
#                          the outputs are \sqrt{\eps_{mPEHE}} mean +- std and \eps_{mATE} mean +- std,
#                          used for News-4/8/16
#  (3) IS_BINARY="jobs" = no counterfactual ground truth available,
#                         the outputs are R_{Pol}(\pi) mean +- std and \eps_{ATT} mean +- std,
#                         used for the Jobs dataset
IS_BINARY="$2"

export PYTHONPATH="./perfect_match/:$PYTHONPATH"
# The following command merges all run files for repeated runs (run_$i/run.txt)
# into a single summary file ($FOLDER_PATH/summary.txt).
python ./perfect_match/apps/main.py --dataset=./ --with_rnaseq --do_train --do_hyperopt --num_hyperopt_runs=10 --do_evaluate --fraction_of_data_set=1.00 --num_units=16 --num_layers=2 --seed=909 --num_epochs=100 --learning_rate=0.001 --dropout=0.0 --batch_size=4 --do_merge_lsf --l2_weight=0.000 --imbalance_loss_weight=0.0 --benchmark=jobs --method=nn --early_stopping_patience=7 --do_not_save_predictions --validation_set_fraction=0.24 --test_set_fraction=0.2 --with_propensity_batch --early_stopping_on_pehe --experiment_index=27 --num_treatments=2 --output_directory=$FOLDER_PATH > /dev/null 2>&1

wait

count=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | wc -l)

if [ "$IS_BINARY" = "ihdp" ] || [ "$IS_BINARY" = "news-2" ]
then
m1=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $20}' | sed 's/.$//' | awk '{total += $1} END {print total/NR}')
s1=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $20}' | sed 's/.$//' |awk '{x+=$0;y+=$0^2}END{print sqrt(y/NR-(x/NR)^2)}')
m2=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $16}' | sed 's/.$//' | awk '{total += $1} END {print total/NR}')
s2=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $16}' | sed 's/.$//' |awk '{x+=$0;y+=$0^2}END{print sqrt(y/NR-(x/NR)^2)}')
elif [ "$IS_BINARY" = "jobs" ]
then
m1=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $4}' | sed 's/.$//' | awk '{total += $1} END {print total/NR}')
s1=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $4}' | sed 's/.$//' |awk '{x+=$0;y+=$0^2}END{print sqrt(y/NR-(x/NR)^2)}')
m2=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $34}' | sed 's/.$//' | awk '{total += $1} END {print total/NR}')
s2=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $34}' | sed 's/.$//' |awk '{x+=$0;y+=$0^2}END{print sqrt(y/NR-(x/NR)^2)}')
else
m1=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $26}' | sed 's/.$//' | awk '{total += $1} END {print total/NR}')
s1=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $26}' | sed 's/.$//' |awk '{x+=$0;y+=$0^2}END{print sqrt(y/NR-(x/NR)^2)}')
m2=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $20}' | sed 's/.$//' | awk '{total += $1} END {print total/NR}')
s2=$(cat $FOLDER_PATH/summary.txt | grep Best_test_score  | awk '{print $20}' | sed 's/.$//' |awk '{x+=$0;y+=$0^2}END{print sqrt(y/NR-(x/NR)^2)}')
fi

m1=$(printf "%.2f" $m1)
s1=$(printf "%.2f" $s1)
m2=$(printf "%.2f" $m2)
s2=$(printf "%.2f" $s2)

# Print summary statistics mean \pm standard deviation.
echo "{$m1} \$\\pm\$ $s1 & {$m2} \$\\pm\$ $s2"