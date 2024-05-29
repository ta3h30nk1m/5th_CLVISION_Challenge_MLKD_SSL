DEVICE=$1
SCENARIO_NUM=$2
RUN_NAME=$3

NOHUP=$4 # NOHUP OR empty

if [ "$NOHUP" == "nohup" ]; then
        CUDA_VISIBLE_DEVICES=$DEVICE nohup python train.py --config scenario_${SCENARIO_NUM}.pkl --run_name $RUN_NAME > ./nohup/${RUN_NAME}_sc${SCENARIO_NUM}.out
else
        CUDA_VISIBLE_DEVICES=$DEVICE python train.py --config scenario_${SCENARIO_NUM}.pkl --run_name $RUN_NAME
fi