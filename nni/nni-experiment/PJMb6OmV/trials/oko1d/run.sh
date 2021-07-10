#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='PJMb6OmV'
export NNI_SYS_DIR='/home/lab402-3090/Desktop/An/NAS-nni/nni/nni-experiment/PJMb6OmV/trials/oko1d'
export NNI_TRIAL_JOB_ID='oko1d'
export NNI_OUTPUT_DIR='/home/lab402-3090/Desktop/An/NAS-nni/nni/nni-experiment/PJMb6OmV/trials/oko1d'
export NNI_TRIAL_SEQ_ID='0'
export NNI_CODE_DIR='/home/lab402-3090/Desktop/An/NAS-nni/nni'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python3 nni_searcher.py 1>/home/lab402-3090/Desktop/An/NAS-nni/nni/nni-experiment/PJMb6OmV/trials/oko1d/stdout 2>/home/lab402-3090/Desktop/An/NAS-nni/nni/nni-experiment/PJMb6OmV/trials/oko1d/stderr
echo $? `date +%s%3N` >'/home/lab402-3090/Desktop/An/NAS-nni/nni/nni-experiment/PJMb6OmV/trials/oko1d/.nni/state'