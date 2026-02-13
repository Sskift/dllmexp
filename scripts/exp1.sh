export HF_HOME=/data/zhangyt/contrastive/hf_cache
export HF_DATASETS_CACHE=/data/zhangyt/contrastive/hf_cache
export LD_LIBRARY_PATH=/home/zhangyt/miniconda3/envs/contrastive/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=8

python eval.py --model_alias llada --task humaneval --alg low_confidence --tokens_per_step 1
python eval.py --model_alias llada --task mbpp --alg low_confidence --tokens_per_step 1
python eval.py --model_alias llada --task gsm8k --alg low_confidence --tokens_per_step 1
python eval.py --model_alias llada --task truthfulqa --alg low_confidence --tokens_per_step 1

python eval.py --model_alias llada --task humaneval --alg low_confidence --tokens_per_step 2
python eval.py --model_alias llada --task mbpp --alg low_confidence --tokens_per_step 2
python eval.py --model_alias llada --task gsm8k --alg low_confidence --tokens_per_step 2
python eval.py --model_alias llada --task truthfulqa --alg low_confidence --tokens_per_step 2

python eval.py --model_alias llada --task humaneval --alg low_confidence --tokens_per_step 4
python eval.py --model_alias llada --task mbpp --alg low_confidence --tokens_per_step 4
python eval.py --model_alias llada --task gsm8k --alg low_confidence --tokens_per_step 4
python eval.py --model_alias llada --task truthfulqa --alg low_confidence --tokens_per_step 4

python eval.py --model_alias llada1.5 --task humaneval --alg low_confidence --tokens_per_step 1
python eval.py --model_alias llada1.5 --task mbpp --alg low_confidence --tokens_per_step 1
python eval.py --model_alias llada1.5 --task gsm8k --alg low_confidence --tokens_per_step 1
python eval.py --model_alias llada1.5 --task truthfulqa --alg low_confidence --tokens_per_step 1

python eval.py --model_alias llada1.5 --task humaneval --alg low_confidence --tokens_per_step 2
python eval.py --model_alias llada1.5 --task mbpp --alg low_confidence --tokens_per_step 2
python eval.py --model_alias llada1.5 --task gsm8k --alg low_confidence --tokens_per_step 2
python eval.py --model_alias llada1.5 --task truthfulqa --alg low_confidence --tokens_per_step 2

python eval.py --model_alias llada1.5 --task humaneval --alg low_confidence --tokens_per_step 4
python eval.py --model_alias llada1.5 --task mbpp --alg low_confidence --tokens_per_step 4
python eval.py --model_alias llada1.5 --task gsm8k --alg low_confidence --tokens_per_step 4
python eval.py --model_alias llada1.5 --task truthfulqa --alg low_confidence --tokens_per_step 4

python eval.py --model_alias dream --task humaneval --alg maskgit_plus --tokens_per_step 1
python eval.py --model_alias dream --task mbpp --alg maskgit_plus --tokens_per_step 1
python eval.py --model_alias dream --task gsm8k --alg maskgit_plus --tokens_per_step 1
python eval.py --model_alias dream --task truthfulqa --alg maskgit_plus --tokens_per_step 1

python eval.py --model_alias dream --task humaneval --alg maskgit_plus --tokens_per_step 2
python eval.py --model_alias dream --task mbpp --alg maskgit_plus --tokens_per_step 2
python eval.py --model_alias dream --task gsm8k --alg maskgit_plus --tokens_per_step 2
python eval.py --model_alias dream --task truthfulqa --alg maskgit_plus --tokens_per_step 2

python eval.py --model_alias dream --task humaneval --alg maskgit_plus --tokens_per_step 4
python eval.py --model_alias dream --task mbpp --alg maskgit_plus --tokens_per_step 4
python eval.py --model_alias dream --task gsm8k --alg maskgit_plus --tokens_per_step 4
python eval.py --model_alias dream --task truthfulqa --alg maskgit_plus --tokens_per_step 4

python eval.py --model_alias diffucoder --task humaneval --alg maskgit_plus --tokens_per_step 1
python eval.py --model_alias diffucoder --task mbpp --alg maskgit_plus --tokens_per_step 1
python eval.py --model_alias diffucoder --task gsm8k --alg maskgit_plus --tokens_per_step 1
python eval.py --model_alias diffucoder --task truthfulqa --alg maskgit_plus --tokens_per_step 1

python eval.py --model_alias diffucoder --task humaneval --alg maskgit_plus --tokens_per_step 2
python eval.py --model_alias diffucoder --task mbpp --alg maskgit_plus --tokens_per_step 2
python eval.py --model_alias diffucoder --task gsm8k --alg maskgit_plus --tokens_per_step 2
python eval.py --model_alias diffucoder --task truthfulqa --alg maskgit_plus --tokens_per_step 2

python eval.py --model_alias diffucoder --task humaneval --alg maskgit_plus --tokens_per_step 4
python eval.py --model_alias diffucoder --task mbpp --alg maskgit_plus --tokens_per_step 4
python eval.py --model_alias diffucoder --task gsm8k --alg maskgit_plus --tokens_per_step 4
python eval.py --model_alias diffucoder --task truthfulqa --alg maskgit_plus --tokens_per_step 4