# pretrain model:
CUDA_VISIBLE_DEVICES=1 python 1_pretrain_model.py --batch_size 60000 &&

# # ADMM + pruning: test (given by Ding!) -libn
# CUDA_VISIBLE_DEVICES=0 python 2_admm_pruning.py --batch_size 60000 --admm --sparsity_type filter --combine_progressive --epochs 5 --optmzr adam --rho 0.001 --rho_num 4 --lr 0.001 --lr_decay 20 --config_file config_v2 &&
# CUDA_VISIBLE_DEVICES=0 python 2_admm_pruning.py --batch_size 60000 --masked_retrain --sparsity_type filter --combine_progressive --epochs 5 --rho 0.001 --optmzr adam --rho_num 4 --lr 0.001 --lr_decay 20 --config_file config_v2

# Test ADMM+pruning: based on irregular pruning! -libn
CUDA_VISIBLE_DEVICES=1 python 2_admm_pruning.py --batch_size 30000 --verbose --admm --sparsity_type irregular --combine_progressive --epochs 50 --optmzr adam --rho 0.001 --rho_num 4 --lr 0.001 --lr_decay 20 --config_file config_v2 &&
CUDA_VISIBLE_DEVICES=1 python 2_admm_pruning.py --batch_size 60000 --verbose --masked_retrain --sparsity_type irregular --combine_progressive --epochs 50 --rho 0.001 --optmzr adam --rho_num 4 --lr 0.001 --lr_decay 20 --config_file config_v2

