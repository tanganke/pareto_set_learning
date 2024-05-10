# Towards Efficient Pareto Set Approximation via Weight-Ensembling Mixture of Experts

## How to reproduce the results:

for CLIP models:

1. Download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw?usp=share_link).
2. The running scripts are placed at `scripts` directory, or you can run `bash clip_pareto_moe.sh <method> <version>`. 
    The method can be `moe_ls`, and `moe_epo`.
    For detailed hyperparameter configuration of different verisons, please refer to the bash script.
    > for LS, EPO, and MGDA, see `clip_mtl.sh`
3. the results would be saved at `results` directory.

for GPT-2 models:

1. Download chekpoints form HuggingFace, fine-tuned checkpoints would be available after double-blind review.
2. The running scripts are placed at `scripts` directory, or you can run `bash gpt2_pareto_moe.sh <method> <version>`. 
   The method can be `moe_ls`, and `moe_epo`.
   For detailed hyperparameter configuration of different verisons, please refer to the bash script.
   > for other baselines, see `gpt2_merge.sh`
3. The results would be saved at `results` directory.
