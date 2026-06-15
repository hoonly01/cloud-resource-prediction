#!/bin/bash
#SBATCH -J RAG-DB-Build
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p batch_ce_ugrad
#SBATCH --time=00:30:00
#SBATCH -w moana-y4
#SBATCH -o /data/hoonly01/cloud-resource-pred/logs/%x-%j.out

source /data/hoonly01/anaconda3/etc/profile.d/conda.sh
conda activate timellm2

cd /data/hoonly01/cloud-resource-pred
mkdir -p logs

python build_rag_db.py \
  --processed_dir /local_datasets/gcluster \
  --out_dir rag_db

echo "=== Done ==="
