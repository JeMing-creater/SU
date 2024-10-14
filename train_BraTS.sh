export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=3
torchrun \
  --nproc_per_node 1 \
  --master_port 29511 \
  train_BraTS2021.py