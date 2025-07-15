export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py   --task_name sim_insertion_scripted   --ckpt_dir   result   --policy_class ACT   --kl_weight   10   --chunk_size 100   --hidden_dim 512   --batch_size 8   --dim_feedforward 3200   --num_steps  40000 --eval_every 1000   --lr         1e-5   --seed       0 
