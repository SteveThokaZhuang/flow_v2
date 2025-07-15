# debug info
## imitate epiosdes
1. print(f"action_std: {stats['action_std']}")
        print(f"action_mean: {stats['action_mean']}")
2. print(f"[DEBUG] target_qpos min={target_qpos.min()}, max={target_qpos.max()}, mean={target_qpos.mean()}")
3. print(f"Step reward: {ts.reward}, done: {ts.last()}, obs_qpos: {ts.observation['qpos']}")
4. print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        print(f"Step {t}:")
        print(f"  raw_action: {raw_action}")
        print(f"  action (post-processed): {action}")
        print(f"  target_qpos: {target_qpos}")
        print(f"  base_action: {base_action}")
        print(f"  ts.reward: {ts.reward}")
        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
    print(f"=== Rollout {rollout_id} Summary ===")
    print(f"  episode_return: {episode_return}")
    print(f"  episode_highest_reward: {episode_highest_reward}")
    print(f"  env_max_reward: {env_max_reward}")
    print(f"  Success (highest_reward == env_max_reward): {episode_highest_reward == env_max_reward}")
    print("=======================================")
## detrvae
1. print feat and pos