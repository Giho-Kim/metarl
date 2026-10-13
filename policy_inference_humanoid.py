"""
Policy inference and visualization for the Humanoid-Dir environment.

This script loads a trained Dreamer policy on the humanoid-dir task,
rolls out one meta-episode, and saves a GIF.  In each frame it draws a
green arrow indicating the target walking direction.

Usage example:

python3 policy_inference_humanoid.py \
    --exp-dir /path/to/experiment \
    --out humanoid_dir.gif
"""

import argparse
import os
import pathlib
import pickle
from contextlib import contextmanager
from typing import Dict, Tuple, Optional, List, Any

import imageio
import cv2  # Used for drawing arrows on RGB frames
import numpy as np
import torch

import envs.wrappers as wrappers
from envs.mujoco.humanoid_dir import HumanoidDirEnv
from dreamer_creme import Dreamer


@contextmanager
def suppress_stdout():
    """간단한 stdout mute용 context manager (MuJoCo 초기화 로그 숨기기)."""
    devnull = open(os.devnull, "w")
    old_stdout = os.sys.stdout
    try:
        os.sys.stdout = devnull
        yield
    finally:
        os.sys.stdout = old_stdout
        devnull.close()


class DummyLogger:
    """Inference 시 Dreamer가 기대하는 최소한의 인터페이스."""

    def get_agent_frames(self) -> int:
        return 0


def get_action_fn(agent: Dreamer):
    """Dreamer 에이전트에서 action 을 꺼내는 함수 생성."""

    def _get_action(
        obs: Dict[str, np.ndarray],
        agent_state: Optional[Tuple] = None,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Tuple]]:
        # Dict of numpy -> batch dimension 추가
        obs_batched = {k: np.array([obs[k]]) for k in obs}
        done_arr = np.array([False], dtype=np.bool_)  # inference 이므로 항상 False

        policy_out, new_state = agent(
            obs_batched, done_arr, agent_state, training=False
        )

        action = {
            k: policy_out[k][0].detach().cpu().numpy() for k in policy_out
        }
        return action, new_state

    return _get_action


def task_to_dir_vec(task: Any) -> Optional[np.ndarray]:
    """
    Humanoid-Dir 의 task 표현을 2D 방향 벡터로 변환.

    - dict 에 들어있는 경우 (예: {"direction": angle} 혹은 vector)
    - 스칼라 각도 (rad)
    - 2D/ND vector (앞의 두 성분 사용)

    반환: shape (2,) 의 unit vector, 실패 시 None
    """
    # dict 인 경우 내부에서 실제 값을 꺼낸다.
    if isinstance(task, dict):
        for key in ["direction", "goal_direction", "goal", "task"]:
            if key in task:
                task = task[key]
                break

    # 이후에는 숫자나 array 라고 가정
    if isinstance(task, (int, float, np.floating)):
        theta = float(task)
        return np.array([np.cos(theta), np.sin(theta)], dtype=float)

    arr = np.array(task, dtype=float).flatten()
    if arr.size == 0:
        return None
    if arr.size == 1:
        theta = float(arr[0])
        return np.array([np.cos(theta), np.sin(theta)], dtype=float)
    else:
        vec = arr[:2]
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
    return None

def load_env_and_agent(
    exp_dir: pathlib.Path,
) -> Tuple[Dreamer, Any, HumanoidDirEnv, int]:
    """실험 디렉토리에서 config, 모델을 읽고 env + agent 생성."""

    # config 로드
    with open(exp_dir / "config.txt", "rb") as f:
        config = pickle.load(f)

    assert (
        config.meta_learning
    ), "HumanoidDir requires meta-learning to be enabled"

    # *** meta-episode을 항상 2개로 설정 ***
    config.num_meta_episodes = 3

    # 기본 MuJoCo env
    with suppress_stdout():
        base_env = HumanoidDirEnv()

    # 학습 때와 동일한 래퍼 적용
    env = base_env
    env = wrappers.NormalizeActions(env)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    env = wrappers.RewardObs(env)
    max_time_steps = config.max_episode_length * config.num_meta_episodes
    env = wrappers.TimeAugmentedState(env, max_time_steps)
    env = wrappers.MetaLearningEnv(
        env, config.num_meta_episodes, config.max_episode_length
    )

    # action dimension 세팅
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # Dreamer 생성 및 가중치 로드
    agent = Dreamer(
        env.observation_space, env.action_space, config, DummyLogger(), None
    ).to(config.device)
    agent.load_state_dict(
        torch.load(exp_dir / "latest_model.pt", map_location=config.device),
        strict=False,
    )
    agent.eval()

    return agent, env, base_env, config.max_episode_length


def run_policy(exp_dir: pathlib.Path, output_gif: pathlib.Path) -> None:
    """한 번의 meta-episode 를 rollout 하며 GIF 생성."""

    agent, env, base_env, max_episode_length = load_env_and_agent(exp_dir)
    get_action = get_action_fn(agent)

    # task (목표 방향) 샘플링 및 reset
    num_task = 5
    logs = np.zeros(3)
    for num in range(num_task):
        imgs_list: List[np.ndarray] = []
        done = False
        episode = 0
        time_s = 0

        task = env.sample_task()

        # *** 태스크가 바뀔 때마다 에이전트 은닉 상태 초기화 ***
        agent_state: Optional[Tuple] = None

        obs = env.reset(task)
        reward_episode = 0.
        while not done:
            action, agent_state = get_action(obs, agent_state)
            obs, reward, done, infos = env.step(action)

            reward_episode += reward
            logs[episode] += reward

            #
            # MuJoCo frame 캡쳐
            frame = base_env.render("rgb_array")  # type: ignore
            arrow_frame = np.copy(frame)

            # 방향 화살표 오버레이
            try:
                dir_vec = task_to_dir_vec(task)

                if dir_vec is not None:
                    h, w, _ = arrow_frame.shape
                    center = (int(w / 2), int(h / 2))
                    scale = 0.3 * min(h, w)  # 화살표 길이

                    end = (
                        int(center[0] + dir_vec[0] * scale),
                        int(center[1] - dir_vec[1] * scale),  # y 축 뒤집기
                    )

                    # BGR = (0,255,0) -> 초록색
                    cv2.arrowedLine(
                        arrow_frame,
                        center,
                        end,
                        color=(0, 255, 0),
                        thickness=3,
                        tipLength=0.2,
                    )
            except Exception:
                arrow_frame = frame

            # === Episode 표시 추가 ===
            episode_str = f"EPISODE: {episode}"
            cv2.putText(
                arrow_frame,
                episode_str,
                org=(20, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(255, 0, 0),  # 빨간색
                thickness=2,
            )

            reward_str = f"REWARD: {reward_episode:.2f}"
            cv2.putText(
                arrow_frame,
                reward_str,
                org=(20, 80),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(255, 0, 0),  # 빨간색
                thickness=2,
            )


            imgs_list.append(arrow_frame)

            time_s += 1
            if time_s % max_episode_length == 0:
                print(time_s)
                episode += 1

        out_path = str(output_gif) + "_" + str(num) + ".mp4"
        height, width, _ = imgs_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(out_path, fourcc, 30, (width, height))  # FPS 30

        for f in imgs_list:
            video.write(f[:, :, ::-1])  # RGB → BGR
        video.release()

        # GIF 저장 (15 FPS)
        # imageio.mimsave(
        #     str(output_gif) + "_" + str(num) + ".gif",
        #     [imageio.core.util.Array(f) for f in imgs_list],
        #     duration=1 / 120,
        #     loop=0,
        # )
        print(f"Saved visualization to {output_gif}")

    # print("Start evaluation(episodes 32000)")
    # print("total training steps : 17,832,130")
    # a = 2465.36/3000.
    # print("eval returns (meta-episodes) : 2465.36(" + f"{100 * a :.2f}" +"%)")
    # print("episode 0 : 1263.11")
    # print("episode 1 : 1202.24")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Policy inference for Humanoid-Dir environment"
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        type=pathlib.Path,
        help="Directory with config.txt and best_model.pt",
    )
    parser.add_argument(
        "--out",
        default="humanoid_dir",
        type=pathlib.Path,
        help="Output GIF filename",
    )
    args = parser.parse_args()
    run_policy(args.exp_dir, args.out)


if __name__ == "__main__":
    main()
