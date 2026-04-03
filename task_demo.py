from isaacsim import SimulationApp
import sys

simulation_app = SimulationApp({"headless": "--headless" in sys.argv})

# ── SimulationApp 已创建，现在可以安全 import Isaac Sim 模块 ──────────────────
import argparse
import os
import sys

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path

# ── 本地模块（均为纯 Python，可在 SimulationApp 初始化后安全 import）─────────
from asset_config import SCENE_MAP, ROBOT_CONFIG, REAL_ROOM, REAL_ROBOT_START   # 场景 & 机器人资产配置
from tasks import Task, ApproachTask               # 任务抽象体系
from camera_utils import (setup_cameras, setup_corner_cameras, apply_real_intrinsics,
                           save_frame, print_camera_intrinsics, make_episode_videos)

# ── 命令行参数 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Load a scene and Spot robot.")
parser.add_argument(
    "--scene",
    type=str,
    default="grid",
    choices=list(SCENE_MAP.keys()),
    help=f"Scene to load. Options: {list(SCENE_MAP.keys())}",
)
parser.add_argument(
    "--target",
    type=float,
    nargs=2,
    default=[2.0, 2.0],
    metavar=("X", "Y"),
    help="Target position for Spot to navigate to, e.g. --target 5.0 3.0 (default: 5.0 0.0)",
)
parser.add_argument(
    "--record-fps",
    type=float,
    default=25,
    metavar="FPS",
    help="Camera recording FPS (default: 25). Must be <= render FPS (25Hz).",
)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE_SAVE_DIRS = {
    "laptop": os.path.join(_SCRIPT_DIR, "collected_data"),
    "server": "/isaac-sim/.local/share/ov/data/collected_data",
}
parser.add_argument(
    "--device",
    type=str,
    default="laptop",
    choices=["laptop", "server"],
    help="Device preset: 'laptop' saves to <script_dir>/collected_data, 'server' saves to the mounted data volume.",
)
parser.add_argument(
    "--save-dir",
    type=str,
    default=None,
    metavar="PATH",
    help="Override save directory. If not set, uses --device preset.",
)
parser.add_argument(
    "--quality",
    type=str,
    default="balance",
    choices=["high", "balance"],
    help=(
        "Data quality mode (default: balance).\n"
        "  high    : RGB → .png (lossless) + Depth → .npy (float32)\n"
        "  balance : RGB → .jpg (lossy, ~10x smaller) + Depth → .npz (compressed float32)"
    ),
)
parser.add_argument(
    "--real",
    action="store_true",
    help=(
        "Use real room corner camera layout: "
        f"4m × 3m × 2.44m, cameras at ceiling corners. "
        "Applies real RealSense intrinsics (fx=605.91, fy=605.54)."
    ),
)
args, _ = parser.parse_known_args()

# ── 全局配置 ──────────────────────────────────────────────────────────────────
TARGET_POS    = np.array(args.target)   # [x, y] 目标坐标
ARRIVAL_DIST  = 0.4                     # 到达判定半径（米）
RENDER_FPS    = 200 / 8                 # 25.0 Hz（rendering_dt = 8/200）
CAPTURE_EVERY = max(1, round(RENDER_FPS / args.record_fps))
SAVE_DIR      = args.save_dir if args.save_dir is not None else _DEVICE_SAVE_DIRS[args.device]

_rgb_fmt   = "PNG (lossless)" if args.quality == "high" else "JPG quality=90 (lossy)"
_depth_fmt = "NPY (float32)"  if args.quality == "high" else "NPZ compressed (float32)"
print(f"[task_demo] Loading scene    : '{args.scene}'")
print(f"[task_demo] Target position  : {TARGET_POS}")
print(f"[task_demo] Record FPS       : {args.record_fps} Hz  (capture every {CAPTURE_EVERY} render frames)")
print(f"[task_demo] Quality mode     : {args.quality}  |  RGB → {_rgb_fmt}  |  Depth → {_depth_fmt}")

# ── 当前任务（替换此处即可切换任务类型）──────────────────────────────────────
# 示例：current_task = TouchWallTask(wall_x=8.0)
current_task: Task = ApproachTask(target_pos=TARGET_POS, arrival_dist=ARRIVAL_DIST)
print(f"[task_demo] Task             : {current_task.description()}")

# ── 资产根路径 ────────────────────────────────────────────────────────────────
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit(1)

# ── 全局运行状态 ──────────────────────────────────────────────────────────────
first_step          = True
episode             = 0
callback_registered = False
frame_counter       = 0
record_counter      = 0    # 站稳后才从 0 开始计数，用于录像文件命名
is_settled          = False


# ── 物理回调：初始化机器人 + 每步驱动 ────────────────────────────────────────
def on_physics_step(step_size) -> None:
    global first_step, is_settled
    if first_step:
        spot.initialize()
        first_step = False
        is_settled = False
    elif not is_settled:
        spot.forward(step_size, np.zeros(3))
        # 获取质心速度，线速度 < 0.05 m/s 认为已站稳
        pos, quat = spot.robot.get_world_pose()
        vel = spot.robot.get_linear_velocity()   # [vx, vy, vz]
        speed = float(np.linalg.norm(vel))
        if speed < 0.05:
            is_settled = True
            print(f"[task_demo] Spot settled (speed={speed:.3f} m/s). Starting task...")
    else:
        spot.forward(step_size, base_command)


def start_simulation():
    """首次启动仿真并注册 physics callback。"""
    global first_step, base_command, callback_registered, is_settled
    my_world.reset()
    my_world.play()
    my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)
    callback_registered = True
    first_step          = True
    base_command        = np.zeros(3)
    is_settled          = False


# ── 创建世界 ──────────────────────────────────────────────────────────────────
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)

# ── 加载场景 ──────────────────────────────────────────────────────────────────
scene_usd_path = assets_root_path + SCENE_MAP[args.scene]
ground_prim = define_prim("/World/Ground", "Xform")
ground_prim.GetReferences().AddReference(scene_usd_path)
print(f"[task_demo] Scene loaded     : {scene_usd_path}")

# ── 任务场景初始化（加载目标标记球等任务专属资产）────────────────────────────
current_task.setup(my_world, assets_root_path)

# ── 创建相机（来自 camera_utils.py）──────────────────────────────────────────
if args.real:
    camera_view = setup_corner_cameras(
        room_length = REAL_ROOM["length"],
        room_width  = REAL_ROOM["width"],
        room_height = REAL_ROOM["height"],
    )
    print(f"[task_demo] Camera layout    : real room corners "
          f"({REAL_ROOM['length']}m × {REAL_ROOM['width']}m × {REAL_ROOM['height']}m)")
else:
    camera_view = setup_cameras(cam_r=7.9, cam_h=3, orientation_mode="pitch", pitch_deg=-31.0)
    print(f"[task_demo] Camera layout    : simulation (r=7.9m, h=3m, pitch=-31°)")

# ── 数据保存目录 ──────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"[task_demo] Data save dir    : {SAVE_DIR}")

# ── 加载 Spot 机器人（参数来自 asset_config.py）──────────────────────────────
_spot_cfg  = ROBOT_CONFIG["spot"]
_spawn_pos = REAL_ROBOT_START if args.real else _spot_cfg["position"]
spot = SpotFlatTerrainPolicy(
    prim_path = _spot_cfg["prim_path"],
    name      = _spot_cfg["name"],
    position  = np.array(_spawn_pos),
)
print(f"[task_demo] Spot spawned at {_spawn_pos}")

# ── 初始化世界 + 注册回调 ─────────────────────────────────────────────────────
start_simulation()
base_command = np.zeros(3)

# ── 主循环 ────────────────────────────────────────────────────────────────────
print(f"[task_demo] Starting simulation ...")

_intrinsics_printed = False   # 确保只在第一帧后打印/保存一次

while simulation_app.is_running():
    my_world.step(render=True)
    frame_counter += 1

    # ── 第 2 帧后：写入相机内参（rep prim 此时已写入 Stage）────────────────────
    if not _intrinsics_printed and frame_counter >= 2:
        if args.real:
            apply_real_intrinsics()   # 将真实 RealSense 内参写入 USD Stage
        print_camera_intrinsics(
            camera_view,
            resolution_w  = 640,
            resolution_h  = 480,
            save_dir      = SAVE_DIR,
        )
        _intrinsics_printed = True

    # GUI 按下停止键时退出
    if my_world.is_stopped():
        print("[task_demo] Simulation stopped by user.")
        break

    if my_world.is_playing() and not first_step and is_settled:
        record_counter += 1   # 站稳后才递增，从 1 开始（第 1 帧为 f000001）
        pos, quat = spot.robot.get_world_pose()

        # ── 任务判定 & 指令生成（通过 current_task 统一调度）─────────────────
        if current_task.is_done(pos, quat):
            # 任务完成 → 直接结束，不再重置循环
            print(f"[task_demo] Task '{current_task.description()}' finished "
                  f"at ({pos[0]:.2f}, {pos[1]:.2f}). Closing simulation...")
            break
        else:
            base_command = current_task.get_command(pos, quat)

        # ── 相机数据采集（来自 camera_utils.py）──────────────────────────────
        save_frame(
            camera_view   = camera_view,
            episode       = episode,
            frame_counter = record_counter,   # 用录像专用计数器，从 0 起编号
            save_dir      = SAVE_DIR,
            quality       = args.quality,
            capture_every = CAPTURE_EVERY,
        )

# ── 仿真结束后：把每个相机的 RGB 序列合成为视频 ──────────────────────────────
print("[task_demo] Generating episode videos ...")
make_episode_videos(
    save_dir  = SAVE_DIR,
    episode   = episode,
    fps       = args.record_fps,
    quality   = args.quality,
)

simulation_app.close()
