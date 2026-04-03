# camera_utils.py
# ── 相机创建 & 数据采集工具 ───────────────────────────────────────────────────
# ⚠️  重要：本文件顶层不导入任何 Isaac Sim / omni / rep 模块。
#     所有 Isaac Sim 相关 import 均在函数内部（懒加载），
#     确保 SimulationApp 在这些模块被加载之前已经完成初始化。
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
from PIL import Image


def _compute_look_at(cam_pos, pitch_deg: float):
    """根据相机位置和俯仰角计算 look_at 目标点。

    水平方位角固定为从相机位置指向原点的方向；
    俯仰角 pitch_deg 控制向下倾斜程度（负值 = 俯视，如 -31°）。
    目标点沿该方向延伸 20 m，供 rep.create.camera(look_at=...) 使用。
    """
    import math
    x, y, z = cam_pos
    # 水平方位角：从相机位置朝向原点
    azim = math.atan2(-y, -x) if (abs(x) > 1e-9 or abs(y) > 1e-9) else 0.0
    pitch = math.radians(pitch_deg)
    fwd = (
        math.cos(pitch) * math.cos(azim),
        math.cos(pitch) * math.sin(azim),
        math.sin(pitch),
    )
    dist = 20.0
    return (x + fwd[0]*dist, y + fwd[1]*dist, z + fwd[2]*dist)


def setup_cameras(
    cam_r: float = 6.0,
    cam_h: float = 2.5,
    orientation_mode: str = "look_at",
    pitch_deg: float = -31.0,
):
    """在场景中创建 4 个环绕固定相机。

    Args:
        cam_r           : 相机到原点的水平距离（米），默认 6.0
        cam_h           : 相机高度（米），默认 2.5
        orientation_mode: 朝向模式，可选值：
                            "look_at" — 所有相机直接看向世界原点 (0,0,0)（默认）
                            "pitch"   — 保持水平方位指向原点，俯仰角由 pitch_deg 控制
        pitch_deg       : 俯仰角（度），仅 orientation_mode="pitch" 时生效，默认 -31°
                          负值 = 俯视，0° = 水平，-90° = 垂直向下

    Returns:
        camera_view: CameraView 实例，可用于后续读取 RGB / Depth 数据
    """
    # ⚠️ 懒加载：在函数内部才 import，保证 SimulationApp 已初始化
    import omni.replicator.core as rep
    from isaacsim.sensors.camera import CameraView

    # 4 个相机位置
    cam_positions = [
        ( cam_r,  0,     cam_h),   # Cam0  +X
        (-cam_r,  0,     cam_h),   # Cam1  -X
        ( 0,      cam_r, cam_h),   # Cam2  +Y
        ( 0,     -cam_r, cam_h),   # Cam3  -Y
    ]

    if orientation_mode == "pitch":
        for pos in cam_positions:
            tgt = _compute_look_at(pos, pitch_deg)
            rep.create.camera(position=pos, look_at=tgt)
        print(f"[camera_utils] 4 cameras created (r={cam_r}m, h={cam_h}m, "
              f"mode=pitch, pitch={pitch_deg}°)")
    else:  # "look_at"
        for pos in cam_positions:
            rep.create.camera(position=pos, look_at=(0, 0, 0))
        print(f"[camera_utils] 4 cameras created (r={cam_r}m, h={cam_h}m, "
              f"mode=look_at → origin)")

    camera_view = CameraView(
        name="fixed_cameras",
        camera_resolution=(640, 480),                        # (width, height)
        prim_paths_expr="/Replicator/Camera_Xform*/Camera",  # 匹配上面 4 个 rep 相机
        output_annotators=["rgb", "depth"],
    )
    return camera_view


def save_frame(
    camera_view,
    episode: int,
    frame_counter: int,
    save_dir: str,
    quality: str,
    capture_every: int,
    save_depth: bool = False,   # 测试阶段可设为 False 跳过 depth 保存
) -> None:
    """从 camera_view 读取当前帧并保存到磁盘。

    每 capture_every 渲染帧调用一次（内部自动跳过不需要保存的帧）。

    保存目录结构：
        <save_dir>/
          episode_0000/
            cam0/
              rgb/    f000025.png / .jpg   shape (480, 640, 3)  uint8
              depth/  f000025.npy / .npz   shape (480, 640)     float32，单位米
            cam1/ ...

    Args:
        camera_view   : setup_cameras() 返回的 CameraView 实例
        episode       : 当前 episode 编号（用于目录命名）
        frame_counter : 当前渲染帧计数（从 1 开始累加）
        save_dir      : 数据根目录
        quality       : "high"（PNG + NPY）或 "balance"（JPG + NPZ）
        capture_every : 每隔几帧采集一次
    """
    if frame_counter % capture_every != 0:
        return

    step_id     = f"f{frame_counter:06d}"
    rgb_tiled   = camera_view.get_rgb_tiled(device="cpu")   # (H_total, W_total, 3)
    depth_tiled = camera_view.get_depth_tiled(device="cpu") # (H_total, W_total, 1)

    # ── 首帧打印 debug 信息 ───────────────────────────────────────────────────
    if frame_counter == capture_every:
        print(f"[DEBUG] rgb_tiled.shape         = {rgb_tiled.shape}")
        print(f"[DEBUG] depth_tiled.shape       = {depth_tiled.shape}")
        print(f"[DEBUG] camera_view.tiled_resolution  = {camera_view.tiled_resolution}")
        print(f"[DEBUG] camera_view.camera_resolution = {camera_view.camera_resolution}")
        print(f"[DEBUG] rgb_tiled  dtype={rgb_tiled.dtype}  "
              f"min={rgb_tiled.min():.3f}  max={rgb_tiled.max():.3f}")

    # ── 自适应转 uint8：max > 1 → 已是 0-255，直接转；否则 *255 ───────────────
    if rgb_tiled.max() > 1.0:
        rgb_uint8 = np.clip(rgb_tiled, 0, 255).astype(np.uint8)
    else:
        rgb_uint8 = (rgb_tiled * 255).astype(np.uint8)

    depth_arr    = depth_tiled[:, :, 0]
    cam_W, cam_H = camera_view.camera_resolution
    d0, d1       = rgb_uint8.shape[0], rgb_uint8.shape[1]

    # ── 若 tiled 图是 (W_total, H_total) 布局则 transpose ────────────────────
    if d0 % cam_W == 0 and d0 % cam_H != 0:
        rgb_uint8 = rgb_uint8.transpose(1, 0, 2)
        depth_arr = depth_arr.T
        d0, d1    = d1, d0
        if frame_counter == capture_every:
            print("[DEBUG] tiled: applied transpose (W,H) → (H,W)")

    H_total, W_total = d0, d1
    n_cols = W_total // cam_W
    n_rows = H_total // cam_H
    if frame_counter == capture_every:
        print(f"[DEBUG] tiled grid: {n_rows}×{n_cols},  cam=(H={cam_H}, W={cam_W})")

    # ── 按相机裁切并保存 ──────────────────────────────────────────────────────
    for cam_id in range(4):
        row = cam_id // n_cols
        col = cam_id %  n_cols
        r0, r1 = row * cam_H, (row + 1) * cam_H
        c0, c1 = col * cam_W, (col + 1) * cam_W

        rgb_cam = rgb_uint8[r0:r1, c0:c1]   # (cam_H, cam_W, 3)

        rgb_dir = os.path.join(save_dir, f"episode_{episode:04d}", f"cam{cam_id}", "rgb")
        os.makedirs(rgb_dir, exist_ok=True)

        if quality == "high":
            Image.fromarray(rgb_cam).save(os.path.join(rgb_dir, f"{step_id}.png"))
        else:  # balance
            Image.fromarray(rgb_cam).save(os.path.join(rgb_dir, f"{step_id}.jpg"), quality=90)

        if save_depth:
            depth_cam = depth_arr[r0:r1, c0:c1]   # (cam_H, cam_W)
            depth_dir = os.path.join(save_dir, f"episode_{episode:04d}", f"cam{cam_id}", "depth")
            os.makedirs(depth_dir, exist_ok=True)
            if quality == "high":
                np.save(            os.path.join(depth_dir, f"{step_id}.npy"), depth_cam)
            else:
                np.savez_compressed(os.path.join(depth_dir, f"{step_id}.npz"), depth=depth_cam)






def make_episode_videos(
    save_dir: str,
    episode: int,
    fps: float = 25.0,
    quality: str = "balance",
    n_cameras: int = 4,
) -> None:
    """把某 episode 每个相机的 RGB 图片序列合成为 MP4 视频。

    视频保存在 rgb 文件夹同级：
        <save_dir>/episode_XXXX/camY/camY.mp4

    Args:
        save_dir  : 数据根目录（与 save_frame 保持一致）
        episode   : episode 编号
        fps       : 视频帧率，应与录制时的实际 FPS 一致
        quality   : "high" → 图片为 .png，"balance" → 图片为 .jpg
        n_cameras : 相机数量，默认 4
    """
    try:
        import cv2
    except ImportError:
        print("[make_episode_videos] opencv-python not found; trying imageio...")
        _make_episode_videos_imageio(save_dir, episode, fps, quality, n_cameras)
        return

    ext = ".png" if quality == "high" else ".jpg"
    ep_dir = os.path.join(save_dir, f"episode_{episode:04d}")

    for cam_id in range(n_cameras):
        rgb_dir    = os.path.join(ep_dir, f"cam{cam_id}", "rgb")
        video_path = os.path.join(ep_dir, f"cam{cam_id}", f"cam{cam_id}.mp4")

        if not os.path.isdir(rgb_dir):
            print(f"[make_episode_videos] rgb dir not found: {rgb_dir}  (skip)")
            continue

        frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(ext)])
        if not frames:
            print(f"[make_episode_videos] no {ext} frames in {rgb_dir}  (skip)")
            continue

        # 读第一帧确定分辨率
        first = cv2.imread(os.path.join(rgb_dir, frames[0]))
        if first is None:
            print(f"[make_episode_videos] failed to read {frames[0]}  (skip)")
            continue
        h, w = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for fname in frames:
            img = cv2.imread(os.path.join(rgb_dir, fname))
            if img is not None:
                writer.write(img)

        writer.release()
        print(f"[make_episode_videos] cam{cam_id}: {len(frames)} frames → {video_path}")

    print(f"[make_episode_videos] Done (episode {episode}).")


def _make_episode_videos_imageio(
    save_dir: str,
    episode: int,
    fps: float,
    quality: str,
    n_cameras: int,
) -> None:
    """imageio 后备实现（当 cv2 不可用时）。"""
    try:
        import imageio
    except ImportError:
        print("[make_episode_videos] imageio not found either. "
              "Install opencv-python or imageio to enable video export.")
        return

    from PIL import Image as _Image

    ext = ".png" if quality == "high" else ".jpg"
    ep_dir = os.path.join(save_dir, f"episode_{episode:04d}")

    for cam_id in range(n_cameras):
        rgb_dir    = os.path.join(ep_dir, f"cam{cam_id}", "rgb")
        video_path = os.path.join(ep_dir, f"cam{cam_id}", f"cam{cam_id}.mp4")

        if not os.path.isdir(rgb_dir):
            continue
        frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(ext)])
        if not frames:
            continue

        with imageio.get_writer(video_path, fps=fps) as writer:
            for fname in frames:
                img = np.array(_Image.open(os.path.join(rgb_dir, fname)).convert("RGB"))
                writer.append_data(img)

        print(f"[make_episode_videos] cam{cam_id}: {len(frames)} frames → {video_path}")

    print(f"[make_episode_videos] Done (episode {episode}).")


def setup_corner_cameras(
    room_length: float = 4.0,
    room_width: float = 3.0,
    room_height: float = 2.44,
) -> "CameraView":
    """创建 4 个天花板角落相机，对应真实物理房间布局。

    房间以世界原点为中心：
      X ∈ [-room_length/2, +room_length/2]
      Y ∈ [-room_width/2,  +room_width/2]
      Z ∈ [0, room_height]

    每个相机朝向地面中心 (0, 0, 0)。调用后需至少执行一次 world.step()，
    再调用 apply_real_intrinsics() 将真实内参写入 USD Stage。

    Returns:
        CameraView 实例（640×480，RGB + Depth）
    """
    import omni.replicator.core as rep
    from isaacsim.sensors.camera import CameraView

    half_l = room_length / 2
    half_w = room_width  / 2

    cam_positions = [
        ( half_l,  half_w, room_height),   # cam0  +X+Y
        ( half_l, -half_w, room_height),   # cam1  +X-Y
        (-half_l,  half_w, room_height),   # cam2  -X+Y
        (-half_l, -half_w, room_height),   # cam3  -X-Y
    ]

    for pos in cam_positions:
        rep.create.camera(position=pos, look_at=(0.0, 0.0, 0.0))

    print(f"[camera_utils] 4 corner cameras: "
          f"room {room_length}m × {room_width}m × {room_height}m, look_at=(0,0,0)")

    camera_view = CameraView(
        name="corner_cameras",
        camera_resolution=(640, 480),
        prim_paths_expr="/Replicator/Camera_Xform*/Camera",
        output_annotators=["rgb", "depth"],
    )
    return camera_view


def apply_real_intrinsics(
    focal_length_mm: float = 1.93,
    h_aperture_mm: float = 2.038,
    v_aperture_mm: float = 1.529,
    n_cameras: int = 4,
) -> None:
    """将真实 RealSense 相机内参写入 USD Stage 上的 rep 相机。

    必须在至少一次 world.step() 之后调用（rep prim 此时已写入 Stage）。

    默认值对应 RealSense D435，分辨率 640×480：
      fx=605.91, fy=605.54, cx=326.88, cy=258.80

    换算关系：
      fx = focal_length_mm * resolution_w / horizontal_aperture_mm
      fy = focal_length_mm * resolution_h / vertical_aperture_mm
    """
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    ok = 0
    for i in range(1, n_cameras + 1):
        path = f"/Replicator/Camera_Xform_{i:02d}/Camera"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[apply_real_intrinsics] prim not found: {path}  (skip)")
            continue
        cam = UsdGeom.Camera(prim)
        cam.GetFocalLengthAttr().Set(focal_length_mm)
        cam.GetHorizontalApertureAttr().Set(h_aperture_mm)
        cam.GetVerticalApertureAttr().Set(v_aperture_mm)
        ok += 1
    print(f"[apply_real_intrinsics] {ok}/{n_cameras} cameras updated — "
          f"focal={focal_length_mm}mm  "
          f"→ fx≈{focal_length_mm*640/h_aperture_mm:.1f}  "
          f"fy≈{focal_length_mm*480/v_aperture_mm:.1f}")


def print_camera_intrinsics(
    camera_view,
    resolution_w: int = 640,
    resolution_h: int = 480,
    save_dir: str = ".",
) -> None:
    """读取并打印 rep 创建的相机的真实 USD 内参，同时保存到 JSON 文件。

    调用时机：必须在 setup_cameras() + 至少一次 world.step() 之后，
              此时 rep 相机 prim 已被写入 USD Stage，属性值才有效。

    Args:
        camera_view  : setup_cameras() 返回的 CameraView 实例（暂未使用，保留供扩展）
        resolution_w : 相机水平分辨率（像素），默认 640
        resolution_h : 相机垂直分辨率（像素），默认 480
        save_dir     : JSON 文件保存目录，默认为当前目录
                       文件名固定为 camera_intrinsics.json
    """
    import json
    import omni.usd
    import numpy as np
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()

    # rep.create.camera 生成的 prim 路径规律：
    #   /Replicator/Camera_Xform_01/Camera
    #   /Replicator/Camera_Xform_02/Camera  ...
    cam_paths = [
        "/Replicator/Camera_Xform_01/Camera",
        "/Replicator/Camera_Xform_02/Camera",
        "/Replicator/Camera_Xform_03/Camera",
        "/Replicator/Camera_Xform_04/Camera",
    ]

    print("\n" + "=" * 62)
    print("[camera_intrinsics] Real camera parameters (from USD stage):")
    print("=" * 62)

    all_cameras = {}   # 用于写入 JSON

    for i, path in enumerate(cam_paths):
        prim = stage.GetPrimAtPath(path)
        cam_key = f"cam{i}"

        if not prim.IsValid():
            print(f"  Cam{i}: prim not found at {path}  (skip)")
            all_cameras[cam_key] = {"error": f"prim not found at {path}"}
            continue

        cam_schema   = UsdGeom.Camera(prim)
        focal_length = cam_schema.GetFocalLengthAttr().Get()        # mm
        h_aperture   = cam_schema.GetHorizontalApertureAttr().Get() # mm
        v_aperture   = cam_schema.GetVerticalApertureAttr().Get()   # mm

        if focal_length is None or h_aperture is None:
            print(f"  Cam{i} ({path}): attributes None — try calling 1 frame later")
            all_cameras[cam_key] = {"error": "attributes not yet available"}
            continue

        hfov   = float(2 * np.degrees(np.arctan(h_aperture / (2 * focal_length))))
        vfov   = float(2 * np.degrees(np.arctan(v_aperture / (2 * focal_length))))
        fx     = float(resolution_w * focal_length / h_aperture)
        fy     = float(resolution_h * focal_length / v_aperture)
        cx     = resolution_w / 2.0
        cy     = resolution_h / 2.0

        # ── 终端打印 ──────────────────────────────────────────────────────────
        print(f"  Cam{i}  ({path})")
        print(f"    focal_length        = {focal_length:.4f} mm")
        print(f"    horizontal_aperture = {h_aperture:.4f} mm")
        print(f"    vertical_aperture   = {v_aperture:.4f} mm")
        print(f"    resolution          = {resolution_w} × {resolution_h} px")
        print(f"    HFOV                = {hfov:.2f}°")
        print(f"    VFOV                = {vfov:.2f}°")
        print(f"    fx, fy              = {fx:.4f}, {fy:.4f}  px")
        print(f"    cx, cy              = {cx:.1f}, {cy:.1f}  px")
        print(f"    K  = [[{fx:.4f}, 0, {cx:.1f}],")
        print(f"           [0, {fy:.4f}, {cy:.1f}],")
        print(f"           [0, 0, 1]]")
        print()

        # ── 构建该相机的 JSON 条目 ─────────────────────────────────────────────
        all_cameras[cam_key] = {
            "prim_path":            path,
            "resolution_width":     resolution_w,
            "resolution_height":    resolution_h,
            "focal_length_mm":      float(focal_length),
            "horizontal_aperture_mm": float(h_aperture),
            "vertical_aperture_mm": float(v_aperture),
            "hfov_deg":             round(hfov, 4),
            "vfov_deg":             round(vfov, 4),
            "fx":                   round(fx, 4),
            "fy":                   round(fy, 4),
            "cx":                   cx,
            "cy":                   cy,
            "K": [
                [round(fx, 4), 0.0,          cx],
                [0.0,          round(fy, 4), cy],
                [0.0,          0.0,          1.0],
            ],
        }

    # ── 写入 JSON ─────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "camera_intrinsics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_cameras, f, indent=2, ensure_ascii=False)

    print(f"[camera_intrinsics] Saved to: {json_path}")
    print("=" * 62 + "\n")