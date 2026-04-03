#!/usr/bin/env python3
"""
camera_fov_visualizer.py
─────────────────────────────────────────────────────────────────────────────
交互式相机视野可视化工具（独立脚本，无需启动 Isaac Sim）

功能：
  - 启动时自动从 camera_intrinsics.json 加载真实内参
  - 3D 显示所有相机的位置与视锥体（frustum）

  左侧控件（从上到下）：
  ① Camera Select  — 选择「全部」或「单个相机」
  ② Object Height  — 目标平面高度
  ③ Zone Width/Depth — 监视区域尺寸
  ④ Camera Orient  — 朝向模式：
       Look at Origin  → 所有相机朝向原点（默认）
       Pitch Control   → 固定水平方位角 + 可调俯仰角
  ⑤ Pitch Angle    — 俯仰角（仅 Pitch Control 模式生效）
  ⑥ Camera Position（选中单个相机时生效）
       Cam X / Cam Y / Cam Z

用法：
  python camera_fov_visualizer.py
  python camera_fov_visualizer.py --json /path/to/camera_intrinsics.json

依赖：
  pip install numpy matplotlib
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ══════════════════════════════════════════════════════════════════════════════
# 命令行参数
# ══════════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_parser = argparse.ArgumentParser(description="Camera FOV Visualizer")
_parser.add_argument("--json", type=str, default=None,
                     metavar="PATH", help="Path to camera_intrinsics.json (auto-selected by mode)")
_parser.add_argument("--real", action="store_true",
                     help="Real room corner-camera layout (4m × 3m × 2.44m)")
_args, _ = _parser.parse_known_args()
_USE_REAL = _args.real

# ══════════════════════════════════════════════════════════════════════════════
# 相机外参 & 布局（--real：真实角落相机  |  默认：模拟环绕相机）
# ══════════════════════════════════════════════════════════════════════════════
PITCH_MIN_DEG = -90.0
PITCH_MAX_DEG =  45.0
POS_XY_MIN, POS_XY_MAX = -12.0, 12.0
POS_Z_MIN,  POS_Z_MAX  =   0.1,  8.0
PLANE_Z_MIN = 0.0

LOOK_AT    = np.array([0.0, 0.0, 0.0])
CAM_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

if _USE_REAL:
    # 真实物理房间：4 m × 3 m × 2.44 m，以原点为中心
    ROOM_L, ROOM_W, ROOM_H = 4.0, 3.0, 2.44
    CAM_R, CAM_H           = 0.0, ROOM_H
    CAMERAS = [
        {"id": 0, "pos": np.array([ 2.0,  1.5, ROOM_H], dtype=float), "label": "+X+Y"},
        {"id": 1, "pos": np.array([ 2.0, -1.5, ROOM_H], dtype=float), "label": "+X-Y"},
        {"id": 2, "pos": np.array([-2.0,  1.5, ROOM_H], dtype=float), "label": "-X+Y"},
        {"id": 3, "pos": np.array([-2.0, -1.5, ROOM_H], dtype=float), "label": "-X-Y"},
    ]
    _DEFAULT_PITCH_DEG = -45.0
    _DEFAULT_ZONE_W    = 4.0
    _DEFAULT_ZONE_D    = 3.0
    PLANE_Z_MAX        = ROOM_H
    JSON_PATH = _args.json or os.path.join(_SCRIPT_DIR, "real_camera_intrinsics.json")
else:
    ROOM_L = ROOM_W = ROOM_H = None
    CAM_R, CAM_H = 6.0, 2.5
    CAMERAS = [
        {"id": 0, "pos": np.array([ CAM_R,     0, CAM_H], dtype=float), "label": "+X"},
        {"id": 1, "pos": np.array([-CAM_R,     0, CAM_H], dtype=float), "label": "-X"},
        {"id": 2, "pos": np.array([    0,  CAM_R, CAM_H], dtype=float), "label": "+Y"},
        {"id": 3, "pos": np.array([    0, -CAM_R, CAM_H], dtype=float), "label": "-Y"},
    ]
    _DEFAULT_PITCH_DEG = -31.0
    _DEFAULT_ZONE_W    = 5.0
    _DEFAULT_ZONE_D    = 5.0
    PLANE_Z_MAX        = 5.0
    JSON_PATH = _args.json or os.path.join(_SCRIPT_DIR, "collected_data", "camera_intrinsics.json")

# ══════════════════════════════════════════════════════════════════════════════
# 从 JSON 加载内参
# ══════════════════════════════════════════════════════════════════════════════

def _load_intrinsics(json_path: str) -> dict:
    FB_HFOV, FB_VFOV = np.radians(47.1686), np.radians(35.3394)
    FB_FOCAL         = 24.0
    FB_FX, FB_FY     = 732.9993, 753.3942
    FB_RW, FB_RH     = 640, 480

    result = {}
    if not os.path.isfile(json_path):
        print(f"[visualizer] JSON not found: {json_path}  (using fallback)")
        for cam in CAMERAS:
            result[cam["id"]] = {"hfov": FB_HFOV, "vfov": FB_VFOV,
                                  "focal_mm": FB_FOCAL, "fx": FB_FX, "fy": FB_FY,
                                  "res_w": FB_RW, "res_h": FB_RH, "source": "fallback"}
        return result

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"[visualizer] Loaded: {json_path}")

    valid_ref = next((raw[f"cam{c['id']}"] for c in CAMERAS
                      if f"cam{c['id']}" in raw and "error" not in raw[f"cam{c['id']}"]), {})

    for cam in CAMERAS:
        cid   = cam["id"]
        entry = raw.get(f"cam{cid}", {})
        ref   = valid_ref

        if "error" in entry:
            hfov = np.radians(ref.get("hfov_deg", np.degrees(FB_HFOV)))
            vfov = np.radians(ref.get("vfov_deg", np.degrees(FB_VFOV)))
            src  = "fallback"
        else:
            hfov, vfov = np.radians(entry["hfov_deg"]), np.radians(entry["vfov_deg"])
            src = "json"

        result[cid] = {
            "hfov":     hfov, "vfov": vfov,
            "focal_mm": entry.get("focal_length_mm",  ref.get("focal_length_mm",  FB_FOCAL)),
            "fx":       entry.get("fx",                ref.get("fx",               FB_FX)),
            "fy":       entry.get("fy",                ref.get("fy",               FB_FY)),
            "res_w":    entry.get("resolution_width",  FB_RW),
            "res_h":    entry.get("resolution_height", FB_RH),
            "source":   src,
        }
        print(f"[visualizer]   cam{cid}: HFOV={np.degrees(hfov):.2f}° "
              f"VFOV={np.degrees(vfov):.2f}° [{src}]")
    return result


CAM_INTRINSICS = _load_intrinsics(JSON_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# 朝向辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _azimuth_toward_origin(cam_pos: np.ndarray) -> float:
    """返回从相机位置指向原点的水平方位角（弧度）。"""
    dx, dy = -cam_pos[0], -cam_pos[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return float(np.arctan2(dy, dx))


def get_target(cam_pos: np.ndarray, orient_mode: str, pitch_rad: float) -> np.ndarray:
    """根据朝向模式计算有效的 look-at 目标点。

    look_at : 直接看向世界原点
    pitch   : 保持水平方位（始终朝向原点方向），俯仰角由 pitch_rad 控制
    """
    if orient_mode == "look_at":
        return LOOK_AT.copy()
    # pitch 模式：沿水平方位 + 俯仰
    azim = _azimuth_toward_origin(cam_pos)
    fwd  = np.array([
        np.cos(pitch_rad) * np.cos(azim),
        np.cos(pitch_rad) * np.sin(azim),
        np.sin(pitch_rad),
    ])
    return cam_pos + fwd * 20.0   # 目标点在前方 20 m


# ══════════════════════════════════════════════════════════════════════════════
# 数学工具
# ══════════════════════════════════════════════════════════════════════════════

def look_at_axes(cam_pos, target):
    fwd = target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0., 0., 1.])
    if abs(np.dot(fwd, world_up)) > 0.99:
        world_up = np.array([0., 1., 0.])
    right = np.cross(fwd, world_up); right /= np.linalg.norm(right)
    up    = np.cross(right, fwd);    up    /= np.linalg.norm(up)
    return right, up, fwd


def frustum_corners(cam_pos, target, hfov, vfov, dist):
    right, up, fwd = look_at_axes(cam_pos, target)
    th = np.tan(hfov / 2) * dist
    tv = np.tan(vfov / 2) * dist
    c  = cam_pos + fwd * dist
    return [c + right*th - up*tv, c - right*th - up*tv,
            c - right*th + up*tv, c + right*th + up*tv]


def intersect_with_plane(cam_pos, target, hfov, vfov, plane_z):
    right, up, fwd = look_at_axes(cam_pos, target)
    th, tv = np.tan(hfov / 2), np.tan(vfov / 2)
    ray_dirs = [fwd + right*th - up*tv, fwd - right*th - up*tv,
                fwd - right*th + up*tv, fwd + right*th + up*tv]
    pts = []
    for d in ray_dirs:
        d = d / np.linalg.norm(d)
        if abs(d[2]) < 1e-9: continue
        t = (plane_z - cam_pos[2]) / d[2]
        if t <= 0: continue
        pts.append(cam_pos + t * d)
    if len(pts) < 3:
        return pts, 0.0
    ctr    = np.mean(pts, axis=0)
    angles = [np.arctan2(p[1]-ctr[1], p[0]-ctr[0]) for p in pts]
    pts    = [p for _, p in sorted(zip(angles, pts))]
    n      = len(pts)
    area   = 0.5 * abs(sum(pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1]
                           for i in range(n)))
    return pts, area


def point_in_polygon_2d(pts_xy, polygon_xy):
    if len(polygon_xy) < 3:
        return np.zeros(len(pts_xy), dtype=bool)
    path = mpath.Path(np.array(polygon_xy))
    return path.contains_points(pts_xy)


def zone_coverage(footprint_polys, zone_cx, zone_cy, zone_w, zone_d, n_samples=60):
    if not footprint_polys:
        return 0.0
    xs = np.linspace(zone_cx - zone_w/2, zone_cx + zone_w/2, n_samples)
    ys = np.linspace(zone_cy - zone_d/2, zone_cy + zone_d/2, n_samples)
    gx, gy  = np.meshgrid(xs, ys)
    pts     = np.column_stack([gx.ravel(), gy.ravel()])
    covered = np.zeros(len(pts), dtype=bool)
    for poly in footprint_polys:
        poly_xy = np.array([[p[0], p[1]] for p in poly])
        covered |= point_in_polygon_2d(pts, poly_xy)
    return covered.mean() * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 全局状态
# ══════════════════════════════════════════════════════════════════════════════
state = {
    "selected":    "all",
    "plane_z":      0.0,
    "zone_w":       _DEFAULT_ZONE_W,
    "zone_d":       _DEFAULT_ZONE_D,
    "orient_mode": "look_at",          # "look_at" | "pitch"
    "pitch_deg":    _DEFAULT_PITCH_DEG,
    "overlay":      None,              # None | "intersection" | "union"
}
ZONE_CX, ZONE_CY = 0.0, 0.0
_slider_updating  = False              # 防止 set_val 触发重复回调

# ══════════════════════════════════════════════════════════════════════════════
# 图形布局  figsize=(17, 11)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 11), facecolor="#1a1a2e")

# 3D 主视图
ax3d = fig.add_axes([0.21, 0.05, 0.78, 0.92], projection="3d")
ax3d.set_facecolor("#1a1a2e")
for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor("#333355")
ax3d.tick_params(colors="#aaaacc", labelsize=7)
for attr in ["xaxis", "yaxis", "zaxis"]:
    getattr(ax3d, attr).label.set_color("#aaaacc")

# ─────────────────────────────────────────────────────────────────────────────
# 左侧控件（从上到下排列，y 从 0.97 递减）
# ─────────────────────────────────────────────────────────────────────────────

# ── 覆盖区域高亮按钮（顶部两个 Button）──────────────────────────────────────
ax_btn_sep = fig.add_axes([0.01, 0.938, 0.19, 0.026], facecolor="#1a1a40")
ax_btn_sep.axis("off")
ax_btn_sep.text(0.5, 0.5, "── Coverage Highlight ──",
                transform=ax_btn_sep.transAxes,
                color="#ffcc44", fontsize=8, ha="center", va="center",
                family="monospace")

ax_btn_isect = fig.add_axes([0.01,  0.967, 0.09, 0.028], facecolor="#1a1a40")
btn_isect = Button(ax_btn_isect, "∩ All Cams", color="#1a1a40", hovercolor="#2a1530")
btn_isect.label.set_color("#ff6644")
btn_isect.label.set_fontsize(8)

ax_btn_union = fig.add_axes([0.105, 0.967, 0.09, 0.028], facecolor="#1a1a40")
btn_union = Button(ax_btn_union, "∪ Any Cam", color="#1a1a40", hovercolor="#152a18")
btn_union.label.set_color("#44ff88")
btn_union.label.set_fontsize(8)

# ① 相机单选框
ax_radio = fig.add_axes([0.01, 0.645, 0.19, 0.285], facecolor="#12122a")
radio_labels = ["All cameras"] + [f"Cam {c['id']}  ({c['label']})" for c in CAMERAS]
radio = RadioButtons(ax_radio, labels=radio_labels, activecolor="#2ecc71")
for txt in radio.labels:
    txt.set_color("white"); txt.set_fontsize(9)
ax_radio.set_title("Camera", color="white", fontsize=10, pad=5)

# ② Object Height
ax_sl_h = fig.add_axes([0.02, 0.62, 0.16, 0.022], facecolor="#12122a")
sl_height = Slider(ax_sl_h, "", PLANE_Z_MIN, PLANE_Z_MAX,
                   valinit=0.0, valstep=0.02, color="#2ecc71")
sl_height.valtext.set_color("white")
ax_sl_h.set_title("Object Height (m)", color="white", fontsize=8, pad=3)

# ③ Zone Width / Depth
ax_sl_w = fig.add_axes([0.02, 0.54, 0.16, 0.022], facecolor="#12122a")
sl_zw = Slider(ax_sl_w, "", 0.5, 20.0, valinit=_DEFAULT_ZONE_W, valstep=0.5, color="#e67e22")
sl_zw.valtext.set_color("white")
ax_sl_w.set_title("Zone Width  X (m)", color="#e67e22", fontsize=8, pad=3)

ax_sl_d = fig.add_axes([0.02, 0.47, 0.16, 0.022], facecolor="#12122a")
sl_zd = Slider(ax_sl_d, "", 0.5, 20.0, valinit=_DEFAULT_ZONE_D, valstep=0.5, color="#e67e22")
sl_zd.valtext.set_color("white")
ax_sl_d.set_title("Zone Depth  Y (m)", color="#e67e22", fontsize=8, pad=3)

# ── 分隔：Camera Orient ────────────────────────────────────────────────────────
ax_orient_sep = fig.add_axes([0.01, 0.425, 0.19, 0.035], facecolor="#1a1a40")
ax_orient_sep.axis("off")
ax_orient_sep.text(0.5, 0.5, "── Camera Orientation ──",
                   transform=ax_orient_sep.transAxes,
                   color="#88aaff", fontsize=8.5, ha="center", va="center",
                   family="monospace")

# ④ 朝向模式单选框（两项）
ax_orient = fig.add_axes([0.01, 0.325, 0.19, 0.09], facecolor="#12122a")
orient_radio = RadioButtons(
    ax_orient,
    labels=["Look at Origin", "Pitch Control"],
    activecolor="#88aaff",
)
for txt in orient_radio.labels:
    txt.set_color("white"); txt.set_fontsize(9)
ax_orient.set_title("Mode", color="#88aaff", fontsize=9, pad=4)

# ⑤ Pitch Angle 滑动条
ax_sl_pitch = fig.add_axes([0.02, 0.265, 0.16, 0.022], facecolor="#12122a")
sl_pitch = Slider(ax_sl_pitch, "", PITCH_MIN_DEG, PITCH_MAX_DEG,
                  valinit=_DEFAULT_PITCH_DEG, valstep=1.0, color="#88aaff")
sl_pitch.valtext.set_color("white")
ax_sl_pitch.set_title(
    f"Pitch Angle (°)  [default = {_DEFAULT_PITCH_DEG:.0f}°]",
    color="#8899cc", fontsize=7.5, pad=3)

# ── 分隔：Camera Position ─────────────────────────────────────────────────────
ax_pos_sep = fig.add_axes([0.01, 0.215, 0.19, 0.035], facecolor="#1a1a40")
ax_pos_sep.axis("off")
_pos_title = ax_pos_sep.text(0.5, 0.65, "── Camera Position ──",
                              transform=ax_pos_sep.transAxes,
                              color="#c39bd3", fontsize=8.5, ha="center", va="center",
                              family="monospace")
_pos_hint = ax_pos_sep.text(0.5, 0.05, "(select single cam to edit)",
                             transform=ax_pos_sep.transAxes,
                             color="#666688", fontsize=7, ha="center", va="bottom",
                             family="monospace")

# ⑥ Cam X / Y / Z
ax_sl_cx = fig.add_axes([0.02, 0.155, 0.16, 0.022], facecolor="#12122a")
sl_cx = Slider(ax_sl_cx, "", POS_XY_MIN, POS_XY_MAX,
               valinit=CAM_R, valstep=0.1, color="#9b59b6")
sl_cx.valtext.set_color("white")
ax_sl_cx.set_title("Cam  X (m)", color="#c39bd3", fontsize=8, pad=3)

ax_sl_cy = fig.add_axes([0.02, 0.09, 0.16, 0.022], facecolor="#12122a")
sl_cy = Slider(ax_sl_cy, "", POS_XY_MIN, POS_XY_MAX,
               valinit=0.0, valstep=0.1, color="#9b59b6")
sl_cy.valtext.set_color("white")
ax_sl_cy.set_title("Cam  Y (m)", color="#c39bd3", fontsize=8, pad=3)

ax_sl_cz = fig.add_axes([0.02, 0.025, 0.16, 0.022], facecolor="#12122a")
sl_cz = Slider(ax_sl_cz, "", POS_Z_MIN, POS_Z_MAX,
               valinit=CAM_H, valstep=0.1, color="#9b59b6")
sl_cz.valtext.set_color("white")
ax_sl_cz.set_title("Cam  Z (m)", color="#c39bd3", fontsize=8, pad=3)


# ══════════════════════════════════════════════════════════════════════════════
# 辅助：同步 Position 滑动条 & Pitch 滑动条
# ══════════════════════════════════════════════════════════════════════════════
def _sync_pos_sliders():
    global _slider_updating
    sel = state["selected"]
    if sel == "all":
        pos = CAMERAS[0]["pos"]
        _pos_hint.set_text("(select single cam to edit)")
        _pos_title.set_color("#444466")
        for sl in (sl_cx, sl_cy, sl_cz):
            sl.poly.set_facecolor("#444455")
    else:
        pos = CAMERAS[sel]["pos"]
        _pos_hint.set_text(f"editing  Cam{sel}  ({CAMERAS[sel]['label']})")
        _pos_title.set_color("#c39bd3")
        for sl in (sl_cx, sl_cy, sl_cz):
            sl.poly.set_facecolor("#9b59b6")
    _slider_updating = True
    sl_cx.set_val(float(np.clip(pos[0], POS_XY_MIN, POS_XY_MAX)))
    sl_cy.set_val(float(np.clip(pos[1], POS_XY_MIN, POS_XY_MAX)))
    sl_cz.set_val(float(np.clip(pos[2], POS_Z_MIN,  POS_Z_MAX)))
    _slider_updating = False


def _sync_pitch_slider():
    """根据朝向模式更新 Pitch 滑动条的样式（active / dimmed）。"""
    is_pitch = (state["orient_mode"] == "pitch")
    sl_pitch.poly.set_facecolor("#88aaff" if is_pitch else "#334455")
    sl_pitch.valtext.set_color("white" if is_pitch else "#556677")


# ══════════════════════════════════════════════════════════════════════════════
# 绘制主函数
# ══════════════════════════════════════════════════════════════════════════════
def draw():
    ax3d.cla()
    ax3d.set_facecolor("#1a1a2e")
    ax3d.set_xlabel("X (m)", color="#aaaacc", labelpad=4)
    ax3d.set_ylabel("Y (m)", color="#aaaacc", labelpad=4)
    ax3d.set_zlabel("Z (m)", color="#aaaacc", labelpad=4)
    ax3d.tick_params(colors="#aaaacc", labelsize=7)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor("#333355")

    plane_z    = state["plane_z"]
    zone_w     = state["zone_w"]
    zone_d     = state["zone_d"]
    sel        = state["selected"]
    orient     = state["orient_mode"]
    pitch_rad  = np.radians(state["pitch_deg"])

    # ── 标题 ──────────────────────────────────────────────────────────────────
    orient_tag = (f"Pitch {state['pitch_deg']:.0f}°"
                  if orient == "pitch" else "Look-at Origin")
    cam_tag    = "All Cameras" if sel == "all" else \
                 f"Cam{sel}({CAMERAS[sel]['label']}) " \
                 f"[{CAMERAS[sel]['pos'][0]:.1f}," \
                 f"{CAMERAS[sel]['pos'][1]:.1f}," \
                 f"{CAMERAS[sel]['pos'][2]:.1f}]"
    ax3d.set_title(f"{cam_tag}  |  {orient_tag}  |  z={plane_z:.2f}m  |  "
                   f"zone {zone_w:.1f}×{zone_d:.1f}m",
                   color="white", fontsize=9, pad=8)

    all_pos = np.array([c["pos"] for c in CAMERAS])
    lim     = max(np.abs(all_pos[:, :2]).max() + 1.5, zone_w/2 + 1.0, zone_d/2 + 1.0)
    max_z   = max(all_pos[:, 2].max() + 0.6, plane_z + 0.3, 2.5)

    # ── 地面参考网格 ──────────────────────────────────────────────────────────
    for v in np.linspace(-lim, lim, 13):
        ax3d.plot([v, v],   [-lim, lim], [0, 0], color="#252540", lw=0.5)
        ax3d.plot([-lim, lim], [v, v],   [0, 0], color="#252540", lw=0.5)

    # ── 真实房间线框（--real 模式）────────────────────────────────────────────
    if _USE_REAL:
        hx, hy = ROOM_L / 2, ROOM_W / 2
        _rc = "#4488aa"
        # 地面 & 天花板矩形
        for zz, lw in [(0, 1.5), (ROOM_H, 1.5)]:
            xs = [-hx,  hx,  hx, -hx, -hx]
            ys = [-hy, -hy,  hy,  hy, -hy]
            ax3d.plot(xs, ys, [zz] * 5, color=_rc, lw=lw, alpha=0.7)
        # 4 根竖柱
        for sx, sy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]:
            ax3d.plot([sx, sx], [sy, sy], [0, ROOM_H], color=_rc, lw=1.0, alpha=0.5)
        ax3d.text(hx, -hy, ROOM_H + 0.08,
                  f"Room {ROOM_L:.0f}m×{ROOM_W:.0f}m×{ROOM_H:.2f}m",
                  color=_rc, fontsize=7, ha="right", va="bottom")

    # ── 监视区域线框（橙色） ───────────────────────────────────────────────────
    ZONE_COLOR = "#ff8c00"
    hx, hy    = zone_w / 2, zone_d / 2
    zone_xy   = np.array([
        [ZONE_CX-hx, ZONE_CY-hy], [ZONE_CX+hx, ZONE_CY-hy],
        [ZONE_CX+hx, ZONE_CY+hy], [ZONE_CX-hx, ZONE_CY+hy],
        [ZONE_CX-hx, ZONE_CY-hy],
    ])
    ax3d.plot(zone_xy[:,0], zone_xy[:,1], np.zeros(5),
              color=ZONE_COLOR, lw=2.0, zorder=7)
    if plane_z > 0.01:
        ax3d.plot(zone_xy[:,0], zone_xy[:,1], np.full(5, plane_z),
                  color=ZONE_COLOR, lw=1.5, ls="--", zorder=7)
        for pt in zone_xy[:4]:
            ax3d.plot([pt[0], pt[0]], [pt[1], pt[1]], [0, plane_z],
                      color=ZONE_COLOR, lw=0.8, ls=":", alpha=0.6)
    ax3d.add_collection3d(
        Poly3DCollection([list(zip(zone_xy[:4,0], zone_xy[:4,1], [0]*4))],
                         alpha=0.06, facecolor=ZONE_COLOR, edgecolor="none"))
    ax3d.text(ZONE_CX, ZONE_CY, 0.05, f"{zone_w:.1f}×{zone_d:.1f} m",
              color=ZONE_COLOR, fontsize=8, ha="center", fontweight="bold", zorder=12)

    # ── 目标平面蓝色背景 ──────────────────────────────────────────────────────
    if plane_z > 0.01:
        px = np.array([-lim, lim, lim, -lim])
        py = np.array([-lim,-lim, lim,  lim])
        ax3d.add_collection3d(
            Poly3DCollection([list(zip(px, py, np.full(4, plane_z)))],
                             alpha=0.07, facecolor="#4488ff",
                             edgecolor="#5599ee", linewidth=0.6))

    # ── 原点标记 & 朝向模式提示 ───────────────────────────────────────────────
    ax3d.scatter([0], [0], [0], color="white", s=40, zorder=8)
    if orient == "pitch":
        # 画一条从原点延伸的水平方位线，帮助理解
        for cam in CAMERAS:
            azim = _azimuth_toward_origin(cam["pos"])
            vx = np.cos(azim) * lim * 0.3
            vy = np.sin(azim) * lim * 0.3
            ax3d.plot([0, vx], [0, vy], [0, 0],
                      color=CAM_COLORS[cam["id"]], lw=0.4,
                      ls=":", alpha=0.3)

    # ── 信息栏初始化 ──────────────────────────────────────────────────────────
    intrinsics_ref = CAM_INTRINSICS.get(0, {})
    info_lines = [
        f"Mode : {orient_tag}",
        f"Focal: {intrinsics_ref.get('focal_mm', '?'):.1f}mm",
        f"Zone : {zone_w:.1f}×{zone_d:.1f}m",
        "─────────────────────",
        f"{'Cam':<4}{'X':>5}{'Y':>5}{'Z':>5}  {'Area':>8}",
        "─────────────────────",
    ]
    all_footprints = []

    # ── 逐相机绘制 ────────────────────────────────────────────────────────────
    for cam in CAMERAS:
        cid   = cam["id"]
        cpos  = cam["pos"].copy()
        color = CAM_COLORS[cid]
        intr  = CAM_INTRINSICS.get(cid, {})
        hfov  = intr.get("hfov", np.radians(47.17))
        vfov  = intr.get("vfov", np.radians(35.34))

        active = (sel == "all") or (sel == cid)
        alpha  = 1.0 if active else 0.12

        # 有效 look-at 目标
        tgt = get_target(cpos, orient, pitch_rad)

        # 相机点 & 标签
        ax3d.scatter(*cpos, color=color, s=100 if active else 60,
                     zorder=10, alpha=alpha, depthshade=False)
        ax3d.text(cpos[0]+0.12, cpos[1]+0.12, cpos[2]+0.14,
                  f"Cam{cid}", color=color, fontsize=8, alpha=alpha, zorder=11)

        # 光轴指示线（虚线到目标点或原点）
        if orient == "pitch":
            # 画到虚拟目标点附近（前方 3 m 处，更直观）
            fwd_vis = (tgt - cpos); fwd_vis /= np.linalg.norm(fwd_vis)
            tgt_vis = cpos + fwd_vis * 3.0
            ax3d.plot([cpos[0], tgt_vis[0]], [cpos[1], tgt_vis[1]],
                      [cpos[2], tgt_vis[2]],
                      color=color, lw=1.2, ls="--", alpha=alpha * 0.8)
        else:
            ax3d.plot([cpos[0], 0], [cpos[1], 0], [cpos[2], 0],
                      color=color, lw=0.8, ls="--", alpha=alpha * 0.5)

        if not active:
            info_lines.append(
                f"C{cid}  {cpos[0]:>4.1f}{cpos[1]:>5.1f}{cpos[2]:>5.1f}   {'─':>7}")
            continue

        # 视锥体线框
        far_dist = max(np.linalg.norm(cpos - tgt) * 1.1, 3.0)
        nc = frustum_corners(cpos, tgt, hfov, vfov, 0.05)
        fc = frustum_corners(cpos, tgt, hfov, vfov, far_dist)
        for corners, a in [(nc, 0.8), (fc, 0.5)]:
            for i in range(4):
                j = (i+1) % 4
                ax3d.plot([corners[i][0], corners[j][0]],
                          [corners[i][1], corners[j][1]],
                          [corners[i][2], corners[j][2]],
                          color=color, lw=0.9, alpha=a)
        for i in range(4):
            ax3d.plot([cpos[0], fc[i][0]], [cpos[1], fc[i][1]],
                      [cpos[2], fc[i][2]],
                      color=color, lw=0.8, alpha=0.4)

        # Footprint
        pts, area = intersect_with_plane(cpos, tgt, hfov, vfov, plane_z)
        if len(pts) >= 3:
            all_footprints.append(pts)
            pts_ring = pts + [pts[0]]
            xs = [p[0] for p in pts_ring]
            ys = [p[1] for p in pts_ring]
            zs = [p[2] for p in pts_ring]
            ax3d.plot(xs, ys, zs, color=color, lw=2.5, zorder=9)
            ax3d.add_collection3d(
                Poly3DCollection([list(zip(xs[:-1], ys[:-1], zs[:-1]))],
                                 alpha=0.25, facecolor=color, edgecolor=color))
            cxp = np.mean([p[0] for p in pts])
            cyp = np.mean([p[1] for p in pts])
            ax3d.text(cxp, cyp, plane_z + 0.1, f"{area:.2f}m²",
                      color=color, fontsize=9, ha="center",
                      fontweight="bold", zorder=12)
            area_str = f"{area:.2f}m²"
        else:
            area_str = "n/a"

        info_lines.append(
            f"C{cid}  {cpos[0]:>4.1f}{cpos[1]:>5.1f}{cpos[2]:>5.1f}  {area_str:>8}")

    # ── 覆盖率 ────────────────────────────────────────────────────────────────
    if all_footprints:
        cov       = zone_coverage(all_footprints, ZONE_CX, ZONE_CY, zone_w, zone_d)
        zone_area = zone_w * zone_d
        cov_color = "#2ecc71" if cov >= 80 else ("#f39c12" if cov >= 50 else "#e74c3c")
        info_lines += [
            "─────────────────────",
            f"Coverage: {cov:.1f}%",
            f"  {cov/100*zone_area:.2f}/{zone_area:.2f} m²",
        ]
        ax3d.text(ZONE_CX, ZONE_CY, plane_z + 0.2,
                  f"Coverage\n{cov:.1f}%",
                  color=cov_color, fontsize=9, ha="center",
                  fontweight="bold", zorder=13)
    else:
        info_lines += ["─────────────────────", "Coverage: n/a"]

    # ── 覆盖区域高亮（Intersection / Union）────────────────────────────────────
    overlay = state["overlay"]
    if overlay:
        # 对全部相机计算投影（与 camera-select 状态无关）
        _all_fp = []
        for cam in CAMERAS:
            cpos = cam["pos"].copy()
            intr = CAM_INTRINSICS.get(cam["id"], {})
            hfov = intr.get("hfov", np.radians(47.17))
            vfov = intr.get("vfov", np.radians(35.34))
            tgt  = get_target(cpos, orient, pitch_rad)
            pts, _ = intersect_with_plane(cpos, tgt, hfov, vfov, plane_z)
            if len(pts) >= 3:
                _all_fp.append(pts)

        if _all_fp:
            n_g   = 90
            xs_g  = np.linspace(-lim, lim, n_g)
            ys_g  = np.linspace(-lim, lim, n_g)
            _gx, _gy = np.meshgrid(xs_g, ys_g)
            pts_g = np.column_stack([_gx.ravel(), _gy.ravel()])

            per_cam = [point_in_polygon_2d(pts_g, np.array([[p[0], p[1]] for p in fp]))
                       for fp in _all_fp]

            if overlay == "intersection":
                mask     = per_cam[0].copy()
                for c in per_cam[1:]:
                    mask &= c
                ov_color = "#ff5533"
                ov_label = "∩ All Cams"
            else:  # union
                mask     = per_cam[0].copy()
                for c in per_cam[1:]:
                    mask |= c
                ov_color = "#33ff88"
                ov_label = "∪ Any Cam"

            if mask.any():
                ax3d.scatter(
                    _gx.ravel()[mask], _gy.ravel()[mask],
                    np.full(mask.sum(), plane_z),
                    c=ov_color, s=6, alpha=0.5, zorder=6,
                    depthshade=False, marker="s",
                )
                cell_area = (2 * lim / n_g) ** 2
                area_m2   = mask.sum() * cell_area
                ax3d.text2D(
                    0.99, 0.02, f"{ov_label}  {area_m2:.2f} m²",
                    transform=ax3d.transAxes,
                    color=ov_color, fontsize=9, ha="right", va="bottom",
                    fontweight="bold",
                    bbox=dict(facecolor="#12122a", alpha=0.8,
                              edgecolor="#333355", boxstyle="round,pad=0.3"),
                )

    # ── 坐标轴范围（等比例缩放：1 m X = 1 m Y = 1 m Z）────────────────────────
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(0, max_z)
    # set_box_aspect 让三轴的视觉长度与数据范围成正比，实现真正的等比例
    ax3d.set_box_aspect([2 * lim, 2 * lim, max_z])

    # ── 图例 ──────────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=CAM_COLORS[c["id"]],
                              label=f"Cam{c['id']} {c['label']}") for c in CAMERAS]
    patches.append(mpatches.Patch(color=ZONE_COLOR, label="Monitor Zone"))
    ax3d.legend(handles=patches, loc="upper right",
                facecolor="#1a1a2e", edgecolor="#555566",
                labelcolor="white", fontsize=8)

    # ── 信息文字叠加在 3D 图左上角 ────────────────────────────────────────────
    ax3d.text2D(0.01, 0.98, "\n".join(info_lines),
                transform=ax3d.transAxes,
                color="white", fontsize=7, va="top", family="monospace",
                bbox=dict(facecolor="#12122a", alpha=0.75, edgecolor="#333355",
                          boxstyle="round,pad=0.4"))

    fig.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
# 回调
# ══════════════════════════════════════════════════════════════════════════════
def on_radio(label):
    state["selected"] = "all" if label == "All cameras" else int(label.split()[1])
    _sync_pos_sliders()
    draw()


def on_orient_radio(label):
    state["orient_mode"] = "look_at" if label == "Look at Origin" else "pitch"
    _sync_pitch_slider()
    draw()


def on_slider_h(val):
    state["plane_z"] = float(val); draw()


def on_slider_w(val):
    state["zone_w"] = float(val); draw()


def on_slider_d(val):
    state["zone_d"] = float(val); draw()


def on_slider_pitch(val):
    state["pitch_deg"] = float(val)
    if state["orient_mode"] == "pitch":
        draw()


def on_slider_cx(val):
    if _slider_updating: return
    sel = state["selected"]
    if sel == "all": return
    CAMERAS[sel]["pos"][0] = float(val); draw()


def on_slider_cy(val):
    if _slider_updating: return
    sel = state["selected"]
    if sel == "all": return
    CAMERAS[sel]["pos"][1] = float(val); draw()


def on_slider_cz(val):
    if _slider_updating: return
    sel = state["selected"]
    if sel == "all": return
    CAMERAS[sel]["pos"][2] = float(val); draw()


def _update_btn_style():
    """根据当前 overlay 状态更新 button 背景色，给用户视觉反馈。"""
    ov = state["overlay"]
    btn_isect.color        = "#3a1520" if ov == "intersection" else "#1a1a40"
    btn_isect.hovercolor   = "#4a2030" if ov == "intersection" else "#2a1530"
    btn_union.color        = "#153520" if ov == "union"        else "#1a1a40"
    btn_union.hovercolor   = "#206030" if ov == "union"        else "#152a18"
    btn_isect.ax.set_facecolor(btn_isect.color)
    btn_union.ax.set_facecolor(btn_union.color)


def on_btn_isect(_event):
    state["overlay"] = None if state["overlay"] == "intersection" else "intersection"
    _update_btn_style()
    draw()


def on_btn_union(_event):
    state["overlay"] = None if state["overlay"] == "union" else "union"
    _update_btn_style()
    draw()


btn_isect.on_clicked(on_btn_isect)
btn_union.on_clicked(on_btn_union)

radio.on_clicked(on_radio)
orient_radio.on_clicked(on_orient_radio)
sl_height.on_changed(on_slider_h)
sl_zw.on_changed(on_slider_w)
sl_zd.on_changed(on_slider_d)
sl_pitch.on_changed(on_slider_pitch)
sl_cx.on_changed(on_slider_cx)
sl_cy.on_changed(on_slider_cy)
sl_cz.on_changed(on_slider_cz)

# ── 首次初始化 ────────────────────────────────────────────────────────────────
_sync_pos_sliders()
_sync_pitch_slider()
draw()
plt.show()
