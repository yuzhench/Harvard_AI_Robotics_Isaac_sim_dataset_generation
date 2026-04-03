# tasks.py
# ── Task 抽象体系 ─────────────────────────────────────────────────────────────
# 只依赖标准库 math 和 numpy，可以在 SimulationApp 初始化之前安全 import。
#
# 使用方式：
#   from tasks import Task, ApproachTask
#   current_task: Task = ApproachTask(target_pos=[5.0, 0.0], arrival_dist=0.4)
#
# 新增任务只需继承 Task 并实现 is_done() / get_command() 两个方法即可。
# ─────────────────────────────────────────────────────────────────────────────

import math
import numpy as np


class Task:
    """所有任务的基类，子类必须实现 is_done() 和 get_command()。"""

    def setup(self, world, assets_root_path: str) -> None:
        """在仿真开始前加载任务所需的额外场景资产（如目标标记球、道具等）。

        默认空实现；不需要加载任何额外资产的任务无需覆盖此方法。

        Args:
            world           : Isaac Sim World 实例
            assets_root_path: Isaac Sim 资产根路径（由 get_assets_root_path() 返回）
        """
        pass

    def is_done(self, pos: np.ndarray, quat: np.ndarray) -> bool:
        """判断任务是否完成。

        Args:
            pos : 机器人世界坐标 [x, y, z]
            quat: 机器人朝向四元数 [w, x, y, z]

        Returns:
            True  → 任务结束，触发 episode reset
            False → 继续执行
        """
        raise NotImplementedError

    def get_command(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """根据当前状态计算机器人指令。

        Args:
            pos : 机器人世界坐标 [x, y, z]
            quat: 机器人朝向四元数 [w, x, y, z]

        Returns:
            base_command: shape (3,)，分别对应前进速度、侧向速度、转向速度
        """
        raise NotImplementedError

    def description(self) -> str:
        """返回任务的可读描述，用于日志打印。"""
        return self.__class__.__name__


class ApproachTask(Task):
    """导航至目标点任务：机器人走到目标位置附近（欧氏距离 < arrival_dist）即完成。

    Args:
        target_pos  : 目标坐标 [x, y]（float 列表或 ndarray）
        arrival_dist: 到达判定半径，单位米，默认 0.4 m

    示例：
        task = ApproachTask(target_pos=[5.0, 3.0], arrival_dist=0.5)
    """

    def __init__(self, target_pos, arrival_dist: float = 0.4):
        self.target_pos   = np.array(target_pos, dtype=float)
        self.arrival_dist = arrival_dist

    def setup(self, world, assets_root_path: str) -> None:
        """在目标位置放一个半透明绿色球，直观标示目的地和到达判定范围。"""
        # ⚠️ 懒加载：pxr / omni 必须在 SimulationApp 初始化后才能 import
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()

        # ── 创建球体 Prim ──────────────────────────────────────────────────────
        sphere_path = "/World/TargetMarker"
        sphere      = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(0.1)  # 球的视觉半径（固定 0.1m，与到达判定半径无关）

        # 位置：目标 (x, y)，z=0 贴地
        UsdGeom.XformCommonAPI(sphere).SetTranslate(
            Gf.Vec3d(float(self.target_pos[0]), float(self.target_pos[1]), 0.0)
        )

        # ── 创建半透明绿色材质 ─────────────────────────────────────────────────
        mat_path = "/World/TargetMarkerMat"
        material = UsdShade.Material.Define(stage, mat_path)
        shader   = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.0, 0.9, 0.2)   # 鲜绿色
        )
        shader.CreateInput("opacity",          Sdf.ValueTypeNames.Float).Set(0.35)
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        # ── 绑定材质到球体 ─────────────────────────────────────────────────────
        UsdShade.MaterialBindingAPI(sphere).Bind(material)

        print(f"[ApproachTask] Target marker: "
              f"pos=({self.target_pos[0]:.1f}, {self.target_pos[1]:.1f}), "
              f"radius={self.arrival_dist}m")

    def is_done(self, pos: np.ndarray, quat: np.ndarray) -> bool:
        dx = self.target_pos[0] - pos[0]
        dy = self.target_pos[1] - pos[1]
        return math.sqrt(dx ** 2 + dy ** 2) < self.arrival_dist

    def get_command(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        dx = self.target_pos[0] - pos[0]
        dy = self.target_pos[1] - pos[1]
        distance    = math.sqrt(dx ** 2 + dy ** 2)
        target_yaw  = math.atan2(dy, dx)
        w, qx, qy, qz = quat
        current_yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        angle_diff  = math.atan2(
            math.sin(target_yaw - current_yaw),
            math.cos(target_yaw - current_yaw),
        )
        forward = min(1.5, distance * 0.4) if abs(angle_diff) < 0.4 else 0.2
        turn    = max(-2.0, min(2.0, angle_diff * 2.0))
        return np.array([forward, 0.0, turn])

    def description(self) -> str:
        return f"ApproachTask(target={self.target_pos.tolist()}, arrival_dist={self.arrival_dist}m)"

