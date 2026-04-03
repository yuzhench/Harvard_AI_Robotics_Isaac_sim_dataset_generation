# asset_config.py
# ── 资产配置中心 ──────────────────────────────────────────────────────────────
# 统一管理场景、机器人等所有资产的路径与参数。
# 只包含纯 Python 数据（字符串、数字、字典），无需 import Isaac Sim，
# 可在 SimulationApp 初始化之前安全 import。
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 场景配置
#   key  : --scene 命令行参数的值
#   value: Isaac Sim 资产库中的相对 USD 路径（会拼接 assets_root_path 使用）
# ══════════════════════════════════════════════════════════════════════════════

SCENE_MAP = {
    "grid":                 "/Isaac/Environments/Grid/default_environment.usd",
    "grid_black":           "/Isaac/Environments/Grid/gridroom_black.usd",
    "warehouse":            "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_full":       "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "warehouse_forklifts":  "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
    "hospital":             "/Isaac/Environments/Hospital/hospital.usd",
    "office":               "/Isaac/Environments/Office/office.usd",
    "simple_room":          "/Isaac/Environments/Simple_Room/simple_room.usd",
}


# ══════════════════════════════════════════════════════════════════════════════
# 机器人配置
#   每个条目描述一种机器人的 prim 路径、名称、初始位置等。
#   之后可按需扩展更多字段（policy 路径、初始姿态等）。
# ══════════════════════════════════════════════════════════════════════════════

ROBOT_CONFIG = {
    "spot": {
        "prim_path": "/World/Spot",
        "name":      "Spot",
        "position":  [-2, -2, 0.8],   # [x, y, z]，单位米
    },
    # 示例：之后添加 H1 或其他机器人时在此处补充
    # "h1": {
    #     "prim_path": "/World/H1",
    #     "name":      "H1",
    #     "position":  [0.0, 0.0, 1.05],
    # },
}


# ══════════════════════════════════════════════════════════════════════════════
# 真实物理环境配置
#   对应实验室房间：4 m（长 X）× 3 m（宽 Y）× 2.44 m（高）
#   坐标系以地面中心为原点：X ∈ [-2, 2]，Y ∈ [-1.5, 1.5]，Z ∈ [0, 2.44]
# ══════════════════════════════════════════════════════════════════════════════

REAL_ROOM = {
    "length": 4.0,    # X 轴方向，米
    "width":  3.0,    # Y 轴方向，米
    "height": 2.44,   # 天花板高度，米
}

# 4 个天花板角落相机（看向地面中心原点）
CORNER_CAMERAS = [
    {"id": 0, "pos": [ 2.0,  1.5, 2.44], "label": "+X+Y"},
    {"id": 1, "pos": [ 2.0, -1.5, 2.44], "label": "+X-Y"},
    {"id": 2, "pos": [-2.0,  1.5, 2.44], "label": "-X+Y"},
    {"id": 3, "pos": [-2.0, -1.5, 2.44], "label": "-X-Y"},
]

# 真实环境中机器人建议初始位置（房间中心附近）
REAL_ROBOT_START = [0.0, 0.0, 0.8]   # [x, y, z]，单位米

