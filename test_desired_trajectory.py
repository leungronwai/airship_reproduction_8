import numpy as np
import matplotlib.pyplot as plt
from airship.trajectory import Trajectory
import argparse

def plot_trajectory(trajectory_type="all", simulation_time=200, num_points=1000):
    """
    绘制指定类型的轨迹
    
    参数:
        trajectory_type: 轨迹类型，可选 "spiral", "figure8", "lemniscate", "linear", "all"
        simulation_time: 仿真总时间(s)
        num_points: 仿真点数
    """
    # 创建时间序列
    t_values = np.linspace(0, simulation_time, num_points)
    
    # 创建轨迹对象
    trajectory = Trajectory()
    
    # 设置图形布局
    plt.figure(figsize=(18, 10))
    
    # 使用2行2列的布局
    if trajectory_type == "all" or trajectory_type == "spiral":
        plot_specific_trajectory(trajectory, t_values, "spiral", 221)
    
    if trajectory_type == "all" or trajectory_type == "figure8":
        plot_specific_trajectory(trajectory, t_values, "figure8", 222)
    
    if trajectory_type == "all" or trajectory_type == "lemniscate":
        plot_specific_trajectory(trajectory, t_values, "lemniscate", 223)
    
    if trajectory_type == "all":
        plot_linear_trajectory(trajectory, t_values, 224)
        # 线性轨迹需要特殊处理，因为它接受额外参数
    elif trajectory_type == "linear":
        plot_linear_trajectory(trajectory, t_values, 111)
    
    
    plt.tight_layout()
    plt.show()

def plot_specific_trajectory(trajectory, t_values, trajectory_type, subplot_position):
    """
    绘制特定类型的轨迹
    
    参数:
        trajectory: Trajectory对象
        t_values: 时间序列
        trajectory_type: 轨迹类型
        subplot_position: 子图位置
    """
    # 初始化存储轨迹数据的数组
    x_values, y_values, z_values = [], [], []
    
    # 根据轨迹类型获取相应的方法
    if trajectory_type == "spiral":
        get_trajectory_method = trajectory.get_spiral_trajectory
    elif trajectory_type == "figure8":
        get_trajectory_method = trajectory.get_figure8_trajectory
    elif trajectory_type == "lemniscate":
        get_trajectory_method = trajectory.get_lemniscate_trajectory
    else:
        raise ValueError(f"未知的轨迹类型: {trajectory_type}")
    
    # 计算每个时间点的位置
    for t in t_values:
        yc, _, _, _, _ = get_trajectory_method(t)
        pos = yc[0:3]  # 提取位置数据
        x_values.append(pos[0])
        y_values.append(pos[1])
        z_values.append(pos[2])
    
    # 绘制3D轨迹
    ax = plt.subplot(subplot_position, projection="3d")
    ax.plot(x_values, y_values, z_values, label=f"{trajectory_type} Trajectory")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title(f"{trajectory_type.capitalize()} Trajectory")
    ax.legend()
    
    # 标记起点和终点
    ax.scatter(x_values[0], y_values[0], z_values[0], color='green', s=100, label='Start')
    ax.scatter(x_values[-1], y_values[-1], z_values[-1], color='red', s=100, label='End')

def plot_linear_trajectory(trajectory, t_values, subplot_position):
    """
    绘制直线轨迹
    
    参数:
        trajectory: Trajectory对象
        t_values: 时间序列
        subplot_position: 子图位置
    """
    # 初始化存储轨迹数据的数组
    x_values, y_values, z_values = [], [], []
    
    # 设置直线轨迹的起点和终点
    start_point = np.array([0.0, 0.0, -19000.0]) 
    end_point = np.array([5000.0, 5000.0, -19000.0])
    
    # 计算每个时间点的位置
    for t in t_values:
        yc, _, _, _, _ = trajectory.get_linear_trajectory(
            t, 
            start_point=start_point,
            end_point=end_point,
            speed=15.0,  # 飞行速度 15 m/s
            hover_at_end=True  # 到达终点后悬停
        )
        pos = yc[0:3]  # 提取位置数据
        x_values.append(pos[0])
        y_values.append(pos[1])
        z_values.append(pos[2])
    
    # 绘制3D轨迹
    ax = plt.subplot(subplot_position, projection="3d")
    ax.plot(x_values, y_values, z_values, label="Linear Trajectory")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("Linear Trajectory")
    ax.legend()
    
    # 标记起点和终点
    ax.scatter(start_point[0], start_point[1], start_point[2], color='green', s=100, label='Start')
    ax.scatter(end_point[0], end_point[1], end_point[2], color='red', s=100, label='End')

if __name__ == "__main__":
    # 添加命令行参数
    parser = argparse.ArgumentParser(description="绘制气艇轨迹")
    parser.add_argument("--type", type=str, default="all",
                      choices=["spiral", "figure8", "lemniscate", "linear", "all"],
                      help="要绘制的轨迹类型")
    parser.add_argument("--time", type=float, default=200.0,
                      help="仿真总时间(s)")
    parser.add_argument("--points", type=int, default=1000,
                      help="仿真点数")
    
    args = parser.parse_args()
    
    # 绘制轨迹
    plot_trajectory(args.type, args.time, args.points)




''' 
# 绘制所有轨迹
python test_desired_trajectory.py --type all

# 只绘制8字形轨迹
python test_desired_trajectory.py --type figure8 --time 2500 --points 2000

# 只绘制莱洛曲线轨迹
python test_desired_trajectory.py --type lemniscate

# 只绘制直线轨迹
python test_desired_trajectory.py --type linear

# 只绘制螺旋轨迹
python test_desired_trajectory.py --type spiral

# 自定义仿真时间和点数
python test_desired_trajectory.py --type all --time 100 --points 500
    
    
'''