"""
# main.py

"""
# pylint: disable=invalid-name
# cspell:ignore dompc levelname figsize traj

import sys
import os
import logging

from datetime import datetime
from simulation.run_simulation import run_simulation
from simulation.run_dompc_simulation import run_dompc_simulation



# 添加项目根目录到 sys.path / Add project root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)




def setup_logger():
    """
    全局日志配置：只需在这里配置一次，项目中其他模块获取同一 logger 即可。
    Global logging configuration: Configure once here, and other modules in the project
    can retrieve the same logger.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # 创建 logs 目录 / Create logs directory if it doesn't exist

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"simulation_{timestamp}.log")

    # 配置日志格式 / Configure log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 配置日志 / Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),  # 文件处理器 / File handler
            logging.StreamHandler(sys.stdout)  # 控制台处理器 / Console handler
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成 / Logging system initialized")
    logger.info("日志文件：%s", log_filename)
    return logger


def main():
    """主程序入口"""
    logger = setup_logger()

    try:
        logger.info("=== 气艇轨迹跟踪仿真开始 ===")

        # 运行选择菜单
        print("\n=== 气艇仿真控制器选择 ===")
        print("1. 传统 PID 控制器")
        print("2. do-mpc Simulator 完整仿真 (推荐)")


        choice = input("请选择控制器类型 (1-2): ").strip()

        # 轨迹选择
        print("\n=== 轨迹类型选择 ===")
        print("1. 直线轨迹 (linear)")
        print("2. 螺旋轨迹 (spiral)")
        print("3. 8 字轨迹 (figure8)")
        print("4. 莱洛曲线 (lemniscate)")

        traj_choice = input("请选择轨迹类型 (1-4): ").strip()

        # 轨迹映射
        trajectory_map = {
            "1": "linear",
            "2": "spiral",
            "3": "figure8",
            "4": "lemniscate"
        }

        trajectory_type = trajectory_map.get(traj_choice, "linear")
        logger.info("选择的轨迹类型：%s", trajectory_type)

        # 根据选择运行相应的仿真
        if choice == "1":
            logger.info("运行传统 PID 控制器仿真")

            run_simulation(trajectory_type=trajectory_type)

        elif choice == "2":
            logger.info("运行 do-mpc NMPC 控制器仿真")
            run_dompc_simulation(
                trajectory_type=trajectory_type,
                use_disturbance_compensation=True,
                use_simulator=False  # 仅使用 MPC 控制器
            )

        else:
            logger.warning("无效选择，运行默认的 do-mpc Simulator 仿真")

            run_dompc_simulation(
                trajectory_type="linear",
                use_disturbance_compensation=True
            )

    except KeyboardInterrupt:
        logger.info("用户中断程序")
    finally:
        logger.info("=== 仿真程序结束 ===")


def _run_comparison_simulation(trajectory_type):
    """运行控制器对比仿真"""
    logger = logging.getLogger(__name__)

    print(f"\n=== 开始对比仿真 (轨迹：{trajectory_type}) ===")

    results = {}

    # 1. 运行传统 PID 控制器
    try:
        print("1/2 运行传统 PID 控制器...")
        logger.info("开始 PID 控制器仿真")
        run_simulation(trajectory_type=trajectory_type)
        results['PID'] = "成功"
        print("PID 控制器仿真完成")
    except KeyboardInterrupt:
        logger.info("用户中断程序 / User interrupted the program")
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("程序执行出错：%s", e)
    except Exception as e:     # pylint: disable=broad-except
        logger.exception("未处理的异常：%s", e)

    # 2. 运行 do-mpc NMPC 控制器
    try:
        print("2/2 运行 do-mpc NMPC 控制器...")
        logger.info("开始 do-mpc NMPC 控制器仿真")
        run_dompc_simulation(
            trajectory_type=trajectory_type,
            use_disturbance_compensation=True
        )
        results['do-mpc NMPC'] = "成功"
        print("do-mpc NMPC 控制器仿真完成")
    except Exception as e:      # pylint: disable=broad-except
        logger.error("do-mpc NMPC 控制器仿真失败：%s", e)
        results['do-mpc NMPC'] = f"失败：{e}"
        print("do-mpc NMPC 控制器仿真失败")

    # 汇总结果
    print("\n=== 对比仿真结果汇总 ===")
    for controller, result in results.items():
        print(f"{controller}: {result}")

    logger.info("对比仿真完成，结果：%s", results)


if __name__ == "__main__":
    main()
