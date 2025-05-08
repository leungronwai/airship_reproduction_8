'''
# main.py

'''
import sys
import os
import logging
import traceback
from datetime import datetime
from simulation.run_simulation import run_simulation, run_nmpc_simulation



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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")

    logging.basicConfig(
        level=logging.DEBUG,  # 可以调整级别 / Adjust the level as needed
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],  # Also output to terminal
    )
    return logging.getLogger(__name__)




def main():
    """
    主函数
    """
    logger = setup_logger()
    logger.info("程序启动 / Program started")

    # ==== 控制仿真模式和轨迹类型（手动切换）/ Control simulation mode and trajectory type ====
    simulation_mode = "nmpc"  # 控制器选择："blf" 或 "nmpc"
    trajectory_type = "linear"  # 轨迹选择："default", "spiral", "figure8", "lemniscate", "linear"
    use_disturbance_compensation = True  # 是否使用扰动补偿

    try:
        if simulation_mode == "blf":
            run_simulation(trajectory_type=trajectory_type)
        elif simulation_mode == "nmpc":
            run_nmpc_simulation(use_disturbance_compensation=use_disturbance_compensation)
        else:
            raise ValueError(f"未知的仿真模式 / unknown simulation mode: {simulation_mode}")
    except Exception as e:
        logger.error("仿真过程中发生错误：%s", e)
        traceback.print_exc()  # 打印完整的错误堆栈 / Print the full error stack
    else:
        logger.info("仿真成功完成 / Simulation completed successfully")  # 3. 结束日志 / End logging


if __name__ == "__main__":
    print("[INFO]: starting information ")
    main()
