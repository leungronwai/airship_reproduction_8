# main.py
import os
import logging
import traceback
from datetime import datetime
from simulation.run_simulation import run
from simulation.run_simulation import run_simulation, run_nmpc_simulation  # 引入两个仿真函数



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
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # 同时输出到终端 / Also output to terminal
        ]
    )
    return logging.getLogger(__name__)


    


def main():
    logger = setup_logger()
    logger.info("程序启动 / Program started")

    # ==== 控制仿真模式（手动切换）/ Control simulation mode (switch manually) ====
    simulation_mode = "nmpc"  # or "blf"  #选择仿真模式 / Choose simulation mode

    try:
        if simulation_mode == "blf":
            run_simulation()

        elif simulation_mode == "nmpc":
            run_nmpc_simulation()
        else:
            raise ValueError(f"未知的仿真模式 / unknown simulation mode: {simulation_mode}")
    except Exception as e:
        logger.error(f"仿真过程中发生错误: {e} / Error during simulation: {e}")
        traceback.print_exc() # 打印完整的错误堆栈 / Print the full error stack
    else:
        logger.info("仿真成功完成 / Simulation completed successfully") # 3. 结束日志 / End logging




if __name__ == '__main__':
    print("[INFO]: starting information ")
    main()





