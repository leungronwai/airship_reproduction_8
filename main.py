# main.py
import os
import logging
from datetime import datetime
from simulation.run_simulation import run_simulation


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
    # 1. 初始化日志 /  Initialize logger
    logger = setup_logger() # test
    logger.info("程序启动 / Program started")

    # 2. 调用仿真主入口（内部已使用 config.parameters 拿到所有常量）
    # Call the main simulation entry point (internally uses config.parameters to access all constants)
    try:
        run_simulation()
    except Exception as e:
        logger.error(f"仿真过程中发生错误: {e} / Error during simulation: {e}")
    else:
        logger.info("仿真成功完成 / Simulation completed successfully") # 3. 结束日志 / End logging
    



if __name__ == '__main__':
    print("[INFO]: starting information ")
    main()





'''
我想 逐步丰富 main.py, 例如:
	•	支持 CLI 参数(用 argparse)
	•	支持不同的仿真模式（调试/发布）
	•	将日志输出到文件等

Ideas to gradually enhance main.py, for example:
    • Support CLI arguments (using argparse)
    • Support different simulation modes (debug/release)
    • Output logs to a file, etc.

'''