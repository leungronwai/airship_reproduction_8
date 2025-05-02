# main.py

import logging
from simulation.run_simulation import run_simulation

def setup_logger():
    """
    全局日志配置：只需在这里配置一次，项目中其他模块获取同一 logger 即可。
    Global logging configuration: Configure once here, and other modules in the project
    can retrieve the same logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    # 1. 初始化日志 /  Initialize logger
    logger = setup_logger() # test
    logger.info("程序启动")

    # 2. 调用仿真主入口（内部已使用 config.parameters 拿到所有常量）
    # Call the main simulation entry point (internally uses config.parameters to access all constants)
    run_simulation()

    # 3. 结束日志 / End logging
    logger.info("仿真结束")

if __name__ == '__main__':
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