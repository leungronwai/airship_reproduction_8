# main.py

import argparse
import logging
import sys

from simulation.run_simulation import run_simulation

def parse_args():
    p = argparse.ArgumentParser(
        description="Airship Simulation"
    )
    p.add_argument(
        "-m", "--mode",
        choices=["debug", "release"],
        default="release",
        help="仿真模式：debug 会打印更多日志，release 只打印 INFO+"
    )
    p.add_argument(
        "-l", "--log-file",
        type=str,
        default=None,
        help="如果提供，日志也会写到这个文件"
    )
    p.add_argument(
        "-t", "--duration",
        type=float,
        default=None,
        help="可选：覆盖配置里的仿真总时长 T_SPAN（单位 s）"
    )
    return p.parse_args()

def setup_logger(mode: str, log_file: str = None):
    """
    根据模式和可选的文件路径，配置 root logger。
    debug 模式：DEBUG 级别；release 模式：INFO 级别
    """
    level = logging.DEBUG if mode == "debug" else logging.INFO

    # 创建 handler 列表：屏幕输出 + （可选）文件输出
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

    logger = logging.getLogger("main")
    logger.debug(f"Logger set to {mode.upper()} (level={level})")
    if log_file:
        logger.info(f"日志也写入：{log_file}")
    return logger

def main():
    args = parse_args()
    logger = setup_logger(args.mode, args.log_file)
    logger.info("程序启动")

    # 如果用户指定了 duration，就动态覆盖 config.parameters.T_SPAN
    if args.duration is not None:
        import config.parameters as params
        logger.info(f"覆盖仿真总时长：{params.T_SPAN} → {args.duration}")
        params.T_SPAN = args.duration

    # 启动仿真
    # run_simulation() 内部会使用 config.parameters 里的 DT、T_SPAN、X0 等
    run_simulation()

    logger.info("仿真结束")

if __name__ == "__main__":
    main()



'''
下面是一个更“工业化”的 main.py 样例，演示如何一步步加入：
	1.	CLI 参数解析（用 argparse）
	2.	多种仿真模式（debug vs release）
	3.	日志同时输出到屏幕和文件

你可以把它直接复制到项目根目录的 main.py，并根据需要做微调。


⸻

如何逐步扩展
	•	更多 CLI 参数
	•	加 --dt 来覆盖步长 DT
	•	加 --output-dir 指定结果图 / 数据的输出目录
	•	多种仿真模式
	•	debug：日志等级 DEBUG，并且可以在 run_simulation() 里针对 mode=="debug" 打开更详细的可视化、加速器、断点等
	•	release：默认 INFO 级别，画图时使用更简洁的样式
	•	日志文件管理
	•	使用 RotatingFileHandler 每次运行切分日志
	•	在文件名里加时间戳：log-$(date).txt
	•	配置热加载
	•	用 yaml/json 读一套更复杂的实验配置，而不仅仅是 parameters.py

这样你的 main.py 就既简洁又灵活，可随项目成长不断迭代。

'''


'''
那段更“工业化”的 main.py 其实就像大多数命令行工具一样，先用 argparse 定义了一些可选参数，然后根据你传入的参数来配置日志、覆盖仿真时长、启动仿真。下面分步给你讲明白怎么用：

⸻

1. 查看帮助信息

在终端（或 VSCode/PyCharm 的 Terminal）里，切换到项目根目录后，输入：

python main.py --help

你会看到类似这样的输出：

usage: main.py [-h] [-m {debug,release}] [-l LOG_FILE] [-t DURATION]

Airship Simulation

optional arguments:
  -h, --help            show this help message and exit
  -m {debug,release}, --mode {debug,release}
                        仿真模式：debug 会打印更多日志，release 只打印 INFO+
  -l LOG_FILE, --log-file LOG_FILE
                        如果提供，日志也会写到这个文件
  -t DURATION, --duration DURATION
                        可选：覆盖配置里的仿真总时长 T_SPAN（单位 s）

这就是所有可用参数的说明。

⸻

2. 默认运行

如果你直接敲：

python main.py

	•	使用默认模式 release（只打印 INFO 和更高级别的日志到屏幕）
	•	不写入日志文件
	•	不覆盖脚本里 config.parameters.T_SPAN（仿真时长用你在 parameters.py 里写的值）

⸻

3. 使用 debug 模式

在开发或调试时，你可能想看更详细的日志（DEBUG 级别），就加上 -m debug：

python main.py -m debug

这样会把日志等级调到 DEBUG，屏幕上会输出非常详细的内部信息（方便排错）。

⸻

4. 同时输出到日志文件

如果你想把日志也写到文件里，带上 -l 参数：

python main.py -l simulation.log

这条命令会把所有日志（INFO 及以上）同时打印到屏幕和 simulation.log 文件。
你也可以跟 debug 模式一起用：

python main.py -m debug -l debug.log



⸻

5. 临时覆盖仿真时长

假设你在 config/parameters.py 里默认 T_SPAN = 200 秒，但这次想只跑 50 秒，就加上 -t 50：

python main.py -t 50

或者连同日志文件一起：

python main.py -t 50 -l short_run.log

脚本运行时会在日志里打印：

INFO  覆盖仿真总时长：200 -> 50



⸻

小结
	•	python main.py：默认运行
	•	-m/--mode：切换 release（默认）或 debug
	•	-l/--log-file：指定一个文件路径，将日志写入该文件
	•	-t/--duration：覆盖 parameters.py 中的 T_SPAN，以秒为单位

你可以根据需要自由组合这些选项，或者用 python main.py --help 随时查看。这样就能够“调”得动你的 main.py 了。祝仿真愉快！


'''