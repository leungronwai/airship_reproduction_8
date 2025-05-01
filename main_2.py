# main.py

import argparse
import logging
import sys
import os
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler

# 可选支持 yaml
try:
    import yaml
except ImportError:
    yaml = None

from simulation.run_simulation import run_simulation


def parse_args():
    p = argparse.ArgumentParser(description="Airship Simulation")
    p.add_argument(
        "-m", "--mode",
        choices=["debug", "release"],
        default="release",
        help="仿真模式：debug 会打印 DEBUG 日志，release 只打印 INFO+"
    )
    p.add_argument(
        "-l", "--log-file",
        type=str,
        default=None,
        help="日志写入文件，如果不指定，则只打印到控制台"
    )
    p.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="可选：外部配置文件路径（.yaml 或 .json），用来覆盖 parameters.py 中的默认常量"
    )
    p.add_argument(
        "--dt",
        type=float,
        default=None,
        help="可选：覆盖仿真步长 DT（单位：秒）"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="可选：仿真结果（图表/数据）的输出目录"
    )
    return p.parse_args()


def setup_logger(mode: str, log_file: str = None):
    """配置 root logger：console + （可选）文件输出 & RotatingFileHandler"""
    level = logging.DEBUG if mode == "debug" else logging.INFO

    # Console handler
    handlers = [logging.StreamHandler(sys.stdout)]

    # 如果指定了文件输出，就加一个滚动切分的 handler
    if log_file:
        # 在文件名里自动加上时间戳
        base, ext = os.path.splitext(log_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{base}_{timestamp}{ext or '.log'}"
        fh = RotatingFileHandler(
            fname,
            maxBytes=5 * 1024 * 1024,  # 5 MB 每个日志文件
            backupCount=3,             # 保留 3 份历史
            encoding="utf-8"
        )
        handlers.append(fh)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    logger = logging.getLogger("main")
    logger.debug(f"Logger initialized in {mode.upper()} mode")
    if log_file:
        logger.info(f"Logging to file: {fh.baseFilename}")
    return logger


def apply_external_config(path: str, logger: logging.Logger):
    """读取 yaml/json 配置，动态写入到 config.parameters 模块"""
    if not os.path.isfile(path):
        logger.error(f"配置文件不存在：{path}")
        return

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            if yaml is None:
                logger.error("要使用 YAML 配置，需要先安装 pyyaml：pip install pyyaml")
                return
            cfg = yaml.safe_load(f)
        elif ext == ".json":
            cfg = json.load(f)
        else:
            logger.error("只支持 .yaml/.yml/.json 格式的配置文件")
            return

    import config.parameters as params
    for key, val in cfg.items():
        if hasattr(params, key):
            setattr(params, key, val)
            logger.info(f"参数覆盖: {key} = {val!r}")
        else:
            logger.warning(f"未知参数跳过: {key}")

    logger.info(f"外部配置 {path} 已加载")


def main():
    args = parse_args()

    # 1. 日志初始化
    logger = setup_logger(args.mode, args.log_file)
    logger.info("=== 程序启动 ===")

    # 2. 外部配置覆盖
    if args.config:
        apply_external_config(args.config, logger)

    # 3. 覆盖单个常量：DT
    if args.dt is not None:
        import config.parameters as params
        old = params.DT
        params.DT = args.dt
        logger.info(f"覆盖 DT: {old} -> {params.DT}")

    # 4. 创建输出目录（并写入 parameters）
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        import config.parameters as params
        params.OUTPUT_DIR = args.output_dir  # run_simulation 里可读取
        logger.info(f"输出目录: {args.output_dir}")

    # 5. 启动仿真
    run_simulation()

    logger.info("=== 仿真结束 ===")


if __name__ == "__main__":
    main()



# 然后你这样在终端跑：
# python main.py --config settings.yaml



'''
下面是一个增强版的 main.py，演示如何一步到位实现：
	•	更多 CLI 参数：--dt、--output-dir、--config
	•	多种仿真模式：debug vs release
	•	日志同时输出到屏幕、文件，并使用滚动切分（RotatingFileHandler）
	•	热加载外部 yaml 或 json 配置，动态覆盖 config.parameters 中的常量

把它放到项目根目录（和 airship/、config/、simulation/ 同级），并确保安装了 pyyaml（若要读 .yaml）：



⸻

功能逐条解读
	1.	argparse 定义 CLI 接口
	•	--mode：切换 debug（DEBUG 级别）或 release（INFO 级别）。
	•	--log-file：日志写入文件，如果提供，则自动附加时间戳并滚动切分。
	•	--config：指定一个 .yaml 或 .json，用来一次性覆盖 config/parameters.py 里的默认值。
	•	--dt：只想临时改步长，就用这个。
	•	--output-dir：仿真生成的图、数据要存哪。
	2.	日志配置
	•	StreamHandler：把日志打到终端。
	•	RotatingFileHandler：如果指定了 --log-file，就把日志写到时间戳文件里，自动切分（5 MB/文件，最多保留 3 个历史）。
	3.	热加载外部配置
	•	根据文件后缀决定走 yaml.safe_load（需安装 PyYAML）还是 json.load。
	•	逐条覆盖 config.parameters 模块中已知的属性（其余跳过，并发出警告）。
	4.	单参数覆盖
	•	如果只想改步长，用 --dt，直接给 params.DT 赋新值。
	5.	输出目录
	•	自动创建目标文件夹并写入 params.OUTPUT_DIR，你的 run_simulation() 里可以读它，把图 plt.savefig(os.path.join(OUTPUT_DIR, ...))。
	6.	统一入口
	•	run_simulation() 本身不带参数，从模块里直接读 config.parameters。
	•	if __name__=="__main__" 确保：
	•	直接 python main.py 会跑仿真；
	•	导入 这个模块时（比如未来做单元测试），不会立刻执行仿真。

这样，你的 main.py 就灵活、可配置且“工业化”了：
	•	日常调试只要 python main.py -m debug
	•	生产环境可 python main.py -c my_settings.yaml --output-dir results/ -l app.log
	•	参数维护依旧在 config/parameters.py 或外部 config 文件里搞定。

'''



'''
三、注意事项
	1.	属性名要一一对应
	•	外部文件中的顶层键（key）要和 parameters.py 里那些常量名完全一致（区分大小写）。
	•	否则会被跳过并给你一个 warning。
	2.	数据类型要对
	•	YAML/JSON 里读出来的数字、列表、字典要和 parameters.py 对应属性原来的类型兼容。
	•	例如你在 parameters.py 定义 DT = 0.05（float），就别在 YAML 里写成字符串 "0.1"，而要写成 DT: 0.1（不带引号）。
	3.	导入时机
	•	一定要在“真正用参数”之前调用 apply_external_config()，否则仿真里拿到的还是老值。
	•	推荐在 main() 一开始、run_simulation() 之前完成覆盖。
	4.	YAML 支持可选
	•	如果项目没装 pyyaml，也能读 .json。装了再读 .yaml。
	•	安装：pip install pyyaml。
	5.	不要在模块顶层改
	•	一定首先 import config.parameters，然后在函数里用 setattr。这样只有当你主动调用 apply_external_config() 时才会改，不会影响到你直接在 REPL 或者单元测试时无意中改动。


'''