import time
from datetime import datetime
from logging import (
    DEBUG,
    INFO,
    FileHandler,
    Formatter,
    Logger,
    StreamHandler,
    getLogger,
)

from pytz import timezone

LOG_FILEPATH = "penguinml.log"
LOG_NAMES = set()
START_SEC = None


def init_logger(filepath: str) -> None:
    """ロガーを初期化する

    Parameters
    ----------
    filepath : str
        ログファイル名
    """
    global LOG_FILEPATH, START_SEC

    LOG_FILEPATH = filepath
    START_SEC = time.time()


def get_logger(name: str) -> Logger:
    """ロガーを取得する

    Parameters
    ----------
    name : str
        ロガー名

    Returns
    -------
    logger : Logger
        ロガー
    """
    global LOG_NAMES

    logger = getLogger(name)
    if name in LOG_NAMES:
        return logger

    LOG_NAMES.add(name)
    logger = _set_handler(logger, StreamHandler(), False)
    logger = _set_handler(logger, FileHandler(LOG_FILEPATH), True)
    logger.setLevel(DEBUG)
    logger.propagate = False
    return logger


def customTime(*args):
    return datetime.now(timezone("Asia/Tokyo")).timetuple()


class CustomFromatter(Formatter):
    def converter(self, *args):
        """timezoneをJSTにする"""
        return datetime.now(timezone("Asia/Tokyo")).timetuple()

    def format(self, record):
        """メッセージに経過時間を追加"""
        elapsed_seconds = time.time() - START_SEC
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        record.elapsed = f"({int(hours):02}:{int(minutes):02}:{int(seconds):02})"
        return super(CustomFromatter, self).format(record)


def _set_handler(logger, handler, verbose):
    if verbose:
        handler.setLevel(DEBUG)
        formatter = CustomFromatter(
            fmt="%(asctime)s %(elapsed)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        handler.setLevel(INFO)
        formatter = CustomFromatter(fmt="%(message)s")

    formatter.converter = customTime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
