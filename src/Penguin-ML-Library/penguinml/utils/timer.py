import time


class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed_sec(self) -> float:
        """経過時間を秒で返す

        Returns
        -------
        elapsed_sec : float
            経過時間（秒）
        """
        return time.time() - self.start

    def time_str(self) -> str:
        """経過時間の hh:mm:ss 形式の文字列を返す

        Returns
        -------
        time_str : str
            経過時間の hh:mm:ss 形式の文字列
        """

        elapsed_sec = self.elapsed_sec()
        h = int(elapsed_sec // 3600)
        m = int((elapsed_sec % 3600) // 60)
        s = int(elapsed_sec % 60)
        time_str = f"{h:02d}:{m:02d}:{s:02d}"
        return time_str
