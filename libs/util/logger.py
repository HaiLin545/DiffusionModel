import time


class Logger:
    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file:
            with open(self.log_file, "w") as f:
                f.truncate()

    def __call__(self, content):
        t = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(time.time()))
        content = f"{t} {content}"
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{content}\n")
        print(content)


def time_cost(start_time):
    sec = time.time() - start_time
    if sec < 2:
        return f"{sec:.2f}s"
    else:
        minu = int(sec // 60)
        h = int(minu // 60)
        sec = int(sec % 60)
        return f"{h:0>2d}:{minu:0>2d}:{sec:0>2d}"
