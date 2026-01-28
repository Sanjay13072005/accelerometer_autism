class RuleBasedSleepDetector:
    def __init__(self, sleep_th, wake_th, min_sleep_windows):
        self.sleep_th = sleep_th
        self.wake_th = wake_th
        self.min_sleep_windows = min_sleep_windows
        self.sleep_counter = 0
        self.state = "AWAKE"

    def update(self, mean_motion):
        if mean_motion < self.sleep_th:
            self.sleep_counter += 1
            if self.sleep_counter >= self.min_sleep_windows:
                self.state = "SLEEP"
        elif mean_motion > self.wake_th:
            self.sleep_counter = 0
            self.state = "WAKE"

        return self.state
