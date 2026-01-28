import requests

class AudioStream:
    def __init__(self, url):
        self.url = url

    def get_rms(self):
        try:
            data = requests.get(self.url, timeout=1).json()
            return float(data.get("sound", {}).get("rms", 0.0))
        except:
            return 0.0
