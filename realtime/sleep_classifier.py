import joblib

class SleepClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, feats):
        X = [[
            feats["mean"],
            feats["std"],
            feats["energy"],
            feats["max"]
        ]]
        return int(self.model.predict(X)[0])
   