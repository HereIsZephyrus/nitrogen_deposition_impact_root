class SoilCalculator:
    def __init__(self):
        self.control = None

    def load_control(self, equation_path: str) -> None:
        pass

    def regression(self, data):
        if self.control is None:
            raise ValueError("Control equation not loaded")

    def predict(self, data):
        pass
