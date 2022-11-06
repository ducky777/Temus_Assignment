class WindPowerPipeline:
    def __init__(self, stack: list):
        """Takes a list of modules to form a pipeline"""
        self.stack = stack

    def predict(self, x):
        """Passes data through the pipeline"""
        result = []

        for i in self.stack:
            x = i(x)

        return x

    def __call__(self, x):
        """Returns predict()"""
        return self.predict(x)

