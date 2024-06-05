class AbstractModel:

    def load(self, model_path):
        raise NotImplementedError

    def predict(self, inputs):
        raise None
