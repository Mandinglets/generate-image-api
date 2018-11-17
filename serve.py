import flask
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model import Generator
import pickle

def get_model_api():
    generator = Generator()
    generator.build((1, 100))
    generator.load_weights('weights/gen_rmsprob_better.h5')

    def model_api(input_noise):
        result = generator.predict(input_noise)
        result = result * 0.5 + 0.5

        # Tanh gives Weird Stuff
        result[result > 1] = 1
        result[result < 0] = 0

        output_data = {
            "image": pickle.dumps(result.tostring())
        }

        return output_data

    return model_api
