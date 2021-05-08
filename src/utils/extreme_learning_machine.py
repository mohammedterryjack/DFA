from numpy import ndarray
from numpy.linalg import pinv
#########################################

class ExtremeLearningMachine:
    @staticmethod
    def learn_weights(input_layer:ndarray, output_layer:ndarray) -> ndarray:
        """
        this is the Extreme learning machine (ELM) method
        to fit weights in one shot (using the pseudoinverse)
        """
        print(f"""
        output_layer @ pinv(input_layer)
        {output_layer.shape} @ pinv({input_layer.shape})
        """)
        x = output_layer @ pinv(input_layer)
        print(f"={x.shape}")
        return x
