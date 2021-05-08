from scipy.special import expit
#########################################
from numpy import ndarray
from numpy.random import seed, uniform
#########################################


class NeuralNetwork:
    @staticmethod
    def random_weights(input_layer_width:int,output_layer_width:int,seed_number:int=0) -> ndarray:
        seed(seed_number)
        return uniform(
            low=-.1, high=.1, 
            size =[input_layer_width, output_layer_width]
        )

    @staticmethod
    def forward_project_layer(layer:ndarray,weights:ndarray) -> ndarray:
        """
        get the next layer (before the activation function)
        """
        return layer @ weights
    
    @staticmethod
    def activation_function(layer:ndarray) -> ndarray:
        """
        sigmoid function
        """
        return expit(layer)
    
    @staticmethod
    def inverse_activation_function(layer:ndarray) -> ndarray:
        """
        differential of sigmoid function
        f'(x) = f(x)(1-f(x))
        """
        return expit(layer) * (
            1 - expit(layer)
        )
