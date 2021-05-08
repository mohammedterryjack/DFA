from typing import List
#########################################
from numpy import ndarray
from scipy.special import softmax
#########################################
from src.utils.feedback_alignment import FeedbackAlignment
from src.utils.extreme_learning_machine import ExtremeLearningMachine
from src.utils.neural_network import NeuralNetwork
#########################################

class D_ELM_DFA:
    def __init__(
        self,
        input_layer_width:int,
        output_layer_width:int,
        hidden_layer_width:int=1000,
        number_of_hidden_layers:int=3
    ) -> None:
        self.N_LAYERS = number_of_hidden_layers
        self.LEARNT_WEIGHTS = self.generate_weights(
            input_layer_width=input_layer_width,
            output_layer_width=output_layer_width,
            hidden_layer_width=hidden_layer_width,
            number_of_hidden_layers=number_of_hidden_layers
        )
        self.RANDOM_WEIGHTS = self.generate_weights_random(
            input_layer_width=input_layer_width,
            output_layer_width=output_layer_width,
            hidden_layer_width=hidden_layer_width,
            number_of_hidden_layers=number_of_hidden_layers
        )
        self.REPRESENTATION = "layers:\n[{input},{hidden},{output}]\n\nweights:\n{learnt_weights}\n\nrandoms:\n{random_weights}".format(
            input=input_layer_width,
            hidden=','.join(map(lambda _:str(hidden_layer_width),range(number_of_hidden_layers))),
            output=output_layer_width,
            learnt_weights="\n".join(map(lambda weights:str(weights.shape), self.LEARNT_WEIGHTS)),
            random_weights="\n".join(map(lambda weights:str(weights.shape), self.RANDOM_WEIGHTS))
        )

    def __repr__(self) -> str:
        return self.REPRESENTATION
        
    def infer(self, inputs:ndarray) -> ndarray:
        return softmax(
            self.forward_pass(
                inputs=inputs,
                weights=self.LEARNT_WEIGHTS,
                depth=self.N_LAYERS
            )
        )
    
    def fit(self,inputs:ndarray,expected_outputs:ndarray) -> None:
        """
        one-shot extreme learning of hidden weights
        """
        self.LEARNT_WEIGHTS = self.backward_pass(
            inputs=inputs,
            expected_outputs=expected_outputs,
            learnt_weights=self.LEARNT_WEIGHTS,
            random_weights=self.RANDOM_WEIGHTS,
            n_layers=self.N_LAYERS,
        )
    
    @staticmethod
    def forward_pass(
        inputs:ndarray,
        weights:List[ndarray],
        depth:int,
        include_final_layer:bool=True
    ) -> ndarray:
        """
        passes forward through the layers
        if final_layer is False
        will not pass through final output layer
        """
        layer = NeuralNetwork.activation_function(
            layer= NeuralNetwork.forward_project_layer(
                layer=inputs,
                weights=weights[0]
            )
        )
        for depth in range(1,depth+1):
            layer = NeuralNetwork.activation_function(
                layer= NeuralNetwork.forward_project_layer(
                    layer=layer,
                    weights=weights[depth]
                )
            )
        return NeuralNetwork.forward_project_layer(
            layer=layer,
            weights=weights[depth+1]
        ) if include_final_layer else layer

    @staticmethod
    def backward_pass(
        inputs:ndarray, 
        expected_outputs:ndarray,
        learnt_weights:ndarray,
        random_weights:ndarray,
        n_layers:int,
    ) -> ndarray:

        predicted_output_logits = D_ELM_DFA.forward_pass(
            inputs=inputs,
            weights=learnt_weights,
            depth=n_layers
        )
        predicted_outputs = softmax(predicted_output_logits)
        predicted_input_layer = D_ELM_DFA.forward_pass(
            inputs=inputs,
            weights=learnt_weights,
            depth=n_layers-1
        )
        error = FeedbackAlignment.get_layer_error(
            layer_predicted=predicted_outputs,
            layer_expected=expected_outputs
        )
        error_prior = FeedbackAlignment.back_project_error(
            layer=predicted_input_layer,
            weights=random_weights[-1],
            error=error.T
        )
        desired_inputs = ExtremeLearningMachine.prior_output_layer(
            next_input_layer=predicted_input_layer,
            delta=error_prior
        )
        learnt_weights[-1] = ExtremeLearningMachine.learn_weights(
            output_layer=expected_outputs,
            input_layer=desired_inputs
        )

        for depth in range(n_layers-1,0,-1):
            expected_outputs = desired_inputs

            predicted_outputs = D_ELM_DFA.forward_pass(
                inputs=inputs,
                weights=learnt_weights,
                depth=depth,
                include_final_layer=False
            )
            predicted_output_logits = NeuralNetwork.inverse_activation_function(predicted_outputs)
            predicted_input_layer = D_ELM_DFA.forward_pass(
                inputs=inputs,
                weights=learnt_weights,
                depth=depth-1,
                include_final_layer=False
            )
            error_prior = FeedbackAlignment.back_project_error(
                layer=predicted_input_layer,
                weights=random_weights[depth],
                error=error.T
            )
            desired_inputs = ExtremeLearningMachine.prior_output_layer(
                next_input_layer=predicted_input_layer,
                delta=error_prior
            )
            learnt_weights[depth] = ExtremeLearningMachine.learn_weights(
                output_layer=expected_outputs,
                input_layer=desired_inputs
            )            
        return learnt_weights

    @staticmethod
    def generate_weights(
        input_layer_width:int,
        output_layer_width:int,
        hidden_layer_width:int,
        number_of_hidden_layers:int
    ) -> List[ndarray]:
        return [
            NeuralNetwork.random_weights(
                input_layer_width=input_layer_width,
                output_layer_width=hidden_layer_width,
                seed_number=0
            )
        ] + list(
            map(
                lambda depth:NeuralNetwork.random_weights(
                    input_layer_width=hidden_layer_width,
                    output_layer_width=hidden_layer_width,
                    seed_number=depth
                ),
                range(1,number_of_hidden_layers+1)
            )
        ) + [
            NeuralNetwork.random_weights(
                input_layer_width=hidden_layer_width,
                output_layer_width=output_layer_width,
                seed_number=number_of_hidden_layers+1
            )
        ]

    @staticmethod
    def generate_weights_random(
        input_layer_width:int,
        output_layer_width:int,
        hidden_layer_width:int,
        number_of_hidden_layers:int
    ) -> List[ndarray]:
        return [
            NeuralNetwork.random_weights(
                input_layer_width=input_layer_width,
                output_layer_width=hidden_layer_width,
                seed_number=0
            )
        ] + list(
            map(
                lambda depth:NeuralNetwork.random_weights(
                    input_layer_width=hidden_layer_width,
                    output_layer_width=output_layer_width,
                    seed_number=depth
                ),
                range(1,number_of_hidden_layers+2)
            )
        ) 