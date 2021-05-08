from typing import List
#########################################
from numpy import ndarray, asarray
from scipy.special import softmax
#########################################
from src.utils.feedback_alignment import FeedbackAlignment
from src.utils.extreme_learning_machine import ExtremeLearningMachine
from src.utils.neural_network import NeuralNetwork
#########################################

class D_ELM_IFA:
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
        self.RANDOM_WEIGHTS = NeuralNetwork.random_weights(
            input_layer_width=hidden_layer_width,
            output_layer_width=output_layer_width,
        )
        self.REPRESENTATION = "layers:\n[{input},{hidden},{output}]\n\nrandoms:\n{random_weights}".format(
            input=input_layer_width,
            hidden=','.join(map(lambda _:str(hidden_layer_width),range(number_of_hidden_layers))),
            output=output_layer_width,
            random_weights=self.RANDOM_WEIGHTS.shape
        )

    def __repr__(self) -> str:
        return self.REPRESENTATION + self.weights_representation(weights=self.LEARNT_WEIGHTS)
    
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
        uses indirect feedback alignment to project error backwards to earlier layers
        and then updates the weights at each layer using one-shot using extreme learning
        """
        self.indirect_feedback_alignment(
            desired_input_layer=inputs,
            desired_output_layer=asarray(expected_outputs)
        )

    def indirect_feedback_alignment(self,desired_input_layer:ndarray,desired_output_layer:ndarray) -> None:
        predicted_output_layer = self.infer(desired_input_layer)
        delta_output_layer = FeedbackAlignment.get_layer_error(
            layer_predicted=predicted_output_layer,
            layer_expected=desired_output_layer
        )
        predicted_hidden_layer_initial = NeuralNetwork.inverse_activation_function(
            layer=self.forward_pass(
                inputs=desired_input_layer,
                weights=self.LEARNT_WEIGHTS,
                depth=1,
                include_final_layer=False
            )
        ).T
        backward_projection_weights_from_outputs_to_hidden_initial = self.RANDOM_WEIGHTS
        delta_hidden_layer_initial = FeedbackAlignment.get_projected_error(
            layer=predicted_hidden_layer_initial,
            error=delta_output_layer.T,
            weights=backward_projection_weights_from_outputs_to_hidden_initial
        )
        desired_hidden_layer_initial = FeedbackAlignment.get_desired_layer(
            predicted_layer=predicted_hidden_layer_initial,
            delta_layer=delta_hidden_layer_initial
        )
        desired_weights_from_inputs_to_hidden_initial = ExtremeLearningMachine.learn_weights(
            output_layer=desired_hidden_layer_initial,
            input_layer=desired_input_layer.T,
        )
        
        learnt_weights = [desired_weights_from_inputs_to_hidden_initial.T]#self.LEARNT_WEIGHTS[0] = desired_weights_from_inputs_to_hidden_initial.T        
        delta_hidden_layer_depth_prior = delta_hidden_layer_initial
        predicted_hidden_layer_depth_prior = predicted_hidden_layer_initial#desired_hidden_layer_depth_prior = desired_hidden_layer_initial
        for depth in range(2,self.N_LAYERS+1):
            predicted_hidden_layer_depth = NeuralNetwork.inverse_activation_function(
                layer=self.forward_pass(
                    inputs=desired_input_layer,
                    weights=self.LEARNT_WEIGHTS,
                    depth=depth,
                    include_final_layer=False
                )
            ).T
            forward_projection_weights_from_hidden_depth_prior_to_hidden_depth = self.LEARNT_WEIGHTS[depth-1]
            delta_hidden_layer_depth = FeedbackAlignment.get_projected_error(
                layer=predicted_hidden_layer_depth,
                error=delta_hidden_layer_depth_prior,
                weights=forward_projection_weights_from_hidden_depth_prior_to_hidden_depth
            )
            desired_hidden_layer_depth = FeedbackAlignment.get_desired_layer(
                predicted_layer=predicted_hidden_layer_depth,
                delta_layer=delta_hidden_layer_depth
            )
            desired_weights_from_hidden_depth_prior_to_hidden_depth = ExtremeLearningMachine.learn_weights(
                output_layer=desired_hidden_layer_depth,
                input_layer=predicted_hidden_layer_depth_prior#desired_hidden_layer_depth_prior
            )
            learnt_weights.append(desired_weights_from_hidden_depth_prior_to_hidden_depth.T)#self.LEARNT_WEIGHTS[depth] = desired_weights_from_hidden_depth_prior_to_hidden_depth.T
            delta_hidden_layer_depth_prior = delta_hidden_layer_depth
            predicted_hidden_layer_depth_prior = predicted_hidden_layer_depth#desired_hidden_layer_depth_prior = desired_hidden_layer_depth

        desired_weights_from_hidden_final_to_outputs = ExtremeLearningMachine.learn_weights(
            output_layer=desired_output_layer.T,
            input_layer=predicted_hidden_layer_depth_prior#desired_hidden_layer_depth_prior,
        )
        learnt_weights.append(desired_weights_from_hidden_final_to_outputs.T)#self.LEARNT_WEIGHTS[-1] = desired_weights_from_hidden_final_to_outputs.T
        self.LEARNT_WEIGHTS = learnt_weights

    @staticmethod
    def weights_representation(weights:List[ndarray]) -> str:
        return "\n\nweights:\n{learnt_weights}".format(
            learnt_weights="\n".join(map(lambda weights:str(weights.shape), weights)),
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
        for depth in range(1,depth):
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
                range(1,number_of_hidden_layers)
            )
        ) + [
            NeuralNetwork.random_weights(
                input_layer_width=hidden_layer_width,
                output_layer_width=output_layer_width,
                seed_number=number_of_hidden_layers+1
            )
        ]