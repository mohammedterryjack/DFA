############   NATIVE IMPORTS  ###########################
from typing import List, Tuple
############ INSTALLED IMPORTS ###########################
from numpy import ndarray, exp, asarray, argmax
from numpy.linalg import pinv
from numpy.random import seed, uniform
from sklearn.metrics import log_loss
############   LOCAL IMPORTS   ###########################
##########################################################
class ELMClassifier:
    def __init__(
        self, 
        input_layer_width: int, 
        output_layer_width:int,
        hidden_layer_width:int = 1000,
        number_of_hidden_layers:int=2
    ) -> None:
    
        seed(0) 
        self.activation_function = self.sigmoid
        self.N_HIDDEN_LAYERS = number_of_hidden_layers
        self.WEIGHTS = self.get_weights(
            input_layer_width=input_layer_width,
            hidden_layer_width=hidden_layer_width,
            output_layer_width=output_layer_width,
            number_of_hidden_layers=number_of_hidden_layers
        )
        self.RANDOM_WEIGHTS = self.get_random_weights(
            input_layer_width=input_layer_width,
            hidden_layer_width=hidden_layer_width,
            output_layer_width=output_layer_width,
            number_of_hidden_layers=number_of_hidden_layers
        )
        self.ERROR_WEIGHTS = self.random_weights(hidden_layer_width,output_layer_width)
        self.REPRESENTATION = "layers:\n[{input},{hidden},{output}]".format(
            input=input_layer_width,
            hidden=','.join(map(lambda _:str(hidden_layer_width),range(number_of_hidden_layers))),
            output=output_layer_width,
        )

    def __repr__(self) -> str:
        return self.REPRESENTATION + self.weights_representation(weights=self.WEIGHTS)

    def infer(self, inputs: ndarray) -> ndarray:
        layers,_= self.forward_pass(inputs)
        return layers[-1]

    def forward_pass(self,inputs:ndarray) -> Tuple[List[ndarray],List[ndarray]]:
        hidden_layers_activated,hidden_layers_logits = self._hidden_layers(inputs)
        predicted_outputs = self._output_layer(hidden=hidden_layers_activated[-1])
        hidden_layers_activated.append(predicted_outputs)
        return hidden_layers_activated,hidden_layers_logits

    def fit_final_weights(self, inputs: ndarray, outputs: ndarray) -> None:
        """
        use Extreme Learning Machine's pseudoinverse method
        to fit final weights (keep other weights as random projections)
        (Reservoir Learning)
        """
        hidden_layers,_ = self._hidden_layers(inputs)
        self.WEIGHTS[-1] = self.extreme_learn_weights(
            layer_before=hidden_layers[-1], 
            layer_after=outputs
        )
    
    def indirect_feedback_alignment(self,inputs:ndarray,outputs:ndarray,epochs:int,learning_rate:float=1e-6) -> None:
        for _ in range(epochs):
            self.indirect_feedback_alignment_step(
                desired_inputs=asarray(inputs),
                desired_outputs=asarray(outputs),
                learning_rate=learning_rate
            )

    def update_weights(self, delta_weights:List[ndarray],learning_rate:float) -> None:
        for depth in range(self.N_HIDDEN_LAYERS):
            self.WEIGHTS[depth+1] += learning_rate*delta_weights[depth]

    def direct_feedback_alignment_step(self, desired_inputs:ndarray, desired_outputs:ndarray, learning_rate:float) -> Tuple[float,float]:    
        hidden_layers, unactivated_layers = self.forward_pass(desired_inputs)
        predicted_outputs = hidden_layers.pop()

        self.update_weights(
            delta_weights=self.get_delta_weights_DFA(
                error=predicted_outputs - desired_outputs, 
                hidden_layers= [desired_inputs] + hidden_layers, 
                random_weights=self.RANDOM_WEIGHTS, 
                unactivated_layers=unactivated_layers, 
                desired_inputs=desired_inputs,
                n_layers=self.N_HIDDEN_LAYERS
            ),
            learning_rate=learning_rate
        )
        return self.get_training_error(predicted_outputs,desired_outputs), log_loss(desired_outputs, predicted_outputs)

    def direct_feedback_alignment(self, inputs:ndarray, outputs:ndarray, epochs:int=100,learning_rate:float=1e-4) -> None:        
        desired_inputs = asarray(inputs)
        desired_outputs = asarray(outputs)
        for epoch in range(epochs):
            error,loss = self.direct_feedback_alignment_step(desired_inputs,desired_outputs,learning_rate)
            if not(epoch%10):
                print(f"""
                Epoch {epoch} 
                Loss : {loss}
                Training Error : {error}
                """)


    def one_shot_extreme_deep_learning(self, desired_inputs:ndarray,desired_outputs:ndarray) -> None:
        """
        novel method:
        Uses ELM to fine-tune last layer
        Keeps all hidden layers random
        Novel part: But also finds error of first layer using error projection from IFA
                and then uses that to perform ELM on the first layer too
                resulting in a superior performance than ELM or IFA alone
        """
        predicted_outputs, predicted_hidden_layers = self.forward_pass(desired_inputs)
        hidden_layer = predicted_hidden_layers[0]
        delta_outputs = desired_outputs - predicted_outputs[-1]
        delta_hidden_layer = self.projected_error(
            layer=hidden_layer.T,
            error=delta_outputs.T,
            weights=self.ERROR_WEIGHTS
        ).T
        self.WEIGHTS[0] = self.extreme_learn_weights(
            layer_before=desired_inputs,
            layer_after= hidden_layer - delta_hidden_layer
        )
        self.fit_final_weights(desired_inputs,desired_outputs)

    def one_shot_extreme_deep_learning_experimental(self, desired_inputs:ndarray,desired_outputs:ndarray) -> None:
        predicted_outputs, predicted_hidden_layers = self.forward_pass(desired_inputs)
        hidden_layer = predicted_hidden_layers[0]
        delta_outputs = desired_outputs - predicted_outputs[-1]
        delta_hidden_layer = self.projected_error(
            layer=hidden_layer.T,
            error=delta_outputs.T,
            weights=self.ERROR_WEIGHTS
        ).T
        desired_hidden_layer = hidden_layer - delta_hidden_layer
        self.WEIGHTS[0] = self.extreme_learn_weights(
            layer_before=desired_inputs,
            layer_after=desired_hidden_layer
        )

        # delta_hidden_layer_prior = delta_hidden_layer
        # for hidden_layer_depth in range(1,self.N_HIDDEN_LAYERS):
        #     hidden_layer = predicted_hidden_layers[hidden_layer_depth]
        #     delta_hidden_layer_prior = self.projected_error(
        #         layer=predicted_hidden_layers[hidden_layer_depth].T,
        #         error=delta_hidden_layer_prior.T,
        #         weights=self.WEIGHTS[hidden_layer_depth]
        #     ).T                
        #     delta_hidden_layer = predicted_outputs[hidden_layer_depth-1].T @ delta_hidden_layer_prior
        #     desired_hidden_layer = hidden_layer - delta_hidden_layer
        #     self.WEIGHTS[hidden_layer_depth] = self.extreme_learn_weights(
        #         layer_before=desired_hidden_layer_prior,
        #         layer_after= desired_hidden_layer
        #     )

        self.fit_final_weights(desired_inputs,desired_outputs)


    def indirect_feedback_alignment_step(self, desired_inputs:ndarray,desired_outputs:ndarray,learning_rate:float) -> None:
        predicted_outputs, predicted_hidden_layers = self.forward_pass(desired_inputs)
        predicted_hidden_layers_transposed = list(map(lambda layer:layer.T,predicted_hidden_layers))
        predicted_outputs_transposed = list(map(lambda layer:layer.T,predicted_outputs))

        delta_outputs = desired_outputs - predicted_outputs[-1]
        delta_hidden_layer = self.projected_error(
            layer=predicted_hidden_layers_transposed[0],
            error=delta_outputs.T,
            weights=self.ERROR_WEIGHTS
        ).T
        updated_weights = self.WEIGHTS[0] - learning_rate* (
            desired_inputs.T @ delta_hidden_layer
        )
        new_weights = [updated_weights]

        for hidden_layer_depth in range(1,self.N_HIDDEN_LAYERS):
            delta_hidden_layer = self.projected_error(
                layer=predicted_hidden_layers_transposed[hidden_layer_depth],
                error=delta_hidden_layer.T,
                weights=self.WEIGHTS[hidden_layer_depth]
            ).T                
            updated_weights = self.WEIGHTS[hidden_layer_depth] - learning_rate* (
                predicted_outputs_transposed[hidden_layer_depth-1] @ delta_hidden_layer
            )
            new_weights.append(updated_weights)
        updated_weights = self.WEIGHTS[-1] - learning_rate* (
            predicted_outputs_transposed[-2] @ delta_outputs
        )
        new_weights.append(updated_weights)        
        self.WEIGHTS = new_weights

    def extreme_indirect_feedback_alignment(self, desired_inputs:ndarray,desired_outputs:ndarray) -> None:
        """
        use Indirect Feedback Alighnment to project error back to each layer
        then use Extreme Learning Machine's pseudoinverse method to fit each weights
        (CURRENTLY NOT WORKING)
        """
        predicted_outputs, predicted_hidden_layers = self.forward_pass(desired_inputs)
        predicted_hidden_layers_transposed = list(map(lambda layer:layer.T,predicted_hidden_layers))
        delta_outputs = desired_outputs - predicted_outputs[-1]
        delta_hidden_layer_initial_transposed = self.projected_error(
            layer=predicted_hidden_layers_transposed[0],
            error=delta_outputs.T,
            weights=self.ERROR_WEIGHTS
        )
        desired_hidden_layer_initial_transposed = predicted_hidden_layers_transposed[0] - delta_hidden_layer_initial_transposed
        desired_hidden_layer_initial = desired_hidden_layer_initial_transposed.T
        self.WEIGHTS[0] = self.extreme_learn_weights(
            layer_before=desired_inputs,
            layer_after=desired_hidden_layer_initial
        )

        desired_hidden_layer_prior = desired_hidden_layer_initial
        delta_hidden_layer_prior_transposed = delta_hidden_layer_initial_transposed
        for hidden_layer_depth in range(1,self.N_HIDDEN_LAYERS):
            delta_hidden_layer_transposed = self.projected_error(
                layer=predicted_hidden_layers_transposed[hidden_layer_depth],
                error=delta_hidden_layer_prior_transposed,
                weights=self.WEIGHTS[hidden_layer_depth]
            )
            desired_hidden_layer_transposed = predicted_hidden_layers_transposed[hidden_layer_depth] - delta_hidden_layer_transposed
            desired_hidden_layer = desired_hidden_layer_transposed.T
            self.WEIGHTS[hidden_layer_depth] = self.extreme_learn_weights(
                layer_before=desired_hidden_layer_prior,
                layer_after=desired_hidden_layer
            )
            desired_hidden_layer_prior = desired_hidden_layer
            delta_hidden_layer_prior_transposed = delta_hidden_layer_transposed

        self.WEIGHTS[-1] = self.extreme_learn_weights(
            layer_before=desired_hidden_layer_prior,
            layer_after=desired_outputs
        )            

    def _hidden_layers(self,layer:ndarray) -> Tuple[List[ndarray],List[ndarray]]:
        activated_layers = []
        unactivated_layers = []
        for depth in range(self.N_HIDDEN_LAYERS):
            logits = layer @ self.WEIGHTS[depth]
            layer = self.activation_function(logits)
            unactivated_layers.append(logits)
            activated_layers.append(layer)
        return activated_layers,unactivated_layers
  
    def _output_layer(self, hidden: ndarray) -> ndarray: 
        return hidden @ self.WEIGHTS[-1]

    @staticmethod
    def get_training_error(predicted_outputs:ndarray, expected_outputs:ndarray) -> float:
        return sum(argmax(predicted_outputs, axis=1) != argmax(expected_outputs, axis=1))

    @staticmethod
    def projected_error(layer:ndarray,error:ndarray,weights:ndarray) -> ndarray:
        return layer * (weights @ error)

    @staticmethod
    def extreme_learn_weights(layer_before:ndarray, layer_after:ndarray) -> ndarray:
        return pinv(layer_before) @ layer_after
    
    @staticmethod
    def sigmoid(x: ndarray) -> ndarray:
        return 1. / (1. + exp(-x))

    @staticmethod
    def differential_of_sigmoid(x: ndarray) -> ndarray:
        """
        differential of sigmoid function
        f'(x) = f(x)(1-f(x))
        """
        sig = ELMClassifier.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def weights_representation(weights:List[ndarray]) -> str:
        return "\n\nweights:\n{learnt_weights}".format(
            learnt_weights="\n".join(map(lambda weights:str(weights.shape), weights)),
        )

    @staticmethod
    def random_weights(input_width:int,output_width:int,low:float=-.1,high:float=.1,) -> ndarray:
        return uniform(low=low,high=high,size=[input_width,output_width])

    @staticmethod
    def get_random_weights(input_layer_width:int,hidden_layer_width:int,output_layer_width:int,number_of_hidden_layers:int) -> List[ndarray]:
        return list(
            map(
                lambda _:ELMClassifier.random_weights(hidden_layer_width,output_layer_width),
                range(number_of_hidden_layers+1)
            )
        )

    @staticmethod
    def get_weights(input_layer_width:int,hidden_layer_width:int,output_layer_width:int,number_of_hidden_layers:int) -> List[ndarray]:
        return [ELMClassifier.random_weights(input_layer_width,hidden_layer_width)] + list(map(
            lambda _:ELMClassifier.random_weights(hidden_layer_width,hidden_layer_width),
            range(number_of_hidden_layers-1)
        )) + [ELMClassifier.random_weights(hidden_layer_width,output_layer_width)]
    
    @staticmethod
    def get_delta_weights_DFA(
        error:ndarray,
        hidden_layers:List[ndarray],
        random_weights:List[ndarray],
        unactivated_layers:List[ndarray],
        desired_inputs:ndarray,
        n_layers:int
    ) -> List[ndarray]:

        delta_weights = []
        e = da = error.T
        for depth in range(n_layers,1,-1):
            h = hidden_layers[depth-1]
            B = random_weights[depth]
            a = unactivated_layers[depth-1]
            dW = da @ h
            delta_weights.append(-dW.T)       
            da =  B @ e * ELMClassifier.differential_of_sigmoid(a).T 
        
        dW = da @ h
        delta_weights.append(-dW.T)
        return delta_weights[::-1]