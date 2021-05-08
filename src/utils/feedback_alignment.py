from numpy import ndarray
#########################################

class FeedbackAlignment:
    @staticmethod
    def get_layer_error(layer_predicted:ndarray,layer_expected:ndarray) -> ndarray:
        """
        e = y - y^
        """
        print(f"""
        layer_expected - layer_predicted
        {layer_expected.shape} - {layer_predicted.shape}
        """)
        return layer_expected - layer_predicted
    
    @staticmethod
    def get_desired_layer(predicted_layer:ndarray,delta_layer:ndarray) -> ndarray:
        print(f"""
        predicted_layer - delta_layer
        {predicted_layer.shape} - {delta_layer.shape}
        """)
        return predicted_layer - delta_layer

    @staticmethod
    def get_projected_error(layer:ndarray,error:ndarray,weights:ndarray) -> ndarray:
        """
        direct feedback alignment (DFA) method
        of propogating errors back to earlier layers
        via directly projecting the error to each layer
        (a simpler alternative to backpropagating the error through each layer)
        """
        print(f"""
        layer * (weights @ error)
        {layer.shape} * ({weights.shape} @ {error.shape})
        """)
        x= layer * (weights @ error)
        print(f"={x.shape}")
        return x