from typing import List

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from numpy import argmax, ndarray

from src.d_elm import ELMClassifier
#++++++++++++++++++++++++++++++++++++++++++++++++++++++
def onehot(class_index:int, vector_size:int) -> List[float]:
    vector = [0.]*(vector_size+1)
    vector[class_index] = 1.
    return vector
    
def encoder(class_indexes:List[int],n_classes:int) -> List[List[float]]:
    return list(
        map(
            lambda class_index: onehot(class_index,n_classes),
            class_indexes
        )
    )

def decoder(vectors:List[List[float]]) -> List[int]:
    return list(map(argmax,vectors))
#++++++++++++++++++++++++++++++++++++++++++++++++++++++
dataset = load_digits()

X_train, X_test, y_train_classes, y_test_classes = train_test_split(
    dataset.images.reshape((len(dataset.images), -1)), 
    dataset.target, 
    test_size=0.5, 
    shuffle=False
)
y_train = encoder(class_indexes=y_train_classes,n_classes=max(y_test_classes))
x_example = X_train[0]
y_example = y_train[0]
input_vector_length = X_train.shape[-1]
output_vector_length = len(y_train[0])
HIDDEN_LAYERS = 500
DEPTH = 1
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
classifier_elm = ELMClassifier(
    input_layer_width=input_vector_length,
    output_layer_width=output_vector_length,
    hidden_layer_width=HIDDEN_LAYERS,
    number_of_hidden_layers=DEPTH
)
print(classifier_elm)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++
y_before = classifier_elm.infer(x_example)
y_predicted = decoder(classifier_elm.infer(X_test))
scores = f1_score(y_test_classes, y_predicted,average=None)
print(confusion_matrix(y_test_classes, y_predicted))
print(f"BEFORE:\n\nexpected = {y_example}\nafter training = {y_before}\nf1s = {scores}\n\n")
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
print(classifier_elm)
classifier_elm.fit_final_weights(inputs=X_train,outputs=y_train)
y_after = classifier_elm.infer(x_example)
y_predicted = decoder(classifier_elm.infer(X_test))
scores_a = f1_score(y_test_classes, y_predicted,average=None)
print(confusion_matrix(y_test_classes, y_predicted))
print(f"AFTER:\n\nexpected = {y_example}\nafter training = {y_after}\nf1s = {scores_a}\n\n")
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
# classifier_ida = ELMClassifier(
#     input_layer_width=input_vector_length,
#     output_layer_width=output_vector_length,
#     hidden_layer_width=HIDDEN_LAYERS,
#     number_of_hidden_layers=DEPTH
# )
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
# print(classifier_ida)
# classifier_ida.indirect_feedback_alignment(inputs=X_train,outputs=y_train,epochs=5)
# y_after = classifier_ida.infer(x_example)
# y_predicted = decoder(classifier_ida.infer(X_test))
# scores_b = f1_score(y_test_classes, y_predicted,average=None)
# print(confusion_matrix(y_test_classes, y_predicted))
# print(f"AFTER:\n\nexpected = {y_example}\nafter training = {y_after}\nf1s = {scores_b}\n\n")
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
# classifier_ida.fit_final_weights(inputs=X_train,outputs=y_train)
# y_after = classifier_ida.infer(x_example)
# y_predicted = decoder(classifier_ida.infer(X_test))
# scores_e = f1_score(y_test_classes, y_predicted,average=None)
# print(confusion_matrix(y_test_classes, y_predicted))
# print(f"AFTER:\n\nexpected = {y_example}\nafter training = {y_after}\nf1s = {scores_e}\n\n")
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
# classifier_dfa = ELMClassifier(
#     input_layer_width=input_vector_length,
#     output_layer_width=output_vector_length,
#     hidden_layer_width=HIDDEN_LAYERS,
#     number_of_hidden_layers=DEPTH
# )
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
# print(classifier_dfa)
# classifier_dfa.direct_feedback_alignment(inputs=X_train,outputs=y_train,epochs=100,learning_rate=1e-6)
# y_after = classifier_dfa.infer(x_example)
# y_predicted = decoder(classifier_dfa.infer(X_test))
# scores_c = f1_score(y_test_classes, y_predicted,average=None)
# print(confusion_matrix(y_test_classes, y_predicted))
# print(f"AFTER:\n\nexpected = {y_example}\nafter training = {y_after}\nf1s = {scores_c}\n\n")
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
# classifier_dfa.fit_final_weights(inputs=X_train,outputs=y_train)
# y_after = classifier_dfa.infer(x_example)
# y_predicted = decoder(classifier_dfa.infer(X_test))
# scores_d = f1_score(y_test_classes, y_predicted,average=None)
# print(confusion_matrix(y_test_classes, y_predicted))
# print(f"AFTER:\n\nexpected = {y_example}\nafter training = {y_after}\nf1s = {scores_d}\n\n")
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
# classifier_novel = ELMClassifier(
#     input_layer_width=input_vector_length,
#     output_layer_width=output_vector_length,
#     hidden_layer_width=HIDDEN_LAYERS,
#     number_of_hidden_layers=DEPTH
# )
# print(classifier_novel)
# classifier_novel.one_shot_extreme_deep_learning(desired_inputs=X_train,desired_outputs=y_train)
# y_after = classifier_novel.infer(x_example)
# y_predicted = decoder(classifier_novel.infer(X_test))
# scores_f = f1_score(y_test_classes, y_predicted,average=None)
# print(confusion_matrix(y_test_classes, y_predicted))
# print(f"AFTER:\n\nexpected = {y_example}\nafter training = {y_after}\nf1s = {scores_f}\n\n")
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
# print(f"""
# comparison:
# No training: {scores}
# DFA only: {scores_c}
# IFA only: {scores_b}
# ELM only: {scores_a}
# ELM with DFA: {scores_d}
# ELM with IFA {scores_e}
# Novel {scores_f}
# """)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

classifier_novel_novel = ELMClassifier(
    input_layer_width=input_vector_length,
    output_layer_width=output_vector_length,
    hidden_layer_width=HIDDEN_LAYERS,
    number_of_hidden_layers=DEPTH
)
print(classifier_novel_novel)
classifier_novel_novel.one_shot_extreme_deep_learning_experimental(desired_inputs=X_train,desired_outputs=y_train)
y_after = classifier_novel_novel.infer(x_example)
y_predicted = decoder(classifier_novel_novel.infer(X_test))
scores_g = f1_score(y_test_classes, y_predicted,average=None)
print(confusion_matrix(y_test_classes, y_predicted))
print(f"AFTER:\n\nexpected = {y_example}\nafter training = {y_after}\nf1s = {scores_g}\n\n")