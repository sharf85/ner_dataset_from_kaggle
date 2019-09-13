import numpy as np
from keras.preprocessing.text import Tokenizer
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


class Evaluator():

    def __init__(self, tokenizer: Tokenizer):
        index_to_words = {id: word for word, id in tokenizer.word_index.items()}
        # index_to_words[0] = '<PAD>'
        index_to_words[0] = 'O'
        self._mapper = index_to_words

    def ids_to_tags(self, label):
        return [self._mapper[id[0]] for id in label]

    def prediction_to_tags(self, prediction):
        return [self._mapper[prediction] for prediction in np.argmax(prediction, 1)]

    def evaluate_metrics(self, y_test, predictions):
        test_labels = [self.ids_to_tags(ids) for ids in y_test]
        pred_labels = [self.prediction_to_tags(prediction) for prediction in predictions]

        return classification_report(test_labels, pred_labels)
