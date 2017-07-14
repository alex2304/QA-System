import os
from typing import List

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from pymystem3 import Mystem

from sklearn.feature_selection import SelectKBest, chi2


# TODO: lemmatize words before passing here (to distinguish different forms of a word)
class FeaturesExtractor:
    
    def __init__(self, n_gram=1, max_features=1000, min_occurrence_rate=1, max_occurrence_rate=1.0):
        self.count_vectorizer = CountVectorizer(
            encoding='utf-8',
            ngram_range=(1, n_gram),
            max_df=max_occurrence_rate,
            min_df=min_occurrence_rate,
            max_features=max_features
        )
        self.mystem = Mystem()

    def stem(self, raw_documents: List[str]):
        return [self.mystem.lemmatize(doc) for doc in raw_documents]
    
    def load_vocabulary(self, raw_documents: List[str]):
        self.count_vectorizer.fit(self.stem(raw_documents))

    def get_features(self, samples: List[str]):
        return self.count_vectorizer.transform(self.stem(samples)).toarray()


# for in-place test
samples_folder = os.path.join(os.path.dirname(__file__), 'samples')
samples_files = ['class1.txt', 'class2.txt']


def load_samples():
    samples = []
    classes = [1 for i in range(19)]
    classes.extend([2 for i in range(10)])

    for f_name in samples_files:
        with open(os.path.join(samples_folder, f_name), encoding='utf-8') as f:
            samples.extend(line.replace('\n', '') for line in f)

    return samples, classes


if __name__ == '__main__':
    f_generator = FeaturesExtractor()

    samples, target = load_samples()

    f_generator.load_vocabulary(samples)

    features = f_generator.get_features(samples)

    # select features
    selector = SelectKBest(chi2, k=10)

    selector.fit(features, target)
    features = selector.transform(features)

    print('New shape: %s' % str(features.shape))

    # study model
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(features, target)

    # predict
    for_prediction = f_generator.get_features(['На какую специальность податься',
                                               'На какой факультет поступить',
                                               'Как поступить в Иннополис',
                                               'Бакалаврский и магистерский'])
    for_prediction = selector.transform(for_prediction)

    predicted_class = clf.predict(for_prediction)
    print(predicted_class)
