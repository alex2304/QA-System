import os
from typing import List

from sklearn.feature_extraction.text import CountVectorizer


# TODO: lemmatize words before passing here (to distinguish different forms of a word)
class BugOfWords:
    def __init__(self):
        self.count_vectorizer = CountVectorizer(
            encoding='utf-8',
            ngram_range=(1, 2),
            max_df=1.0,
            min_df=1,
            max_features=500
        )

    def get_features(self, samples: List[str]):
        return self.count_vectorizer.fit_transform(samples).toarray()


# for in-place test
samples_folder = os.path.join(__file__, 'samples')
samples_files = ['class1', 'class2']


# TODO: distinguish classes of samples
def load_samples():
    samples = []

    for f_name in samples_files:
        with open(os.path.join(samples_folder, f_name), encoding='utf-8') as f:
            samples.extend(line.replace('\n', '') for line in f)

    return samples


if __name__ == '__main__':
    vectorizer = BugOfWords()

    samples = load_samples()

    print(vectorizer.get_features(samples))
