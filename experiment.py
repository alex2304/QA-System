import os
from typing import Tuple, List, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2

from text_processing.features_gen import FeaturesExtractor
from text_processing.mystem import TextProcessor


def load_samples(samples_folder: str) -> Tuple[List[str], List[Any]]:
    samples, classes = [], []

    for f_name in os.listdir(samples_folder):
        with open(os.path.join(samples_folder, f_name), encoding='utf-8') as f:
            file_samples = [line.replace('\n', '') for line in f]

            samples.extend(file_samples)

            classes.extend([f_name] * len(file_samples))

    if len(samples) != len(classes):
        raise ValueError('Number of train_samples does not correspond number of classes')

    return samples, classes


train_samples_folder = os.path.join(os.path.dirname(__file__), 'samples/train')
test_samples_folder = os.path.join(os.path.dirname(__file__), 'samples/test')

if __name__ == '__main__':
    with TextProcessor() as tp:
        f_extractor = FeaturesExtractor()

        # load train_samples and their targets
        train_samples, train_target = load_samples(train_samples_folder)

        # stemming train_samples
        train_samples = tp.stemming(train_samples)

        # print(train_samples)

        # load vocabulary to bag of words
        f_extractor.load_vocabulary(train_samples)

        # print(f_extractor.get_features_names())

        # extract train_features
        train_features = f_extractor.extract_features(train_samples)

        # select train_features
        selector = SelectKBest(chi2, k=10)

        selector.fit(train_features, train_target)
        train_features = selector.transform(train_features)

        # print('New shape: %s\n' % str(train_features.shape))

        # study model
        clf = RandomForestClassifier(n_jobs=-1)
        clf.fit(train_features, train_target)

        # get samples to predict
        test_samples, test_target = load_samples(test_samples_folder)

        test_samples = tp.stemming(test_samples)

        print(test_samples)

        # get unknown train_features
        test_features = f_extractor.extract_features(test_samples)
        test_features = selector.transform(test_features)

        # predict
        predicted_classes = clf.predict(test_features)

        print('\nPrediction results\n'
              '? | Number | Sample | Predicted | Real')

        for i, s in enumerate(test_samples):
            print('{state} | {i} | {s} | {p} | {r}'.format(
                state=int(predicted_classes[i] == test_target[i]),
                i=i,
                s=s,
                p=predicted_classes[i],
                r=test_target[i]
            ))
