from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import numpy as np


def load_data():
    f1 = open("clean_real.txt", "r")
    real_lines = f1.read().splitlines()

    f2 = open("clean_fake.txt", "r")
    fake_lines = f2.read().splitlines()

    f1.close()
    f2.close()

    real_labels = []
    fake_labels = []

    for _ in range(len(real_lines)):
        real_labels.append("real")

    for _ in range(len(fake_lines)):
        fake_labels.append("fake")

    data = real_lines.copy()
    data.extend(fake_lines)
    labels = real_labels.copy()
    labels.extend(fake_labels)

    vectorizer = CountVectorizer()
    data_vectorized = vectorizer.fit_transform(data)

    train_data, test_validation_data, train_labels, test_validation_labels = train_test_split(data_vectorized, labels, train_size=0.7, random_state=42)
    validation_data, test_data, validation_labels, test_labels = train_test_split(test_validation_data, test_validation_labels, train_size=0.5, random_state=42)

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels, vectorizer.get_feature_names_out()


def select_model(trainData, trainLabels, validationData, validationLabels, maxDepthList):
    for d in maxDepthList:
        for criterion in ["gini", "entropy"]:
            decTree = DecisionTreeClassifier(criterion=criterion, max_depth=d, random_state=42)
            decTree.fit(trainData, trainLabels)
            validationAccuracy = decTree.score(validationData, validationLabels) * 100
            print(f'{format(validationAccuracy, ".2f")}% validation accuracy with {criterion} '
                  f'split criterion and maximum depth of {d}')


def compute_information_gain(trainData, trainLabels, keyword, featureNames):
    realProportion = trainLabels.count("real") / len(trainLabels)
    fakeProportion = trainLabels.count("fake") / len(trainLabels)
    HY = - (realProportion * math.log2(realProportion) + fakeProportion * math.log2(fakeProportion))

    trainArray = trainData.toarray()
    keywordIndex = np.where(featureNames == keyword)[0][0]
    numberFake = 0
    numberFakeWithoutKeyword = 0
    numberReal = 0
    numberRealWithoutKeyword = 0
    numberKeyword = 0
    numberWithoutKeyword = 0

    for i in range(len(trainLabels)):
        if trainArray[i][keywordIndex] > 0:
            numberKeyword += 1
            if trainLabels[i] == "real":
                numberReal += 1
            else:
                numberFake += 1
        else:
            numberWithoutKeyword += 1
            if trainLabels[i] == "real":
                numberRealWithoutKeyword += 1
            else:
                numberFakeWithoutKeyword += 1

    HYXWith = - ((numberReal / numberKeyword * math.log2(numberReal / numberKeyword)) + (numberFake / numberKeyword * math.log2(numberFake / numberKeyword)))

    HYXWithout = - ((numberRealWithoutKeyword / numberWithoutKeyword * math.log2(numberRealWithoutKeyword / numberWithoutKeyword)) + (numberFakeWithoutKeyword / numberWithoutKeyword * math.log2(numberFakeWithoutKeyword / numberWithoutKeyword)))

    HYX = (numberKeyword / len(trainLabels)) * (HYXWith) + (numberWithoutKeyword / len(trainLabels)) * (HYXWithout)

    return HY - HYX


if __name__ == '__main__':
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels, feature_names = load_data()
    select_model(train_data, train_labels, validation_data, validation_labels, [50, 100, 250, 500, 1000])

    # Visualization code
    decTree = DecisionTreeClassifier(criterion="entropy", max_depth=100, random_state=42)
    decTree.fit(train_data, train_labels)
    fig = plt.figure(figsize=(12, 12))
    plot_tree(decTree, max_depth=2, feature_names=feature_names, class_names=decTree.classes_, fontsize=10)
    plt.savefig("decTree", dpi=100, bbox_inches="tight")

    # Information gain code
    print(f'The information gain for splitting on the keyword "the" is {format(compute_information_gain(train_data, train_labels, "the", feature_names), ".5f")}')
    print(f'The information gain for splitting on the keyword "hillary" is {format(compute_information_gain(train_data, train_labels, "hillary", feature_names), ".5f")}')
    print(f'The information gain for splitting on the keyword "trumps" is {format(compute_information_gain(train_data, train_labels, "trumps", feature_names), ".5f")}')
