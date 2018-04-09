#https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes
import os
os.chdir('/home/victor/ml_projects/week4-ex')

import csv
import random
import math
from math import sqrt, pow, pi, exp
def load_csv(filename):
    lines = csv.reader(open(filename))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]

    return dataset

def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]

def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            #vector[-1] = [0, 1]
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries

def calculate_probability(x, mean, std_dev):
    exponent = exp(-(pow(x - mean, 2) / (2 * pow(std_dev, 2))))
    return (1 / (sqrt(2 * pi) * std_dev)) * exponent

def calculate_class_probabilites(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stddev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stddev)

    return probabilities

def predicate(summaries, input_vector):
    probabilites = calculate_class_probabilites(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilites.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def get_predications(summaries, test_sets):
    predications = []
    for i in range(len(test_sets)):
        result = predicate(summaries, test_sets[i])
        predications.append(result)

    return predications

def get_accuracy(test_set, predications):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predications[i]:
            correct += 1

    return (correct / float(len(test_set))) * 100.0

def main():
    filename = 'pima-indians-diabetes.data.csv'
    split_ratio = 0.67
    dataset = load_csv(filename)
    training_set, test_set = split_dataset(dataset, split_ratio)
    summaries  = summarize_by_class(training_set)
    predications = get_predications(summaries, test_set)

    accuracy = get_accuracy(test_set, predications)

    print('Accuray {0}'.format(accuracy))

main()
