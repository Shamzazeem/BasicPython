import csv
import random
import operator
import math

def load_dataset(filename, split):
    training_set = []
    test_set = []

    csv_file = open(filename, 'r')
    lines = csv.reader(csv_file)
    dataset = list(lines)
    for i in range(len(dataset) - 1):
        for j in range(4):
            dataset[i][j] = float(dataset[i][j])
            if random.random() < split:
                training_set.append(dataset[i])
            else:
                test_set.append(dataset[i])
    return training_set, test_set

def euclidean_distance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)

    return math.sqrt(distance)

def get_neighbors(training_set, test_data, k):

    distance = []
    length = len(test_data) - 1

    for i in range(len(training_set)):
        dist = euclidean_distance(test_data, training_set[i], k)

        distance.append((training_set[i], dist))

    distance.sort(key=operator.itemgetter(1))

    neighbors = []

    for i in range(k):
        neighbors.append(distance[i][0])

    return neighbors

def get_response(neighbors):
    class_votes = {}

    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1

    sorted_votes = sorted(class_votes.items(), key= operator.itemgetter(1), reverse=True)

    return sorted_votes[0][0]


def get_accuracy(test_set, predications):
    correct = 0

    for i in range(len(test_set)):
        if test_set[i][-1] == predications[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0

def predicate(test_set, training_set, k):
    predications = []

    for i in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[i], k)

        result = get_response(neighbors)
        predications.append(result)

        print('predicated = '+ result + 'actual =' + test_set[i][-1])

    return predications

def main():
    split = 0.5
    training_set, test_set = load_dataset('iris.data', split)
    print('Training set:' + str(len(training_set)))

    print("test set:" + str(len(test_set)))

    k = 3
    predications = predicate(training_set, training_set, k)

    accuracy_train = get_accuracy(training_set, predications)

    print('Accuracy train:' + str(accuracy_train) + '%')

    print('================')


    predications = predicate(test_set, training_set, k)

    accuracy_test = get_accuracy(test_set, predications)
    print('Accuracy test: ' + str(accuracy_test) + '%')

