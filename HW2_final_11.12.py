from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools


class Label(Enum):
    T_SHIRT = 0
    TROUSER = 1
    PULLOVER = 2
    DRESS = 3
    COAT = 4
    SANDAL = 5
    SHIRT = 6
    SNEAKER = 7
    BAG = 8
    ANKLE_BOOT = 9


def getName(self):
    return self.name


def data_normal(data, mean=0, var=0):
    if mean == 0:
        mean = np.mean(data)
        var = np.var(data)
        mean = np.mean(mean)
        var = np.mean(var)
    normalised_data = (data - mean) / ((var + 1e-8) ** 0.5)
    return normalised_data, mean, var


def sigmoid(self, x):
    x = np.clip(x, a_min=self.sigmoid_min, a_max=self.sigmoid_max)
    a = 1 / (1 + np.exp(-x))
    return a


def sigmoid_derivative(self, x):
    a = sigmoid(self, x) * (1 - sigmoid(self, x))
    return a


def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)


def softmax_derivative(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))


def adapt_labels_to_matrix(labels):
    """converts label's values to a array with "1" in the location of its value
    returns matrix [10, BATCH_SIZE]"""
    label_matrix = np.zeros((10, len(labels)))
    for i in range(len(labels)):
        label_matrix[labels[i]][i] = 1
    return label_matrix


def cross_entropy(Y_hat, label_matrix):
    a = np.argmax(label_matrix, axis=0)
    b = []
    for i in range(len(a)):
        b.append(Y_hat[a[i]][i])
    log_likelihood = -np.log(b)
    cost = np.sum(log_likelihood)
    return cost


def cross_entropy_softmax_derivative(Y_hat, label_matrix):  # TODO: check works correctly
    a = np.argmax(label_matrix, axis=0)
    for i in range(len(a)):
        Y_hat[a[i]][i] -= 1
    return Y_hat


def label_prediction(prob_matrix):
    result = np.argmax(prob_matrix, axis=0)
    return result


def compute_accuracy(predictions, labels):
    success = []
    for i in range(len(predictions)):
        success.append(predictions[i] == labels[i])
    accuracy = np.mean(success) * 100
    return accuracy


def data_split(data_file):
    """used to split data into train, validation and test data"""
    np.random.shuffle(data_file.values)
    total_length    = len(data_file)  # 56,000
    train_data      = int(0.7*total_length)
    validation_data = int(0.2*total_length)
    test_data       = int(0.1*total_length)

    train = data_file[:(train_data)]
    validation = data_file[(train_data+1):(train_data+1+validation_data)]
    test = data_file[(train_data+1+validation_data+1):]

    return train, validation, test


def data_grid(train_file):
    """prints out the 10x4 grid of pictures"""
    count = 0
    running_index = 0
    filling_array = np.zeros(10)

    num_row = 10
    num_col = 4

    fig, axes = plt.subplots(num_row, num_col, figsize=(1 * num_col, 1 * num_row))
    while count < 40:
        row_number = train_file.iloc[running_index].values[0]
        col_number = int(filling_array[row_number])
        if col_number<4:
            ax = axes[row_number, col_number]
            image = np.delete(train_file.iloc[running_index].values, [0])
            image.resize(28, 28)
            ax.imshow(image, cmap='gray')

            filling_array[row_number] += 1
            count += 1
        running_index += 1

    cols = ['Image {}'.format(col) for col in range(1, 5)]
    rows = ['{}'.format(getName(Label(i))) for i in range(10)]

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, size='small', color='green')

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0, size='small', color='blue')

    plt.tight_layout()
    plt.show()


class NeuralNetworkQ2:
    def __init__(self):
        self.batch_size = 200
        self.num_of_classes = 10
        self.lamb = 0.2
        self.epochs = 10
        self.learning_rate = 0.005
        self.seed = 0
        self.sigmoid_max = 100
        self.sigmoid_min = -self.sigmoid_max
        self.epsilon = np.finfo(float).eps
        self.mean = 0
        self.var = 0

    def initializeParameters(self):
        """initialize weights and bias"""
        """
        Model sizes:
        W = [784,10], x = [batch_size, 784], b = [10, 1]
        W.T * x + b = [10, batch_size]
        """
        if self.seed >= 0:
            np.random.seed(self.seed)
        W = np.random.randn(784, self.num_of_classes)  # [784, 10]
        b = np.random.randn(self.num_of_classes, 1)  # [10, 1]
        parameters = {"b": b, "W": W}
        return parameters


    def forwardPropagation(self, data, labels, parameters):
        W = parameters["W"]
        b = parameters["b"]

        h = np.dot(W.T, data.T)  # [10,784] * [784, batch_size]
        h += b  # [10, batch_size] + [10, 1]
        f = sigmoid(self, h)
        cost = 0
        label_matrix = 0
        if labels is not False:
            label_matrix = adapt_labels_to_matrix(labels)
            cost = np.sum((label_matrix - f) ** 2) + 0.5 * self.lamb * np.sum(W ** 2)
        parameters = {"b": b, "W": W, "h": h, "Y_hat": f, "label_matrix": label_matrix}
        return cost, parameters

    def backwardPropagation(self, data, parameters):
        b                = parameters["b"]
        W                = parameters["W"]
        h                = parameters["h"]
        f                = parameters["Y_hat"]
        label_matrix     = parameters["label_matrix"]

        dldf = 2 * (f-label_matrix)
        dfdh = sigmoid_derivative(self, h)
        dhdw = data
        dhdb = np.ones((self.num_of_classes, self.batch_size))

        dldw = np.dot(np.multiply(dldf, dfdh), dhdw)  # [10xBATCH_SIZE] element-wise [10xBATCH_SIZE] * [BATCH_SIZEx784]
        dldw = dldw.T
        dldh = np.multiply(dldf, dfdh)  # [10xBATCH_SIZE] element-wise [10xBATCH_SIZE] = [10xBATCH_SIZE]
        dldb = np.sum(dldh, axis=1, keepdims=True) / self.batch_size  # [10x1]

        gradients = {"dW": dldw, "db": dldb}
        return gradients

    def updateParameters(self, parameters, gradients):
        parameters["W"] = parameters["W"] - (self.learning_rate * gradients["dW"]) - (self.learning_rate * self.lamb * parameters["W"])
        parameters["b"] -= self.learning_rate * gradients["db"]
        return parameters

    def train(self, epoch_data, epoch_labels, parameters):
        print("*** START of training *** ")

        run_index = 0
        run_per_epoch = int(len(epoch_data) / self.batch_size)

        losses = np.zeros((int(self.epochs * run_per_epoch), 1))  #
        accuracy = np.zeros((int(self.epochs * run_per_epoch), 1))

        for i in range(self.epochs):
            for j in range(0, int(len(epoch_data)), self.batch_size):
                data = epoch_data.iloc[j:j + self.batch_size]
                labels = epoch_labels[j:j + self.batch_size]

                losses[run_index, 0], parameters = self.forwardPropagation(data, labels, parameters)
                accuracy[run_index, 0] = compute_accuracy(label_prediction(parameters["Y_hat"]), labels)
                run_index += 1
                gradients = self.backwardPropagation(data, parameters)
                parameters = self.updateParameters(parameters, gradients)

        return losses, accuracy, parameters


class NeuralNetworkQ3:
    def __init__(self):
        self.num_of_classes = 10
        self.seed = 0
        self.sigmoid_max = 100
        self.sigmoid_min = -self.sigmoid_max
        self.epsilon = np.finfo(float).eps
        self.mean = 0
        self.var = 0

        self.batch_size = 100  # must divide 39200
        self.lamb = 0.2
        self.epochs = 10
        self.learning_rate = 0.005
        self.hidden_neurons = 128

    def initializeParameters(self):
        """initialize weights and bias"""
        """
        Model Sizes:
        W1 = [784,d], W2 = [d,10], b1 = [d, 1],  b2 = [10, 1]
        x = [784, batch_size] ; data = x.transpose
        z1 = W1.T * x + b1 = [d, batch_size]
        h = sigmoid(z) = [d, batch_size]
        z2 = W2.T * h + b2 = [10, batch_size]
        Y_hat = softmax(z2) = [10, batch_size]
        """
        if self.seed >= 0:
            np.random.seed(self.seed)
        W1 = np.random.randn(784, self.hidden_neurons)  # [784, d]
        b1 = np.random.randn(self.hidden_neurons, 1)  # [d, 1]
        W2 = np.random.randn(self.hidden_neurons, self.num_of_classes)  # [d, 10]
        b2 = np.random.randn(self.num_of_classes, 1)  # [10, 1]
        parameters = {"b1": b1, "W1": W1, "b2": b2, "W2": W2}
        return parameters

    def forwardPropagation(self, data, labels, parameters):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        z1 = np.dot(W1.T, data.T)  # [d,784] * [784, batch_size]
        z1 += b1  # [10, batch_size] + [10, 1]
        h = sigmoid(self, z1)

        z2 = np.dot(W2.T, h)
        z2 += b2
        Y_hat = softmax(z2)
        cost, label_matrix = 0, 0
        if labels is not False:
            label_matrix = adapt_labels_to_matrix(labels)
            cost = cross_entropy(Y_hat, label_matrix) + 0.5 * self.lamb * np.sum(W1 ** 2) + 0.5 * self.lamb * np.sum(W2 ** 2)

        parameters = {"b1": b1, "W1": W1, "b2": b2, "W2": W2, "z1": z1, "z2": z2, "h": h, "Y_hat": Y_hat, "label_matrix": label_matrix}
        return cost, parameters

    def backwardPropagation(self, data, parameters):
        b1                   = parameters["b1"]
        W1                   = parameters["W1"]
        b2                   = parameters["b2"]
        W2                   = parameters["W2"]
        h                    = parameters["h"]
        Y_hat                = parameters["Y_hat"]
        label_matrix         = parameters["label_matrix"]

        dldz2 = cross_entropy_softmax_derivative(Y_hat, label_matrix)  # [10, batchsize]
        dz2db2 = 1
        dz2dw2 = h.T

        dz2dh = W2.T  # [10, d]
        dhdz1 = sigmoid_derivative(self, h)  # [d, batchsize]
        dz1db1 = 1
        dz1dw1 = data  # [batchsize, 784]

        # dldw1 = dldz2 dz2dh dhdz1 dz1dw1
        a = np.dot(dz2dh.T, dldz2)  # [d, batchsize]
        b = np.multiply(a, dhdz1)  # [d, batchsize]
        dldw1 = (np.dot(b, dz1dw1)).T  # [784, d]

        # dldb1 = dldz2 dz2dh dhdz1 dz1db1
        a = np.dot(dz2dh.T, dldz2)  # [d, batchsize]
        b = np.multiply(a, dhdz1)  # [d, batchsize]
        dldb1 = np.sum(b, axis=1, keepdims=True)

        # dldw2 = dldz2 dz2dw2
        dldw2 = (np.dot(dldz2, dz2dw2)).T  # [d, 10]

        # dldb2 = dldz2 dz2db2
        b = dldz2  # [10, batchsize]
        dldb2 = np.sum(b, axis=1, keepdims=True)  # [10, 1]

        gradients = {"dW1": dldw1, "db1": dldb1, "dW2": dldw2, "db2": dldb2}
        return gradients

    def updateParameters(self, parameters, gradients):
        parameters["W1"] = parameters["W1"] - (self.learning_rate * gradients["dW1"]) - (self.learning_rate * self.lamb * parameters["W1"])
        parameters["b1"] -= self.learning_rate * gradients["db1"]
        parameters["W2"] = parameters["W2"] - (self.learning_rate * gradients["dW2"]) - (self.learning_rate * self.lamb * parameters["W2"])
        parameters["b2"] -= self.learning_rate * gradients["db2"]
        return parameters

    def train(self, epoch_data, epoch_labels, parameters):
        print("*** START of training *** ")

        run_index = 0
        run_per_epoch = int(len(epoch_data) / self.batch_size)

        losses = np.zeros((int(self.epochs * run_per_epoch), 1))  #
        accuracy = np.zeros((int(self.epochs * run_per_epoch), 1))

        for i in range(self.epochs):
            for j in range(0, int(len(epoch_data)), self.batch_size):
                data = epoch_data.iloc[j:j + self.batch_size]
                labels = epoch_labels[j:j + self.batch_size]

                losses[run_index, 0], parameters = self.forwardPropagation(data, labels, parameters)
                accuracy[run_index, 0] = compute_accuracy(label_prediction(parameters["Y_hat"]), labels)
                run_index += 1
                gradients = self.backwardPropagation(data, parameters)
                parameters = self.updateParameters(parameters, gradients)
        print("*** END of training *** ")

        return losses, accuracy, parameters


def prepare_data_for_model(self, train_csv):
    train_data, validation_data, test_data = data_split(train_csv)

    train_data_only = train_data.drop('label', 1)
    train_data_normal, mean, var = data_normal(train_data_only)
    train_data_labels = train_data.transpose().values[0]

    validation_data_only = validation_data.drop('label', 1)
    validation_data_normal, _, __ = data_normal(validation_data_only, mean, var)
    validation_data_labels = validation_data.transpose().values[0]

    test_data_only = test_data.drop('label', 1)
    test_data_normal, _, __ = data_normal(test_data_only, mean, var)
    test_data_labels = test_data.transpose().values[0]

    self.mean = mean
    self.var = var
    return train_data_normal, train_data_labels, validation_data_normal, validation_data_labels, test_data_normal, test_data_labels


def plot_lossos_and_accuracy(nnq, losses, accuracy):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Runs")
    plt.ylabel("Loss value")
    plt.show()

    plt.figure()
    plt.plot(np.log(losses + nnq.epsilon))
    plt.xlabel("Runs")
    plt.ylabel("Loss value - LOG scale")
    plt.show()

    plt.figure()
    plt.plot(accuracy)
    plt.xlabel("Runs")
    plt.ylabel("Accuracy [%]")
    plt.show()


def personal_test_run(nnq, test_data, test_labels, parameters):
    print(" *** start test our model ***")

    __, parameters = nnq.forwardPropagation(test_data, test_labels, parameters)
    prediction = label_prediction(parameters["Y_hat"])
    print("Test prediction: ", prediction)
    print("Test label: ", test_labels)
    test_accuracy = compute_accuracy(prediction, test_labels)
    print("Test Accuracy: ", test_accuracy)


def question_1(train_csv):
    data_grid(train_csv)


def question_2_preperation(train_csv):
    nnq2 = NeuralNetworkQ2()
    train_data_normal, train_data_labels, validation_data_normal, validation_data_labels, test_data_normal, test_data_labels = prepare_data_for_model(
        nnq2, train_csv)
    parameters = nnq2.initializeParameters()

    # Iteration lists
    # after some test we got:
    # [100,0.2,30,0.001] got 84.428
    # [50,0.1,10,0.005] got 84.04464
    # [100,0.2,50,0.001] got 84.91
    # [100,0.2,1000,0.001] got 85.1375

    # all combinations we thought to test
    batch_size_list = [10, 50, 100, 200, 400]
    lamb_list = [0.2, 0.1, 0.01]
    epochs_list = [50, 100, 200, 500]
    learning_rate_list = [0.001, 0.005, 0.01, 0.05]

    validation_accuracy_list = []
    validation_batch_size_list = []
    validation_lamb_list = []
    validation_epochs_list = []
    validation_learning_rate_list = []
    combination_list = itertools.product(batch_size_list, lamb_list, epochs_list, learning_rate_list)

    for combination in combination_list:
        validation_batch_size_list.append(combination[0])
        validation_lamb_list.append(combination[1])
        validation_epochs_list.append(combination[2])
        validation_learning_rate_list.append(combination[3])

        parameters = nnq2.initializeParameters()
        nnq2.batch_size = combination[0]
        nnq2.lamb = combination[1]
        nnq2.epochs = combination[2]
        nnq2.learning_rate = combination[3]

        # Train
        losses, accuracy, parameters = nnq2.train(train_data_normal, train_data_labels, parameters)

        # Validation
        __, parameters = nnq2.forwardPropagation(validation_data_normal, validation_data_labels, parameters)
        validation_prediction = label_prediction(parameters["Y_hat"])
        validation_test_accuracy = compute_accuracy(validation_prediction, validation_data_labels)
        validation_accuracy_list.append(validation_test_accuracy)
        print("For combination: {}, accuracy is: {}".format(combination, validation_test_accuracy))

    """write validation data -hyper parameters- to csv file"""
    df = pd.DataFrame({'Validation Accuracy': validation_accuracy_list, 'Batch_size': validation_batch_size_list,
                       'Lambda': validation_lamb_list, 'Epochs': validation_epochs_list,
                       'Learning Rate': validation_learning_rate_list})
    df.to_csv('our_test_file_q2.csv', index=False)

    # plt.figure()
    # plt.plot(validation_accuracy_list)
    # plt.xlabel("Runs")
    # plt.ylabel("Loss value")
    # plt.show()

    i = validation_accuracy_list.index(max(validation_accuracy_list))
    best_combination = [validation_batch_size_list[i], validation_lamb_list[i], validation_epochs_list[i],
                        validation_learning_rate_list[i]]

    return best_combination


def question_2(train_csv, test_csv, best_combination=[]):
    nnq2 = NeuralNetworkQ2()
    train_data_normal, train_data_labels, _, __, test_data_normal, test_data_labels = prepare_data_for_model(nnq2, train_csv)

    # best_combination = [400, 0.02, 100, 0.005] # old "best combinataion"
    best_combination = [100,0.2,1000,0.001]  # for this combination, got 85.1375

    combination = best_combination

    parameters = nnq2.initializeParameters()
    nnq2.batch_size = combination[0]
    nnq2.lamb = combination[1]
    nnq2.epochs = combination[2]
    nnq2.learning_rate = combination[3]

    parameters = nnq2.initializeParameters()
    losses, accuracy, parameters = nnq2.train(train_data_normal, train_data_labels, parameters)

    plot_lossos_and_accuracy(nnq2, losses, accuracy)

    personal_test_run(nnq2, test_data_normal, test_data_labels, parameters)

    test_csv_normal, mean, var = data_normal(test_csv, nnq2.mean, nnq2.var)
    __, parameters = nnq2.forwardPropagation(test_csv_normal, False, parameters)
    prediction = label_prediction(parameters["Y_hat"])

    df = pd.DataFrame(prediction)
    df.to_csv('lr_pred.csv', index=False)



def question_3_preperation(train_csv):
    nnq3 = NeuralNetworkQ3()
    train_data_normal, train_data_labels, validation_data_normal, validation_data_labels, test_data_normal, test_data_labels = prepare_data_for_model(nnq3, train_csv)
    parameters = nnq3.initializeParameters()

    # Iteration lists
    batch_size_list = [10, 20, 50, 100, 200, 400]
    lamb_list = [0.5, 0.2, 0.1, 0.01]
    epochs_list = [5, 15]
    learning_rate_list = [0.001, 0.005, 0.01, 0.05]
    hidden_neurons_list = [16, 32, 64, 128]

    validation_accuracy_list = []
    validation_batch_size_list = []
    validation_lamb_list = []
    validation_epochs_list = []
    validation_learning_rate_list = []
    validation_hidden_neurons_list = []
    combination_list = itertools.product(batch_size_list, lamb_list, epochs_list, learning_rate_list,
                                         hidden_neurons_list)

    for combination in combination_list:
        validation_batch_size_list.append(combination[0])
        validation_lamb_list.append(combination[1])
        validation_epochs_list.append(combination[2])
        validation_learning_rate_list.append(combination[3])
        validation_hidden_neurons_list.append(combination[4])

        parameters = nnq3.initializeParameters()
        nnq3.batch_size = combination[0]
        nnq3.lamb = combination[1]
        nnq3.epochs = combination[2]
        nnq3.learning_rate = combination[3]
        nnq3.hidden_neurons = combination[4]

        # Train
        losses, accuracy, parameters = nnq3.train(train_data_normal, train_data_labels, parameters)

        # Validation
        __, parameters = nnq3.forwardPropagation(validation_data_normal, validation_data_labels, parameters)
        validation_prediction = label_prediction(parameters["Y_hat"])
        validation_test_accuracy = compute_accuracy(validation_prediction, validation_data_labels)
        validation_accuracy_list.append(validation_test_accuracy)
        print("For combination: {}, accuracy is: {}".format(combination, validation_test_accuracy))

    """write validation data -hyper parameters- to csv file"""
    df = pd.DataFrame({'Validation Accuracy': validation_accuracy_list, 'Batch_size': validation_batch_size_list,
                       'Lambda': validation_lamb_list, 'Epochs': validation_epochs_list,
                       'Learning Rate': validation_learning_rate_list,
                       'Hidden Neurons': validation_hidden_neurons_list})
    df.to_csv('our_test_file_q3.csv', index=False)

    # plt.figure()
    # plt.plot(validation_accuracy_list)
    # plt.xlabel("Runs")
    # plt.ylabel("Loss value")
    # plt.show()

    i = validation_accuracy_list.index(max(validation_accuracy_list))
    best_combination = [validation_batch_size_list[i], validation_lamb_list[i], validation_epochs_list[i], validation_learning_rate_list[i], validation_hidden_neurons_list[i]]

    return best_combination



def question_3(train_csv, test_csv, best_combination=[]):
    nnq3 = NeuralNetworkQ3()

    train_data_normal, train_data_labels, validation_data_normal, validation_data_labels, test_data_normal, test_data_labels = prepare_data_for_model(nnq3, train_csv)

    best_combination = [20, 0.01, 35, 0.005, 16]
    combination = best_combination

    parameters = nnq3.initializeParameters()
    nnq3.batch_size = combination[0]
    nnq3.lamb = combination[1]
    nnq3.epochs = combination[2]
    nnq3.learning_rate = combination[3]
    nnq3.hidden_neurons = combination[4]

    losses, accuracy, parameters = nnq3.train(train_data_normal, train_data_labels, parameters)

    plot_lossos_and_accuracy(nnq3, losses, accuracy)

    personal_test_run(nnq3, test_data_normal, test_data_labels, parameters)

    test_csv_normal, mean, var = data_normal(test_csv, nnq3.mean, nnq3.var)
    __, parameters = nnq3.forwardPropagation(test_csv_normal, False, parameters)
    prediction = label_prediction(parameters["Y_hat"])

    df = pd.DataFrame(prediction)
    df.to_csv('NN_pred.csv', index=False)


def main():
    train_csv = pd.read_csv(r"train.csv")
    test_csv = pd.read_csv(r"test.csv")

    # -- Q1 --
    question_1(train_csv)
    # --------

    """
    Note for Q2, Q3
    run 2 first lines to find the best hyperparameters
    rub only 3rd line for train the best model we found
    """

    # -- Q2 --
    # best_combination = question_2_preperation(train_csv)
    # question_2(train_csv, test_csv, best_combination)  # run after preperation
    question_2(train_csv, test_csv, [])
    # --------

    # -- Q3 --
    # best_combination = question_3_preperation(train_csv)
    # question_3(train_csv, test_csv, best_combination)  # run after preperation
    question_3(train_csv, test_csv, [])

if __name__ == '__main__':
    main()
