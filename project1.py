from string import punctuation, digits
import numpy as np
import random

# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.

    # test cases
    # Feature Vector: [0.11310995 0.6071438  0.51690602 0.97410408 0.42965912 0.46070069
    #          0.9182426  0.03623677 0.78338951 0.54704968]
    # Label: 1.0
    # Theta: [0.44204776 0.08235281 0.09672938 0.05132922 0.11637132 0.10853033
    #         0.05445184 1.37981375 0.06382521 0.09139938]
    # Theta_0: 0.5
    """

    y = theta @ feature_vector + theta_0 #@ is short for dot product
    return max(0, 1 - y * label)



def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """

    ys = feature_matrix @ theta + theta_0
    loss = np.maximum(1 - ys * labels, np.zeros(len(labels)))
    return np.mean(loss)


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.

    # Test case:
    # perceptron_single_step_update input:
    # feature_vector: [ 0.15734784  0.30962619  0.45018307 -0.35750355 -0.01120774 -0.45445275
    # 0.1508538   0.00223642 -0.36760953 -0.48241598]
    # label: -1
    # theta: [-0.09181657 -0.37214022 -0.46489302  0.37119251  0.03532131 -0.36824484
    # -0.43174457 -0.09373253  0.36249025 -0.42288127]
    # theta_0: 0.31137850783903953
    # perceptron_single_step_update output is (['-0.2491644', '-0.6817664', '-0.9150761', '0.7286961',
    # '0.0465290', '0.0862079', '-0.5825984', '-0.0959690', '0.7300998', '0.0595347'], '-0.6886215')
    """
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 0:
        current_theta += label * feature_vector
        current_theta_0 += label
    return (current_theta, current_theta_0)


    # #alternative 2
    # if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 1e-7:
    #     return (current_theta + label * feature_vector, current_theta_0 + label)
    # return (current_theta, current_theta_0)


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.

    #test case:
    # perceptron input:
    # feature_matrix: [[ 0.09276226  0.29969414 -0.14348213  0.41037267  0.34980009 -0.01307742
    # -0.46465955 -0.36809754 -0.02475357 -0.27320309]
    # [ 0.04573037 -0.12274694 -0.04114562 -0.33292924 -0.49403575 -0.08245804
    # 0.02159775  0.46524822  0.4276589   0.07692704]
    # [ 0.1115032  -0.29177014  0.35204286 -0.26654687  0.41665848 -0.21609301
    # -0.1187601  -0.03536978  0.45003094 -0.01691681]
    # [-0.49099629  0.00220846 -0.44502155 -0.10131711  0.33967393 -0.1499547
    # -0.26106462 -0.1220876   0.00164181  0.41068832]
    # [ 0.29024384 -0.49501723 -0.05938192  0.23892441  0.39758512 -0.29669152
    # -0.04286374  0.08031101  0.33260653  0.16070966]]
    # labels: [-1  1  1 -1  1]
    # T: 5
    # perceptron output is ['0.7812401', '-0.4972257', '0.3856396', '0.3402415', '0.0579112', '-0.1467368',
    # '0.2182009', '0.2023986', '0.3309647', '-0.2499787']
    """

    (nsamples, nfeatures) = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_0 = 0.0
    for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0)
    return (theta, theta_0)


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.

    # test case:
    # average_perceptron input:
    # feature_matrix: [[ 0.00842127 -0.47513722 -0.25169748 -0.09554435 -0.12349877 -0.24327709
    # 0.15239019  0.49517927 -0.39874755 -0.22925087]
    # [-0.23062165 -0.44726918  0.47129341  0.25879761  0.35045375  0.42302778
    # 0.15644843  0.40672444 -0.20555914  0.30063678]
    # [-0.44576204 -0.34525493  0.23226196  0.49400633  0.47037901 -0.0131673
    # -0.19652864 -0.00798211 -0.25046913  0.14531536]
    # [ 0.10465833 -0.15852725 -0.43872975  0.35941197  0.23726859 -0.38465724
    # 0.3435205  -0.19173274  0.2108306  -0.42025546]
    # [ 0.4468699  -0.3425897  -0.27529389 -0.00743711  0.26953853 -0.09754821
    # -0.29158533  0.179751    0.32186342  0.39844366]]
    # labels: [-1  1  1  1  1]
    # T: 5
    # average_perceptron output is ['0.2821505', '0.0606001', '-0.1154400', '0.6640049', '0.8772379', '-0.0366846',
    # '-0.3582421', '-0.7029819', '0.9419971', '0.4939048']
    """
    (nsamples, nfeatures) = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_sum = np.zeros(nfeatures)
    theta_0 = 0.0
    theta_0_sum = 0.0
    for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0)
            theta_sum += theta
            theta_0_sum += theta_0
    return (theta_sum / (nsamples * T), theta_0_sum / (nsamples * T))


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.

    # test case:
    # pegasos_single_step_update input:
    # feature_vector: [ 0.05122659  0.34014854  0.49386615  0.03604035 -0.09206133  0.10819428
    # 0.0102943   0.25442957  0.3942945   0.11803004]
    # label: 1
    # L: 0.8255698508220228
    # eta: 0.2320471458287635
    # theta: [-0.16537425  0.30978265  0.12336812 -0.19674994 -0.3389787  -0.4924394
    # -0.19111944  0.30472518  0.26771502  0.17334271]
    # theta_0: 1.2678377365491778
    # pegasos_single_step_update output is (['-0.1336933', '0.2504372', '0.0997343', '-0.1590583', '-0.2740402', '-0.3981022',
    # '-0.1545065', '0.2463486', '0.2164286', '0.1401353'], '1.2678377')
    """
    mult = 1 - (eta * L)
    if label * (np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        return ((mult * current_theta) + (eta * label * feature_vector),
                (current_theta_0) + (eta * label))
    return (mult * current_theta, current_theta_0)


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    # test case:
    # pegasos input:
    # feature_matrix: [[ 0.1837462   0.29989789 -0.35889786 -0.30780561 -0.44230703 -0.03043835
    # 0.21370063  0.33344998 -0.40850817 -0.13105809]
    # [ 0.08254096  0.06012654  0.19821234  0.40958367  0.07155838 -0.49830717
    # 0.09098162  0.19062183 -0.27312663  0.39060785]
    # [-0.20112519 -0.00593087  0.05738862  0.16811148 -0.10466314 -0.21348009
    # 0.45806193 -0.27659307  0.2901038  -0.29736505]
    # [-0.14703536 -0.45573697 -0.47563745 -0.08546162 -0.08562345  0.07636098
    # -0.42087389 -0.16322197 -0.02759763  0.0297091 ]
    # [-0.18082261  0.28644149 -0.47549449 -0.3049562   0.13967768  0.34904474
    # 0.20627692  0.28407868  0.21849356 -0.01642202]]
    # labels: [-1 -1 -1  1 -1]
    # T: 10
    # L: 0.1456692551041303
    # pegasos output is ['-0.0850387', '-0.7286435', '-0.3440130', '-0.0560494', '-0.0260993', '0.1446894',
    # '-0.8172203', '-0.3200453', '-0.0729161', '0.1008662']
    """
    (nsamples, nfeatures) = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_0 = 0
    count = 0
    for t in range(T):
        for i in get_order(nsamples):
            count += 1
            eta = 1.0 / np.sqrt(count)
            (theta, theta_0) = pegasos_single_step_update(
                feature_matrix[i], labels[i], L, eta, theta, theta_0)
    return (theta, theta_0)

# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    (nsamples, nfeatures) = feature_matrix.shape
    predictions = np.zeros(nsamples)
    for i in range(nsamples):
        feature_vector = feature_matrix[i]
        prediction = np.dot(theta, feature_vector) + theta_0
        if (prediction > 0):
            predictions[i] = 1
        else:
            predictions[i] = -1
    return predictions

    #alternative 1
    #use the fact that a boolean will be implicitly casted 
    #by NumPy into 0 or 1 when mutiplied by a float. 
    #We identified 0 to the range[-e, +e] for numerical reasons.
    #return (feature_matrix @ theta + theta_0 > 1e-7) * 2.0 - 1 

def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_predictions = classify(train_feature_matrix, theta, theta_0)
    val_predictions = classify(val_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(val_predictions, val_labels)
    return (train_accuracy, validation_accuracy)


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Your code here { 
    stopwords = open('stopwords.txt', 'r').read().split()
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in stopwords:
                continue
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary
    # Your code here }
    
    # dictionary = {} # maps word to unique index
    # for text in texts:
    #     word_list = extract_words(text)
    #     for word in word_list:
    #         if word not in dictionary:
    #             dictionary[word] = len(dictionary)
    # return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here { 
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix
    # Your code here }
   
    # num_reviews = len(reviews)
    # feature_matrix = np.zeros([num_reviews, len(dictionary)])

    # for i, text in enumerate(reviews):
    #     word_list = extract_words(text)
    #     for word in word_list:
    #         if word in dictionary:
    #             feature_matrix[i, dictionary[word]] = 1
    # return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
