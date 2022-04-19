### code base: ai.berkeley.edu

import nn


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.IMAGE_SIZE = 784  # num_input_features???
        self.NUM_CLASSES = 10

        # number of hidden leyers recommended between 1 and 3.
        self.nHL = 1

        # Learning rate: between -0.001 and -1.0.
        self.learn_rate = -1

        # number of units per hidden layer or hidden layer sizes recommended between 10 and 400.
        self.num_hid_units = 500

        # batch size: between 1 and the size of the dataset.
        self.batch_size = 2000

        # initialize the weights and bias matrix to random numbers
        # input-to-hidden weight matrix
        self.Wji_matrix = nn.Parameter(self.IMAGE_SIZE, self.num_hid_units)
        self.bji = nn.Parameter(1, self.num_hid_units)

        # hidden-to-output weight matrix
        self.Wkj_matrix = nn.Parameter(self.num_hid_units, self.NUM_CLASSES)
        self.bkj = nn.Parameter(1, self.NUM_CLASSES)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # input to hidden layer
        # sum over weighted features and add bias
        netj = nn.AddBias(nn.Linear(x, self.Wji_matrix), self.bji)

        # pass through the ReLU activation unit
        # yj is a Node with the same shape as netj, but no negative entries
        yj = nn.ReLU(netj)

        # hidden layer to output layer
        # sum over the weighted outputs of hidden units and add bias
        netk = nn.AddBias(nn.Linear(yj, self.Wkj_matrix), self.bkj)

        # output of the network with shape (batch_size x 10) before the softmax unit
        logits = netk

        return logits


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # calculate the scores/logits
        logits = self.run(x)

        # cross entropy loss function outputs a scalar Node (containing a single floating-point number) ???????
        return nn.SoftmaxLoss(logits, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # check for stopping criteria otherwise use the next batch of training dataset
        # run the updated model on validation dataset
        while (dataset.get_validation_accuracy() <= 0.98):
            for x, y in dataset.iterate_once(batch_size=self.batch_size):
                # calculate the loss function for the current batch
                loss = self.get_loss(x, y)

                # claculate the gradianet of loss function with respect to all the weights and biases in the network
                # tf.gradients(ys, xs) constructs symbolic derivatives of sum of ys w.r.t. x in xs.
                # xs is a list of tensors
                g_list = nn.gradients(loss, [self.Wji_matrix, self.bji, self.Wkj_matrix, self.bkj])
                dLoss_dWji, dLoss_dbji, dLoss_dWkj, dLoss_dbkj = g_list

                # update the weights and biases accordingly
                self.Wji_matrix.update(dLoss_dWji, self.learn_rate)
                self.bji.update(dLoss_dbji, self.learn_rate)
                self.Wkj_matrix.update(dLoss_dWkj, self.learn_rate)
                self.bkj.update(dLoss_dbkj, self.learn_rate)

                print("Accuracy on Validation is: {}" .format(dataset.get_validation_accuracy()))
