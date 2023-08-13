# linear-reg
This linear regression class has 3 functions:
# Fit Function:
This is where all the math happens
# Predict Function:
This is where we predict the outcome given some data
# Score Fucntion:
This function evaluates the predictions made based on mean squared error technique

## Techniques Used:

# Gradient-Descent: 
Batch gradient descent for (weights and bias)
# Early Stopping: 
Prevents overfitting by monitoring validation loss and stops training if the stopping criteria is met.
# L2-Regularization:
L2 regularization is used with gradient-descent step to prevent overfitting. It stops the weights from growing too large.
# Loss Function (Mean Squared Error):
MSE is used in the score function to measure the difference between the predicted values and the actual target values.
# Calculus and Linear Algebra:
The fit function relies on these
# Data Splitting (Training and Validation Sets): 
The code splits the input data into training and validation sets to train the model and monitor its performance on unseen data.

# Visualization (Matplotlib): 
The code uses Matplotlib for visualizing the loss curve during training.
