Instructions:

- Initialize the Regressor with the Degree of Polynomial
- .name() -> set's the name of the Model 
- .train() -> trains the Model on the Data.
  Accepts the following parameters:
    * "epochs" = number of epochs 
    * "trainingData" = tuple: first element is a feature matrix and second element as target vector
    * "validationData" = tuple: first element is a feature matrix, and second element as target vector
    * "batchSize"
- .save() -> accepts path to save model weights (requires name of the model to be saved)
