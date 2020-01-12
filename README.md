# FastAI Sklearn Wrapper
This project has two goals:
- Allow me to review the FastAI Syntax
- Provide an easy interface for those more familiar with Sklearn and Keras style (such as `fit` and `predict` methods). 

## TODO: Learners:
I would like to implement the following learners:
- `TabularClassifier`: A learner that has as inputs tabular data. The output should be classes. Categorical features will be encoded using embeddings.
- `TabularRegressor`: A learner that has as inputs tabular data. The output should be numerical. Categorical features will be encoded using embeddings.
- `ImageClassifier` (or `CNNClassifier` or `CnnClassifier` or `ConvolutionalNeuralNetworkClassifier` or `ConvNetClassifier`): A general purpose image classifier. There will be attributes with the options of architectures. By default it will use preloaded weights from another architecture trained on ImageNet. 
- `ImageRegressor` (or `CNNRegressor` or `CnnRegressor` or `ConvolutionalNeuralNetworkRegressor` or `ConvNetRegressor`): A general purpose image regressor. There will be attributes with the options of architectures. By default it will use preloaded weights from another architecture trained on ImageNet. 
- `TextClassifier` (or `RNNClassifier`): A learner that has as inputs text data. The output should be classes. Categorical features will be encoded using embeddings.
- `TabularRegressor` (or `RNNRegressor`): A learner that has as inputs text data. The output should be classes. Categorical features will be encoded using embeddings.

## TODO: Learner Methods

### Sklearn Methods
- `fit(X, y)`: Maybe it will look more like fit from keras, which would be equivalent to `partial_fit(X, y)` in sklearn (i.e., fit can be called multiple times). Still checking options...
- `predict(X)`
- `score(X, y)`
- `predict_proba(X)`

### New Methods
I will consider implementing the following methods:
  - `fit_from_dataframe(df)` inspired on ImageDataGenerator of Keras)
  - `fit_from_directory(path)`
  - `lr_finder`

## Note
I understand that a wrapper like this would put more distance from the source code made in Pytorch. Although it feels like a wrapper of a wrapper, this package should be used for those in transition from Sklearn and Keras environment to FastAI. For them it will be faster to use the wrappers provided here instead of having to learn the new terminologies. I highly suggest inspecting the source code of the wrapper to check how the implementation would originally be done on FastAI syntax. 
