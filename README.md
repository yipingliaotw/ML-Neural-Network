#Machine Learning Neural Network
## Radial Basis Network

### Training
- Input:
  - Parameters:
    - number of neurals
    - regression parameters lambda
    - covariance of Gaussian basis function
  - Data
    - training data
    - label of training data

- Output:
  - training results: weight of output layer

- Usage:
```
rbf = RBF(NumofNeurons=30,_lambda=0,_covar = 0.1)
rbf.Train(train_data,train_label)
```

### Test
- Input:
  - Data
    - test data
- Output:
  - test results

- Usage:
  ```
  result = rbf.Test(test_data)
  ```

## Extreme Learning Machine
### Training
- Input:
  - Parameters:
    - number of neurals
    - regression parameters lambda
  - Data
    - training data
    - label of training data

- Output:
  - training results: weight of output layer

- Usage:
```
elm= ELM(NumofHiddenNeurons = 10,_lambda = 0)
elm.Train(train_data,train_label)
```

### Test
- Input:
  - Data
    - test data
- Output:
  - test results

- Usage:
  ```
  result = elm.Test(test_data)
  ```
