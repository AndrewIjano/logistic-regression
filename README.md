# Logistic Regression

## Description
Implements a logistic regression algorithm using gradient descent.

## Usage
```
./logistic_regresion_test.py [-h] [-N N] [--batch_size B_SIZE]
                                  [--learning_rate L_RATE] [--iterations ITER]
                                  [--3d] [--random]

optional arguments:
  -h, --help            show this help message and exit
  -N N                  the number of samples used
  --batch_size B_SIZE   the batch size used
  --learning_rate L_RATE
                        the fitting learning rate used
  --iterations ITER     the number of iterations used in fitting
  --3d                  test with a 3D set of points
  --random              use random means and covariances
```

## Examples

```
./logistic_regresion_test.py
```

![](img/Figure_2d_1.png) 

![](img/Figure_2d_2.png)


```
./logistic_regresion_test.py --random -N=50000 --batch_size=50 --iterations=200
```

![](img/Figure_2d_3.png) 

![](img/Figure_2d_4.png)

```
./logistic_regresion_test.py --3d
```

![](img/Figure_3d_1.png) 

![](img/Figure_3d_2.png)

```
./logistic_regresion_test.py --3d --random
```

![](img/Figure_3d_3.png) 

![](img/Figure_3d_4.png)