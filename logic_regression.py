import numpy as np
import matplotlib.pyplot as plt


class linearRegresson:
    def __init__(self):
        self.coefficient = None
        self.intercept = None
        self.r2_score = None

    def fit(self, X, y):
        n = len(X)
        x_b = np.c_[np.ones((n,1)), X]
        self.coefficient = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
        self.intercept = self.coefficient[0]
        y_pred = self.predict(X)
        #R2
        self.r2_score = 1 - (np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2))
        self.y_pred = y_pred
        
    def predict(self, X):
        X_b = np.c_[np.ones((len(X),1)), X]
        return X_b.dot(self.coefficient)


X_simple = np.array([1,2,3,4,5,6,7,8,9,10])
y_simple = np.array([2,4,5,4,5,7,8,9,10,12])

slr = linearRegresson()
slr.fit(X_simple, y_simple)
print("Coefficient:", slr.coefficient)
print("Intercept:", slr.intercept)
print("R2 Score:", slr.r2_score)

plt.scatter(X_simple, y_simple, color='blue', label='Data Points')
plt.plot(X_simple, slr.y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()