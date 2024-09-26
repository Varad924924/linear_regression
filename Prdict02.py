import numpy as np

x_input = input("Enter the value of X (comma-separated): ")
y_input = input("Enter the value of Y (comma-separated): ")


x = list(map(float, x_input.split(',')))
y = list(map(float, y_input.split(',')))


class Linearregression:
    def fit(self, x, y):
        self.arrayx = np.array(x)
        self.xm = np.mean(self.arrayx)
        self.arrayy = np.array(y)
        self.ym = np.mean(self.arrayy)
        self.numerator = np.sum((self.arrayx - self.xm) * (self.arrayy - self.ym))
        self.denominator = np.sum((self.arrayx - self.xm) ** 2)
        self.b1 = self.numerator / self.denominator
        self.b0 = self.ym - (self.b1 * self.xm)
        return self.b0, self.b1

    def predict(self, x):
        ypre = self.b0 + (self.b1 * np.array(x))
        return ypre

    def residual(self, ypre, y):
        residuals = np.sum((np.array(y) - np.array(ypre))**2)
        return residuals

    def tss(self, y):
        tss = np.sum(np.array(y) - ((np.mean(y))**2))
        return tss

    def loss(self, tss, residual):
        r = 1 - (residual / tss)
        return r


model = Linearregression()
b0, b1 = model.fit(x, y)


print(f"b0: {b0}")
print(f"b1: {b1}")


ypred = model.predict(x)
residuals = model.residual(ypred, y)
tss = model.tss(y)
r = model.loss( tss , residuals)


print(f"Residuals (TRS): {residuals}")
print(f"Total  (TSS): {tss}")
print(f"Total  (loss): {r}")

if r == 1:
    print("Model is fit")
else:
    print("Model is not fit")