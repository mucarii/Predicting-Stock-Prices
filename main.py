# %% [bibliotecas]
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# %%
dates = []
prices = []


# %%
def get_data(GME_stock):
    with open(GME_stock, "r") as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split("-")[0]))  # Ano
            prices.append(float(row[2]))  # Preço


# %%
def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))  # Convertendo em uma coluna

    svr_lin = SVR(kernel="linear", C=1e3)
    svr_poly = SVR(kernel="poly", C=1e3, degree=2)
    svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)  # gamma = 0.1
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color="black", label="Dados")
    plt.plot(dates, svr_lin.predict(dates), color="green", label="Linear")
    plt.plot(dates, svr_poly.predict(dates), color="blue", label="Polinomial")
    plt.plot(dates, svr_rbf.predict(dates), color="red", label="RBF")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.legend()
    plt.show()

    return (
        svr_lin.predict(np.array([[x]]))[0],
        svr_poly.predict(np.array([[x]]))[0],
        svr_rbf.predict(np.array([[x]]))[0],
    )


# %%
get_data("GME_stock.csv")
predictions = predict_prices(dates, prices, 29)

# %%
print(predictions)
