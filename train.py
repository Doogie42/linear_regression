import math
import json
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from predict import estimate_price


class LinearRegression():
    """Class that perform a linear regression using gradient descent"""
    def __init__(self, data: pd.DataFrame, learning_rate: float, max_epoch: float,
                 graphic: bool = True) -> None:
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.graphic = graphic
        self.error_history = np.empty(max_epoch)
        self.error_history.fill(-1.0)
        self.theta = {
            "0": 0.0,
            "1": 0.0
        }
        self.theta_std = {
            "0": 0.0,
            "1": 0.0
        }
        self.data_std = data.copy(deep=True)
        self.data = data
        self.standardize_data(data)

    def destandardize_theta(self) -> None:
        """destandardize theta"""
        prv_theta1 = self.theta_std["1"]
        self.theta["1"] = self.theta_std["1"] * (self.std_price / self.std_km)
        self.theta["0"] = self.mean_price -\
            prv_theta1 * self.std_price * self.mean_km / self.std_km +\
            self.theta_std["0"] * self.mean_km

    def standardize_data(self, data: pd.DataFrame) -> None:
        """Standardize the data before linear regression"""
        self.mean_km = np.mean(data["km"])
        self.mean_price = np.mean(data["price"])
        self.std_km = np.std(data["km"])
        self.std_price = np.std(data["price"])

        assert self.std_km, \
            "km std is 0, can't standardize value" + \
            "=> do you have only 1 value or only the same value ?"
        assert self.std_price, \
            "price std is 0, can't standardize value" + \
            "=> do you have only 1 value or only the same value ?"
        self.data_std["km"] = data["km"].apply(
            lambda x: (x - self.mean_km) / self.std_km
        )
        self.data_std["price"] = data["price"].apply(
            lambda x: (x - self.mean_price) / self.std_price
        )

    def perform_linear(self) -> (float):
        """Perform a linear regression on the given data"""
        if self.graphic:
            self.set_up_plot(self.data)
        for i in tqdm(range(self.max_epoch)):
            sum_theta0 = sum(
                estimate_price(mileage, self.theta_std["0"], self.theta_std["1"]) - price
                for mileage, price in zip(self.data_std["km"], self.data_std["price"])
            )
            tmptheta0 = self.learning_rate * sum_theta0 / len(self.data)

            sum_theta1 = sum(
                (estimate_price(mil, self.theta_std["0"], self.theta_std["1"]) - price) * mil
                for mil, price in zip(self.data_std["km"], self.data_std["price"])
            )
            tmptheta1 = self.learning_rate * sum_theta1 / len(self.data)
            self.theta_std["0"] -= tmptheta0
            self.theta_std["1"] -= tmptheta1

            self.destandardize_theta()
            self.error_history[i] = self.calculate_mse_error(self.data)
            if self.graphic and not i % 10:
                self.update_plot(self.data)

        if self.graphic:
            plt.waitforbuttonpress()
        return (self.theta["0"], self.theta["1"])

    def set_up_plot(self, data: pd.DataFrame) -> None:
        """Prepare for interactive plot, using non standardized data"""
        plt.ion()
        self.fig = plt.figure()

        ax_ln = self.fig.add_subplot(211)
        ax_ln.set_xlabel("price")
        ax_ln.set_ylabel("km")
        self.line1_ln, = ax_ln.plot(data["km"], data["price"], "bo")
        self.line1_ln.set_ydata(data["price"])
        y = self.theta["1"] * data["km"] + self.theta["0"]
        self.line2_ln, = ax_ln.plot(data["km"], y)

        self.ax_mse = self.fig.add_subplot(212)
        self.ax_mse.set_xlabel("epoch number")
        self.ax_mse.set_ylabel("mse error")
        self.ax_mse.set_ylim(0)
        self.line1_mse, = self.ax_mse.plot(np.arange(0, self.max_epoch),
                                           self.error_history, "x")
        self.line1_mse.set_ydata(self.error_history)

        self.fig.canvas.draw()

    def update_plot(self, data: pd.DataFrame) -> None:
        """Update plot after each iteration, using non standardized data"""
        y = self.theta["1"] * data["km"] + self.theta["0"]

        self.line2_ln.set_ydata(y)

        max_mse = np.nanmax(self.error_history) * 2
        min_mse = 0.0
        max_step = int(max_mse / 5)
        step = max(1, max_step)
        max_val = max(5, max_mse)
        ytick = np.arange(min_mse, max_val, step)
        self.line1_mse.set_ydata(self.error_history)
        self.ax_mse.set_yticks(ytick)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def calculate_mse_error(self, data: pd.DataFrame) -> float:
        """Calculate mean square error of current iteration,
        using destandardize theta"""
        sum_squared_error = sum(
            (estimate_price(mileage, self.theta["0"], self.theta["1"]) - price) ** 2
            for mileage, price in zip(data["km"], data["price"])
        )
        sum_squared_error = sum_squared_error / len(data)
        return sum_squared_error


def load(path: str) -> pd.DataFrame:
    """Read csv file and return it as a panda Dataframe.
    Rais ValueError if problem"""
    if path.rfind(".csv") == -1:
        raise ValueError("Wrong type of file")
    data = pd.read_csv(path)
    return data


def calculate_rss_tot(data: pd.DataFrame,
                      theta0: float, theta1: float) -> float:
    """Calculate the total sum of squares"""
    mean_price = np.mean(data["price"])
    rss_tot = sum(
        (estimate_price(mileage, theta0, theta1) - mean_price) ** 2
        for mileage in data["km"]
    )
    return rss_tot


def calculate_rss(data: pd.DataFrame, theta0: float, theta1: float) -> float:
    """Calculate the  residual sum of squares"""
    rss = sum(
        (estimate_price(mileage, theta0, theta1) - price) ** 2
        for mileage, price in zip(data["km"], data["price"])
    )
    return rss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="data file to load", type=str,
                        nargs="?", default="./data.csv")
    parser.add_argument("-m", help="max epochs", type=int,
                        nargs="?", default=1000)
    parser.add_argument("-l", help="learning rate", type=float,
                        nargs="?", default=0.009)
    parser.add_argument("-g", help="enable graphic", action="store_true")
    args = parser.parse_args()

    assert args.m > 0, "need a strictly positive value for epochs"
    assert args.l > 0 and args.l <= 1, "need a learning rate between 0 and 1"
    try:
        data = load(args.d)
    except Exception as e:
        print(e)
        return
    assert "km" in data
    assert "price" in data
    assert not np.isnan(data["km"])[0], "Empty value in km column"
    assert not np.isnan(data["price"])[0], "Empty value in price column"

    linear = LinearRegression(data, args.l, args.m, args.g)

    theta0, theta1 = linear.perform_linear()
    rss = calculate_rss(data, theta0, theta1)
    rss_tot = calculate_rss_tot(data, theta0, theta1)
    r_squared = 1 - rss / rss_tot
    print(f"theta0: {theta0}")
    print(f"theta1: {theta1}")
    print(f"Approximated equation : price(km) = {theta1} * km",
          "+" if theta0 > 0 else "-", f"{abs(theta0)}")
    print(f"Mean squared error : {linear.error_history[-1]}")
    print(f"Average deviation : {math.sqrt (linear.error_history[-1]) }")
    print(f"Coefficient of determination (r2) {r_squared}")
    if r_squared < 0 or r_squared > 1:
        print("Model has no sense ->",
              "Maybe there is no correlation between data point",
              "or the model is wrong")

    with open("theta.json", "w", encoding="utf-8") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)


if __name__ == "__main__":
    main()


# https://www.statology.org/rmse-vs-r-squared/


# destandardize theta:
# y = ax + b
# y = y - meany / stdy
# x = x - meanx / stdx
# => (y - meany) / stdy = a * (x - meanx) / stdx + b
# => y = a * stdy * (x - meanx) / stdx + b * stdy + meany
# => y = ax stdy / stdx - a *stdy * meanx / stdx + b *stdy + meany

# y' = a'x + y'
# => a' = a * stdy / stdx
# => b' = - a * stdy * meanx / stdx + meany + b * stdy
# => or b' = - a * stdy * meanx / stdx + meany , b * stdy can be omitted
