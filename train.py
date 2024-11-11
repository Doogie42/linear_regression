import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import argparse
from predict import estimate_price


class LinearRegression():
    def __init__(self, learning_rate: float, max_epoch: float,
                 graphic: bool = True) -> None:
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.graphic = graphic
        self.error_history = np.empty(max_epoch)
        self.error_history.fill(-1.0)
        self.theta0 = 0.0
        self.theta1 = 0.0

    def perform_linear(self, data: pd.DataFrame) -> (float):
        if self.graphic:
            self.set_up_plot(data)
        for i in tqdm(range(self.max_epoch)):
            sum_theta0 = sum([
                estimate_price(mileage, self.theta0, self.theta1) - price
                for mileage, price in zip(data["km"], data["price"])
            ])
            tmptheta0 = self.learning_rate * sum_theta0 / len(data)

            sum_theta1 = sum([
                (estimate_price(mil, self.theta0, self.theta1) - price) * mil
                for mil, price in zip(data["km"], data["price"])
            ])
            tmptheta1 = self.learning_rate * sum_theta1 / len(data)

            self.error_history[i] = self.calculate_mse_error(data)
            self.theta0 -= tmptheta0
            self.theta1 -= tmptheta1
            if self.graphic and not i % 10:
                self.update_plot(data)

        if self.graphic:
            plt.waitforbuttonpress()
        return (self.theta0, self.theta1)

    def calculate_mse_error(self, data: pd.DataFrame) -> float:
        sum_squared_error = sum([
            (estimate_price(mileage, self.theta0, self.theta1) - price) ** 2
            for mileage, price in zip(data["km"], data["price"])
        ])
        sum_squared_error = sum_squared_error / len(data)
        return sum_squared_error

    def set_up_plot(self, data: pd.DataFrame) -> None:
        plt.ion()
        self.fig = plt.figure()

        ax_ln = self.fig.add_subplot(211)
        self.line1_ln, = ax_ln.plot(data['km'], data['price'], 'bo')
        self.line1_ln.set_ydata(data['price'])
        y = self.theta1 * data['km'] + self.theta0
        self.line2_ln, = ax_ln.plot(data['km'], y)

        self.ax_mse = self.fig.add_subplot(212)
        self.ax_mse.set_ylim(0)
        self.line1_mse, = self.ax_mse.plot(np.arange(0, self.max_epoch),
                                           self.error_history, 'x')
        self.line1_mse.set_ydata(self.error_history)

        self.fig.canvas.draw()

    def update_plot(self, data: pd.DataFrame) -> None:
        y = self.theta1 * data['km'] + self.theta0

        self.line2_ln.set_ydata(y)

        max_mse = np.nanmax(self.error_history)
        min_mse = 0.0
        max_step = int(max_mse / 5)
        step = max(1, max_step)
        max_val = max(5, max_mse)
        ytick = np.arange(min_mse, max_val, step)
        self.line1_mse.set_ydata(self.error_history)
        self.ax_mse.set_yticks(ytick)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def load(path: str) -> pd.DataFrame:
    """Read csv file and return it as a panda Dataframe.
    Rais ValueError if problem"""
    if path.rfind(".csv") == -1:
        raise ValueError("Wrong type of file")
    try:
        data = pd.read_csv(path)
    except Exception:
        raise ValueError("Wrong encoding of file")
    print(f"Loading dataset of dimension {data.shape}")
    return data


def main():
    parser.add_argument("-d", help="data file to load", type=str,
                        nargs="?", default="./data.csv")
    parser.add_argument("-m", help="max epochs", type=int,
                        nargs="?", default=1000)
    parser.add_argument("-l", help="learning rate", type=float,
                        nargs="?", default=0.009)
    parser.add_argument("-g", help="enable graphic", action="store_true")
    args = parser.parse_args()

    data = load(args.d)
    mean_km = np.mean(data['km'])
    mean_price = np.mean(data['price'])

    std_km = np.std(data['km'])
    std_price = np.std(data['price'])

    data["km"] = data["km"].apply(lambda x: (x - mean_km) / std_km)
    data["price"] = data["price"].apply(lambda x: (x - mean_price) / std_price)

    linear = LinearRegression(args.l, args.m, args.g)
    theta0, theta1 = linear.perform_linear(data)

    prv_theta1 = theta1
    theta1 = theta1 * (std_price / std_km)
    theta0 = mean_price - prv_theta1 * std_price * mean_km / std_km +\
        theta0 * mean_km
    with open('theta.json', 'w') as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()

# destandardize theta:

# y = ax + b
# y,x = y - mean / std
# => (y - meany) / stdy = a * (x - meanx) / stdx + b
# => y = a * stdy * (x - meanx) / stdx + b * stdy + meany
# => y = ax stdy / stdx - a *stdy * meanx / stdx + b *stdy + meany
# => a' = a * stdy / stdx
# => b' = - a * stdy * meanx / stdx + meany + b * stdy
# => or b' =  - a * stdy * meanx / stdx + meany , b * stdy can be omitted
