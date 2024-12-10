import json


def estimate_price(mileage: int, theta0: float, theta1: float) -> float:
    return float(theta0 + (theta1 * mileage))


def main():
    try:
        with open('./theta.json', 'r', encoding="utf-8") as f:
            theta = json.load(f)
    except Exception as e:
        print(e)
        return
    assert theta.get("theta0"), "Missing theta0"
    assert theta.get("theta1"), "Missing theta1"
    while True:
        try:
            mileage = int(input("Please enter a mileage > "))
        except ValueError:
            print("Mileage need to be an integer")
            continue
        estimation = estimate_price(mileage, theta["theta0"], theta["theta1"])
        print(f"Estimated price is {estimation:.3f}")


if __name__ == "__main__":
    main()
