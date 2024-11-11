import json
import sys


def estimate_price(mileage, theta0, theta1) -> float:
    return float(theta0 + (theta1 * mileage))


def main():
    assert len(sys.argv) == 2, "Need a mileage"
    try:
        with open('./theta.json', 'r') as f:
            theta = json.load(f)
    except Exception as e:
        print(f"Tried to open trained theta but got {e}")
        exit(1)
    assert theta.get("theta0"), "Missing theta0"
    assert theta.get("theta1"), "Missing theta1"
    try:
        mileage = int(sys.argv[1])
    except ValueError:
        print("Mileage need to be an integer")
        exit(1)
    estimation = estimate_price(mileage, theta["theta0"], theta["theta1"])
    print(f"Estimated price is {estimation}")


if __name__ == "__main__":
    main()
