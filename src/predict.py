import argparse
import joblib
import pandas as pd

from .config import MODEL_PATH, FEATURE_COLUMNS

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    sample = pd.DataFrame([{
        "SepalLengthCm": sepal_length,
        "SepalWidthCm": sepal_width,
        "PetalLengthCm": petal_length,
        "PetalWidthCm": petal_width,
    }])

    prediction = model.predict(sample)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        probabilities = dict(zip(bundle["classes"], probabilities))

    return prediction, probabilities

def main():
    parser = argparse.ArgumentParser(description="Predict Iris species.")
    parser.add_argument("--sepal-length", type=float, required=True)
    parser.add_argument("--sepal-width", type=float, required=True)
    parser.add_argument("--petal-length", type=float, required=True)
    parser.add_argument("--petal-width", type=float, required=True)
    args = parser.parse_args()

    prediction, probabilities = predict_species(
        args.sepal_length, args.sepal_width,
        args.petal_length, args.petal_width
    )

    print(f"Predicted species: {prediction}")
    if probabilities:
        print("Class probabilities:")
        for species, probability in probabilities.items():
            print(f"  {species}: {probability:.2%}")

if __name__ == "__main__":
    main()
