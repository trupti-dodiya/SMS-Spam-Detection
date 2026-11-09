
import joblib

# Load our saved machine learning pipeline
model = joblib.load("spam_model.pkl")


# Function to predict an SMS message
def predict_sms(message):

    prediction = model.predict([message])[0]

    if prediction == "spam":
        print("Prediction: SPAM")
    else:
        print("Prediction: HAM (Genuine Message)")

    return prediction


# Take messages from the user
while True:

    message = input("\nEnter an SMS message (or type exit): ")

    if message.lower() == "exit":
        print("Program stopped.")
        break

    if message.strip() == "":
        print("Please enter a message.")
        continue

    predict_sms(message)
