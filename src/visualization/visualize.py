import matplotlib.pyplot as plt


def plot_training_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_and_validation_loss.png')
    plt.show()


def plot_prediction(date, actual, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(date, actual, label='Actual Temperature')
    plt.plot(date, predicted, label='Predicted Temperature')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig('temperature_prediction.png')
    plt.show()
