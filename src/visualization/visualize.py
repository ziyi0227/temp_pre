import matplotlib.pyplot as plt


def plot_training_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
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
