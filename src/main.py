import torch
from datetime import datetime, timedelta
from src.data.data_loader import load_data, get_data_loader
from src.model.attention_lstm import AttentionLSTM
from src.train.train import train_model
from src.visualization.visualize import plot_training_loss, plot_prediction
from src.config import config


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = load_data(config['file_path'], config['sequence_length'])
    dataloader = get_data_loader(dataset, config['batch_size'])

    model = AttentionLSTM(config['input_size'], config['hidden_size'], config['output_size'], config['num_layers'])
    model.to(device)

    losses = train_model(model, dataloader, config['num_epochs'], config['learning_rate'], device)

    # 可视化训练损失
    plot_training_loss(losses)

    # 预测2024年6月18日下午5点的温度
    prediction_date = datetime(2024, 6, 18, 17, 0)
    input_data = dataset.data.iloc[-config['sequence_length']:].values  # 取最后一段数据作为输入
    input_data_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)  # 将数据移动到GPU

    model.eval()
    with torch.no_grad():
        predicted_temp = model(input_data_tensor.unsqueeze(0)).item()

    print(f"Predicted temperature at {prediction_date} is {predicted_temp:.2f}°C")

    # 可视化预测结果
    dates = [prediction_date - timedelta(hours=i) for i in range(config['sequence_length'])]
    dates.reverse()
    actual_temps = dataset.data['temperature'][-config['sequence_length']:]
    predicted_temps = [predicted_temp] * config['sequence_length']

    plot_prediction(dates, actual_temps, predicted_temps)


if __name__ == '__main__':
    main()
