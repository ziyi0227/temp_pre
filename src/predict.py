import torch
import numpy as np
from datetime import datetime, timedelta
from src.data.data_loader import load_data
from src.model.attention_lstm import AttentionLSTM
from src.visualization.visualize import plot_prediction
from src.config import config
from src.utils.min_max_scaler import CustomMinMaxScaler


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = load_data(config['file_path'], config['sequence_length'])

    model = AttentionLSTM(config['input_size'], config['hidden_size'], config['output_size'], config['num_layers'])
    model.to(device)

    # 加载模型权重
    model.load_state_dict(torch.load('alstm_weights.pth', map_location=device))
    model.eval()

    # 预测2024年6月18日下午5点的温度
    prediction_date = datetime(2024, 6, 18, 17, 0)
    input_data = dataset.data.iloc[-config['sequence_length']:].values  # 取最后一段数据作为输入
    input_data_tensor = torch.tensor(input_data, dtype=torch.float32).to(device).unsqueeze(0)

    with torch.no_grad():
        predicted_temp = model(input_data_tensor).item()

    scaler = CustomMinMaxScaler()
    scaler.load_params()
    input_data = np.array([[predicted_temp] * 6])
    predicted_temp_original_scale = scaler.inverse_transform(input_data)[0][0]

    print(f"Predicted temperature at {prediction_date} is {predicted_temp_original_scale:.2f}°C")

    # 可视化预测结果
    dates = [prediction_date - timedelta(hours=i) for i in range(config['sequence_length'])]
    dates.reverse()
    actual_temps = dataset.data['temperature'][-config['sequence_length']:]
    predicted_temps = [predicted_temp] * config['sequence_length']

    plot_prediction(dates, actual_temps, predicted_temps)


if __name__ == '__main__':
    main()
