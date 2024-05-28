import torch
from src.data.data_loader import load_data, get_data_loader
from src.model.attention_lstm import AttentionLSTM
from src.train.train import train_model
from src.visualization.visualize import plot_training_loss
from src.config import config

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = load_data(config['file_path'], config['sequence_length'])
    # dataloader = get_data_loader(dataset, config['batch_size'])

    model = AttentionLSTM(config['input_size'], config['hidden_size'], config['output_size'], config['num_layers'])
    model.to(device)

    train_losses, val_losses = train_model(model, dataset, config['num_epochs'], config['learning_rate'], device)

    # 可视化训练损失
    plot_training_loss(train_losses, val_losses)

    # 保存模型权重
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Model weights saved to 'model_weights.pth'")

if __name__ == '__main__':
    main()
