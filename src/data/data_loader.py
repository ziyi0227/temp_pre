import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from src.utils.min_max_scaler import CustomMinMaxScaler


class WeatherDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx + self.sequence_length].values
        y = self.data.iloc[idx + self.sequence_length]['temperature']
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_data(file_path, sequence_length):
    # 指定分隔符为分号，并处理双引号包裹的数据
    data = pd.read_csv(file_path, encoding='utf-8', index_col=False)
    # 删除temperature为空的行
    data = data.dropna(subset=['temperature'])

    # 尝试解析日期时间
    date_formats = ['%d.%m.%Y %H:%M', '%d.%m.%Y']
    for date_format in date_formats:
        try:
            data['datetime'] = pd.to_datetime(data['datetime'], format=date_format, errors='coerce')
            if data['datetime'].notna().all():
                break
        except ValueError as e:
            continue

    if data['datetime'].isna().any():
        # 输出异常数据
        print(data[data['datetime'].isna()])
        raise ValueError("日期时间格式不匹配，请检查CSV文件中的日期时间字段。")

    data.set_index('datetime', inplace=True)

    # 选择需要的特征
    features = ['temperature', 'pressure', 'humidity', 'wind_speed', 'Td']

    # 填充风速缺失值
    data['wind_speed'].fillna(0, inplace=True)

    # 将风向编码为数值
    label_encoder = LabelEncoder()
    data['wind_direction'] = label_encoder.fit_transform(data['DD'])
    features.append('wind_direction')

    # 只保留所需的特征列
    data = data[features]

    # 在选择需要的特征之后
    scaler = CustomMinMaxScaler()
    scaler.fit(data[features])
    data[features] = scaler.transform(data[features])

    # 构建数据集
    dataset = WeatherDataset(data, sequence_length)
    return dataset


def get_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
