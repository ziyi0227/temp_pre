import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CustomMinMaxScaler:
    def __init__(self, scalers_path='D:\\大学\\课程\\各课程\\数据科学导论\\期末\\temp_pre\\src\\utils\\params\\scale_params.npz'):
        self.scaler = MinMaxScaler()
        self.scalers_path = scalers_path  # 用于存储和加载参数的文件路径

    def fit(self, data):
        """
        根据给定的数据集fit scaler，并保存参数到文件。
        """
        self.scaler.fit(data)
        if self.scalers_path:
            np.savez(self.scalers_path,
                     data_min_=self.scaler.data_min_,
                     data_max_=self.scaler.data_max_,
                     scale_=self.scaler.scale_)

    def load_params(self, scalers_path='D:\\大学\\课程\\各课程\\数据科学导论\\期末\\temp_pre\\src\\utils\\params\\scale_params.npz'):
        """
        加载之前保存的归一化参数。
        """
        if scalers_path or self.scalers_path:
            scalers_path = scalers_path or self.scalers_path
            params = np.load(scalers_path, allow_pickle=True)  # 确保可以加载保存的参数
            self.scaler.data_min_ = params['data_min_']
            self.scaler.data_max_ = params['data_max_']
            self.scaler.scale_ = params['scale_']
            self.scaler.min_ = 0  # Assuming the default feature range (0, 1)
            self.scaler.data_range_ = self.scaler.data_max_ - self.scaler.data_min_

    def transform(self, data):
        """
        对数据进行归一化变换。
        """
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        """
        对归一化后的数据进行反变换，得到原始尺度的数据。
        """
        return self.scaler.inverse_transform(data)
