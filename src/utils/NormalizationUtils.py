

class NormalizationUtils:
    @staticmethod
    def min_max_scaling(X, feature_range=(0, 1)):
        """
        类中的静态方法实现最小最大缩放。
        """
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        scaled_X = (X - min_val) / (max_val - min_val)  # 确保没有除以零的错误
        scaled_X = scaled_X * (feature_range[1] - feature_range[0]) + feature_range[0]
        return scaled_X
        pass