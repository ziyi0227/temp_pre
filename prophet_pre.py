import pandas as pd
from prophet import Prophet
from src.config import config
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv(config['file_path'], index_col=False)  # 替换为你的CSV文件名

# 查看数据格式
print(data.head())

# 确保数据包含日期时间和温度列，例如 'date' 和 'temperature'
# 并将其重命名为 Prophet 所需的格式 'ds' 和 'y'
data.rename(columns={'datetime': 'ds', 'temperature': 'y'}, inplace=True)

# 检查和转换数据类型，确保 'ds' 列是日期时间格式
data['ds'] = pd.to_datetime(data['ds'])

# 初始化并拟合Prophet模型
model = Prophet()
model.fit(data)

# 创建未来一个月的日期数据框架
future = model.make_future_dataframe(periods=30)  # 预测未来30天
forecast = model.predict(future)

# 进行预测，并获取2024年6月18日下午5点的预测值
specific_date = pd.DataFrame({'ds': ['2024-06-18 17:00:00']})
specific_date['ds'] = pd.to_datetime(specific_date['ds'])
specific_forecast = model.predict(specific_date)

print("2024年6月18日下午5点的温度预测：")
print(specific_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# 绘制历史数据和预测数据
fig1 = model.plot(forecast)
plt.title('Temperature Forecast')
plt.savefig('temperature_forecast_prophet_fig1.png')

# 绘制成分图（趋势、周效应和年度效应）
fig2 = model.plot_components(forecast)

plt.savefig('temperature_forecast_prophet_fig2.png')
plt.show()

# 未来一个月的预测结果可视化
forecast_next_month = forecast[(forecast['ds'] >= '2024-06-01') & (forecast['ds'] <= '2024-06-30')]
plt.figure(figsize=(10, 6))
plt.plot(forecast_next_month['ds'], forecast_next_month['yhat'], label='Predicting temperature')
plt.fill_between(forecast_next_month['ds'], forecast_next_month['yhat_lower'], forecast_next_month['yhat_upper'], color='gray', alpha=0.2, label='不确定性区间')
plt.xlabel('date')
plt.ylabel('Predicting temperature')
plt.title('Temperature prediction for the next month')
plt.legend()
plt.savefig('temperature_forecast_next_month_prophet.png')
plt.show()

if __name__ == '__main__':
    pass
