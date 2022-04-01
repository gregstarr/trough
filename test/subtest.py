from datetime import datetime
import trough

start_date = datetime(2020, 9, 8, 9)
end_date = datetime(2020, 9, 9, 12)
data = trough.get_data(start_date, end_date, 'north')
print(data['tec'].shape)
print(data['kp'].shape)
print(data['labels'].shape)
print(data['tec'].isnull().mean().item())
print(data['labels'].mean().item())
