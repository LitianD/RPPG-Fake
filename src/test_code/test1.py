import torch

# 包含字符串的 Python 列表
string_list = ['hello', 'world']

# 使用 torch.tensor 将字符串列表转换为 PyTorch 张量
tensor_from_strings = torch.tensor(string_list)

print(tensor_from_strings)