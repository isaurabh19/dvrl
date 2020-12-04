import torch

tensor1 = torch.rand(10)
tensor2 = torch.rand(10)
output = [tensor1, tensor2]

fin = torch.cat(output, dim=0)
print(fin)
sorted_fin, indices = torch.sort(fin, descending=True)
print(sorted_fin)
mean_fin = torch.mean(sorted_fin)
print(mean_fin, torch.std(sorted_fin))
print(sorted_fin[sorted_fin > mean_fin])
