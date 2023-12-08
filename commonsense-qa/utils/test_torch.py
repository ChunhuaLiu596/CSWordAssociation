import torch 
sl = 10
p_num, q_num, a_num = torch.tensor(8), torch.tensor(5) , torch.tensor(3) 
pmask = torch.arange(sl) >= p_num  # (nbz, sl)
qmask = torch.arange(sl) >= q_num  # (nbz, sl)
amask = torch.arange(sl) >= a_num  # (nbz, sl)

mask = qmask.unsqueeze(2) | amask.unsqueeze(1)  # (nbz, sl, sl)
print(mask)