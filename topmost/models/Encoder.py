# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#         #x = 200,10000
#         #times = 200
#     #output = 200,hidden_dim

# # class MLPEncoder(nn.Module):
# #     def __init__(self, vocab_size, num_topic, hidden_dim, dropout):
# #         super().__init__()

# #         self.fc11 = nn.Linear(vocab_size, hidden_dim)
# #         self.fc12 = nn.Linear(hidden_dim, hidden_dim)
# #         self.fc21 = nn.Linear(hidden_dim, num_topic)
# #         self.fc22 = nn.Linear(hidden_dim, num_topic)

# #         self.fc1_drop = nn.Dropout(dropout)
# #         self.z_drop = nn.Dropout(dropout)

# #         self.mean_bn = nn.BatchNorm1d(num_topic, affine=True)
# #         self.mean_bn.weight.requires_grad = False
# #         self.logvar_bn = nn.BatchNorm1d(num_topic, affine=True)
# #         self.logvar_bn.weight.requires_grad = False

# #     def reparameterize(self, mu, logvar):
# #         if self.training:
# #             std = torch.exp(0.5 * logvar)
# #             eps = torch.randn_like(std)
# #             return mu + (eps * std)
# #         else:
# #             return mu

# #     def forward(self, x):
# #         e1 = F.softplus(self.fc11(x))
# #         e1 = F.softplus(self.fc12(e1))
# #         e1 = self.fc1_drop(e1)
# #         mu = self.mean_bn(self.fc21(e1))
# #         logvar = self.logvar_bn(self.fc22(e1))
# #         theta = self.reparameterize(mu, logvar)
# #         theta = F.softmax(theta, dim=1)
# #         theta = self.z_drop(theta)
# #         return theta, mu, logvar
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MLPEncoder(nn.Module):
#     def __init__(self, vocab_size, num_topic, hidden_dim, dropout, num_experts):
#         super().__init__()
        
#         self.num_experts = num_experts
#         self.experts = nn.ModuleList([SingleMLPEncoder(vocab_size, num_topic, hidden_dim, dropout) for _ in range(num_experts)])
#         self.gate = nn.Linear(vocab_size, num_experts)  # This is the gating mechanism
#         self.std_dev = 1.0  # Standard deviation of the Gaussian gate function
    
#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return mu + (eps * std)
#         else:
#             return mu
    
#     def forward(self, x, time_index):
#         batch_size = x.shape[0]
        
#         # Ensure time_index is a column vector of size (batch_size, 1)
#         time_indices = time_index.unsqueeze(1).expand(-1, self.num_experts)  # (batch_size, num_experts)
#         expert_indices = torch.arange(self.num_experts, device=x.device).expand(batch_size, -1)  # (batch_size, num_experts)
        
#         # Compute the Gaussian distribution for expert assignment
#         dist = -((expert_indices - time_indices) ** 2) / (2 * self.std_dev ** 2)  # Gaussian distribution over time
        
#         # Apply softmax to get the gate scores
#         gate_scores = F.softmax(dist, dim=-1)  # (batch_size, num_experts)
        
#         mu_list, logvar_list, theta_list = [], [], []
        
#         for i, expert in enumerate(self.experts):
#             # Get the topic distribution, mean, and logvar from each expert
#             theta, mu, logvar = expert(x)
            
#             # Apply the gate scores to weight the outputs from each expert
#             mu_list.append(mu.unsqueeze(1) * gate_scores[:, i].unsqueeze(-1))  # (batch_size, 1, num_topic)
#             logvar_list.append(logvar.unsqueeze(1) * gate_scores[:, i].unsqueeze(-1))  # (batch_size, 1, num_topic)
#             theta_list.append(theta.unsqueeze(1) * gate_scores[:, i].unsqueeze(-1))  # (batch_size, 1, num_topic)
        
#         # Combine the results from all experts, weighted by the gate scores
#         mu = torch.sum(torch.cat(mu_list, dim=1), dim=1)  # (batch_size, num_topic)
#         logvar = torch.sum(torch.cat(logvar_list, dim=1), dim=1)  # (batch_size, num_topic)
#         theta = torch.sum(torch.cat(theta_list, dim=1), dim=1)  # (batch_size, num_topic)
        
#         return theta, mu, logvar

# class SingleMLPEncoder(nn.Module):
#     def __init__(self, vocab_size, num_topic, hidden_dim, dropout):
#         super().__init__()
#         self.fc11 = nn.Linear(vocab_size, hidden_dim)
#         self.fc12 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc21 = nn.Linear(hidden_dim, num_topic)
#         self.fc22 = nn.Linear(hidden_dim, num_topic)
        
#         self.fc1_drop = nn.Dropout(dropout)
#         self.z_drop = nn.Dropout(dropout)
        
#         self.mean_bn = nn.BatchNorm1d(num_topic, affine=True)
#         self.mean_bn.weight.requires_grad = False
#         self.logvar_bn = nn.BatchNorm1d(num_topic, affine=True)
#         self.logvar_bn.weight.requires_grad = False
    
#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return mu + (eps * std)
#         else:
#             return mu
    
#     def forward(self, x):
#         e1 = F.softplus(self.fc11(x))
#         e1 = F.softplus(self.fc12(e1))
#         e1 = self.fc1_drop(e1)
#         mu = self.mean_bn(self.fc21(e1))
#         logvar = self.logvar_bn(self.fc22(e1))
#         theta = self.reparameterize(mu, logvar)
#         theta = F.softmax(theta, dim=1)
#         theta = self.z_drop(theta)
#         return theta, mu, logvar

# vocab_size = 10000
# num_topics = 50 
# en_units = 100 
# dropout = 0.
# encoder = MLPEncoder(vocab_size, num_topics, en_units, dropout,num_experts=200)
# x = torch.rand(200,10000)
# times = torch.tensor([i+1 for i in range(200)])
# theta,mu,logvar = encoder(x,times)
# # print(theta.shape)
# # print(mu.shape)
# # print(logvar.shape)
# import torch
# import torch.nn.functional as F

# # Initialize MLPEncoder
# vocab_size = 10000
# num_topics = 50 
# en_units = 100 
# dropout = 0.
# encoder = MLPEncoder(vocab_size, num_topics, en_units, dropout, num_experts=200)

# # Generate some random input data
# x = torch.rand(200, 10000)

# # Create time indices (for example, we'll choose timestamp 5)
# times = torch.tensor([5] * 200)  # All batch entries at time 5

# # Get theta, mu, logvar from the encoder
# theta, mu, logvar = encoder(x, times)

# # Access the gate scores (softmaxed distances for each expert at time 5)
# # gate_scores are computed inside the MLPEncoder's forward method
# batch_size = x.shape[0]
# time_indices = times.unsqueeze(1).expand(-1, encoder.num_experts)  # (batch_size, num_experts)
# expert_indices = torch.arange(encoder.num_experts, device=x.device).expand(batch_size, -1)  # (batch_size, num_experts)

# # Compute the Gaussian distribution for the gate
# dist = -((expert_indices - time_indices) ** 2) / (2 * encoder.std_dev ** 2)  # (batch_size, num_experts)

# # Apply softmax to get the gate scores (probabilities)
# gate_scores = F.softmax(dist, dim=-1)  # (batch_size, num_experts)

# # At timestamp 5, we can access the gate probabilities
# gate_probabilities_at_5 = gate_scores[0, :]  # Taking the first example, as all times are 5 for this batch

# print("Gate probabilities for timestamp 5 across experts:")
# print(gate_probabilities_at_5)
