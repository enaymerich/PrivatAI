def acknowledge_workers(users):
    """Making workers know each other"""
    for user in users:
        for worker in users:
            if worker != user:
                user.add_worker(worker)

def create_workers(usernames=['bob','alice']):
    """Initializing users"""
    users = list()
    for i, user in enumerate(usernames):
        users.append(sy.VirtualWorker(hook, id=user))
    return users


import sys

import torch as th
import syft as sy
import torch.optim as optim
from torch import nn
import numpy as np
hook = sy.TorchHook(th)

usernames = ['bob', 'alice', 'john', 'pippo']
## Creating users
users = create_workers(usernames)

inputs = list()
targets = list()
models = list()
optimizers = list()
##This is our model, it can be arbitrary complex
model = nn.Linear(2, 1)
designated_worker = sy.VirtualWorker(hook, id='secure worker')
acknowledge_workers(users +[designated_worker])
for i, user in enumerate(users):
    ##Distributing input data to users: each user has slightly different data
    data = th.tensor([[1., 1], [0, 1.], [1, 0], [0, 0]] + np.random.randn(4, 2), dtype=th.float32)
    print('User data ', i, '\n', data)
    inputs.append(data.send(user))
    targets.append(th.tensor([[1.], [0], [1], [0]]).send(user))
    models.append(model.copy().send(user))
    optimizers.append(optim.SGD(params=models[i].parameters(), lr=0.001))

    ##Letting user know each other

for major_epoch in range(20):
    print('Iteration: ' + str(major_epoch+1))
    ##Sharing new model to the users
    for i, user in enumerate(users):
        models[i] = model.copy().send(user)
        optimizers[i] = optim.SGD(params=models[i].parameters(), lr=0.001)
        ##Training the model on each user
    for epoch in range(10):
        for i, user in enumerate(users):
            user_target = targets[i]
            user_input = inputs[i]
            user_optimizer = optimizers[i]
            user_model = models[i]

            user_optimizer.zero_grad()
            pred = user_model.forward(user_input)
            loss = ((pred-user_target)**2).sum()
            loss.backward()
            user_optimizer.step()
            sys.stdout.write(user.id + ': ' + str(loss.get()) + '\t')
            if i == len(users)-1:
                sys.stdout.write('\n')
                sys.stdout.flush()
    weight_sum = 0
    bias_sum = 0
    ##Sending to trusted aggregator
    for i, user in enumerate(users):
        models[i].move(designated_worker)
        weight_sum += models[i].weight.data
        bias_sum += models[i].bias.data
    model.weight.data = (weight_sum.get())/len(users)
    model.bias.data = (bias_sum.get())/len(users)
