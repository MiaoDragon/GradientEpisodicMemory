from .common import MLP
import torch.nn as nn
import torch
import numpy as np

# Auxiliary functions useful for GEM's inner optimization.
class Net(nn.Module):
    def __init__(self, input_size, output_size, \
                 n_tasks, args):
        super(MinstNet, self).__init__()
        self.net = MLP([input_size] + [100] * 2 + [output_size])
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = output_size
        self.opt = torch.optim.SGD(self.parameters(), args.lr)
        #self.opt = torch.optim.Adagrad(list(self.encoder.parameters())+list(self.mlp.parameters()))
        '''
        Below is the attributes defined in GEM implementation
        reference: https://arxiv.org/abs/1706.08840
        code from: https://github.com/facebookresearch/GradientEpisodicMemory
        '''

    def set_opt(self):
        self.opt = torch.optim.SGD(self.parameters(), args.lr)
        #self.opt = torch.optim.Adam(self.parameters(), 1e-1)
        #self.opt = torch.optim.Adagrad(self.parameters(), 1e-1)
    def forward(self, x):
        # xobs is the input to encoder
        # x is the input to mlp
        #print(x.shape)
        return self.net(x)

    def loss(self, pred, truth):
        return self.ce(pred, truth)

    def remember(self, x, t, y):
        # put new data (x,y) into memory t
        # if memory is not full, then go as usual
        # otherwise:
        # firstly compute the loss of past states and new state
        # delete the lowest loss one
        # will update self.mem_cnt after inserting memory
        #print('memory size: %d' %(self.mem_cnt))
        bsz = y.data.size(0)
        if bsz+self.mem_cnt <= self.n_memories:
            eff_bsz = bsz
            endcnt = self.mem_cnt + bsz
            effbsz = endcnt - self.mem_cnt
            self.memory_data[t, self.mem_cnt: endcnt].copy_(
                x.data[: effbsz])
            if bsz == 1:
                self.memory_labs[t, self.mem_cnt].copy_(y.data[0])
            else:
                self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                    y.data[: effbsz])
            self.mem_cnt += effbsz
        else:
            #print('prioritzing')
            # use network to get the loss
            data = []
            data = list(self.memory_data[t, :self.mem_cnt])
            data += list(x)
            labels = list(self.memory_labs[t, :self.mem_cnt])
            labels += list(y)
            data = torch.stack(data)
            labels = torch.stack(labels)
            preds = self.forward(data)
            losses = [self.loss(preds[i].unsqueeze(0), labels[i].unsqueeze(0)) for i in range(len(labels))]
            _, indices = torch.topk(torch.stack(losses), self.n_memories)
            self.memory_data[t].copy_(data[indices])
            self.memory_labs[t].copy_(labels[indices])
            self.mem_cnt = self.n_memories

    '''
    Below is the added GEM feature
    reference: https://arxiv.org/abs/1706.08840
    code from: https://github.com/facebookresearch/GradientEpisodicMemory
    '''
    def observe(self, x, t, y):
        # scenario 1:
        # t is a new task (no previous tasks or have previous tasks)
        # scenario 2:
        # t is an old task
        ########################
        # update memory
        # everytime we treat the new data as a new task
        # compute gradient on all tasks
        # (prevent forgetting previous experience of same task, too)
        # now compute the grad on the current minibatch
        self.zero_grad()
        loss = self.loss(self.forward(x), y)
        loss.backward()

        # check if gradient violates constraints
        # treat gradient of current data as a new task (max of observed task + 1)
        # just to give it a new task label            #print('yes is projecting!')
        self.opt.step()
        # when storing into memory, we use the correct task label
        # Update ring buffer storing examples from current task
