from .common import MLP
import torch.nn as nn
import torch
import numpy as np
import quadprog
# Auxiliary functions useful for GEM's inner optimization.
def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    def __init__(self, input_size, output_size, \
                 n_tasks, args):
        super(Net, self).__init__()
        self.net = MLP([input_size] + [100] * 2 + [output_size])
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = output_size
        self.opt = torch.optim.SGD(self.parameters(), args.lr)
        #self.opt = torch.optim.Adagrad(self.parameters(), args.lr)
        #self.opt = torch.optim.Adagrad(list(self.encoder.parameters())+list(self.mlp.parameters()))
        '''
        Below is the attributes defined in GEM implementation
        reference: https://arxiv.org/abs/1706.08840
        code from: https://github.com/facebookresearch/GradientEpisodicMemory
        '''
        self.margin = args.memory_strength
        self.n_memories = args.n_memories
        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            n_tasks, self.n_memories, input_size)
        #self.memory_labs = torch.LongTensor(n_tasks, self.n_memories, output_size)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        #self.memory_loss = np.zeros( (n_tasks, self.n_memories, output_size) )
        if torch.cuda.is_available():
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        # edit: need one more dimension for newly observed data
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks+1)
        if torch.cuda.is_available():
            self.grads = self.grads.cuda()
        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = np.zeros(n_tasks).astype(int)
        self.num_seen = np.zeros(n_tasks).astype(int)   # count the number of samples seen, for sampling

    def set_opt(self, lr):
        self.opt = torch.optim.SGD(self.parameters(), lr)
        #self.opt = torch.optim.Adam(self.parameters(), 1e-1)
        #self.opt = torch.optim.Adagrad(self.parameters(), lr)
    def forward(self, x, t):
        # xobs is the input to encoder
        # x is the input to mlp
        #print(x.shape)
        return self.net(x)

    def loss(self, pred, truth):
        return self.ce(pred, truth)

    def load_memory(self, data):
        # data: (tasks, xs, ys)
        # continuously load memory based on previous memory loading
        tasks, xs, ys = data
        #batch_size = 100  # remember 100 at a time
        for i in range(len(tasks)):
            if tasks[i] != self.old_task:
                # new task, clear mem_cnt
                self.observed_tasks.append(tasks[i])
                self.old_task = tasks[i]
                self.mem_cnt[tasks[i]] = 0
            x = torch.tensor(xs[i])
            y = torch.tensor(ys[i])
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            self.remember(x, tasks[i], y)
            #for k in range(len(xs[i]), batch_size):
            #    end_idx = min(k+batch_size, len(xs[i]))
            #    x = tensor.torch(xs[i][k:end_idx])
            #    y = tensor.torch(ys[i][k:end_idx])
            #    if torch.cuda.is_available():
            #        x = x.cuda()
            #        y = y.cuda()
            #    self.remember(x, tasks[i], y)

    def remember(self, x, t, y):
        # follow reservoir sampling
        # i-th item is remembered with probability min(B/i, 1)
        print('remembering...')
        for i in range(len(x)):
            self.num_seen[t] += 1
            prob_thre = min(self.n_memories, self.num_seen[t])
            rand_num = np.random.choice(self.num_seen[t], 1) # 0---self.num_seen[t]-1
            if rand_num < prob_thre:
                # keep the new item
                #print('remembering...')
                if self.mem_cnt[t] < self.n_memories:
                    self.memory_data[t, self.mem_cnt[t]].copy_(x.data[i])
                    self.memory_labs[t, self.mem_cnt[t]].copy_(y.data[i])
                    self.mem_cnt[t] += 1
                else:
                    # randomly choose one to rewrite
                    idx = np.random.choice(self.n_memories, size=1)
                    idx = idx[0]
                    self.memory_data[t, idx].copy_(x.data[i])
                    self.memory_labs[t, idx].copy_(y.data[i])

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
        t = 0
        self.grad_step = 1
        for _ in range(self.grad_step):
            if len(self.observed_tasks) >= 1:
                for tt in range(len(self.observed_tasks)):
                    if self.mem_cnt[tt] == 0 and tt == len(self.observed_tasks) - 1:
                        # nothing to train on
                        continue
                    self.zero_grad()
                    # fwd/bwd on the examples in the memory
                    past_task = self.observed_tasks[tt]
                    if tt == len(self.observed_tasks) - 1:
                        ptloss = self.loss(
                            self.forward(
                            self.memory_data[past_task][:self.mem_cnt[tt]],tt),   # TODO
                            self.memory_labs[past_task][:self.mem_cnt[tt]])   # up to current
                    else:
                        ptloss = self.loss(
                            self.forward(
                            self.memory_data[past_task][:self.mem_cnt[tt]],tt),   # TODO
                            self.memory_labs[past_task][:self.mem_cnt[tt]])
                    ptloss.backward()
                    store_grad(self.parameters, self.grads, self.grad_dims,
                               past_task)

            # now compute the grad on the current minibatch
            self.zero_grad()
            loss = self.loss(self.forward(x,t), y)
            loss.backward()

            # check if gradient violates constraints
            # treat gradient of current data as a new task (max of observed task + 1)
            # just to give it a new task label
            if len(self.observed_tasks) >= 1:
                # copy gradient
                new_t = max(self.observed_tasks)+1  # a new dimension
                store_grad(self.parameters, self.grads, self.grad_dims, new_t)
                indx = torch.cuda.LongTensor(self.observed_tasks) if torch.cuda.is_available() \
                    else torch.LongTensor(self.observed_tasks)   # here we need all observed tasks
                #indx = torch.cuda.FloatTensor(self.observed_tasks[:-1]) if torch.cuda.is_available() \
                #    else torch.FloatTensor(self.observed_tasks[:-1])
                # here is different, we are using new_t instead of t to ditinguish between
                # newly observed data and previous data
                dotp = torch.mm(self.grads[:, new_t].unsqueeze(0),
                                self.grads.index_select(1, indx))
                #print('dot product computed')
                if (dotp < 0).sum() != 0:
                    # remember norm
                    print('projecting...')
                    norm = torch.norm(self.grads[:, new_t], 2)
                    project2cone2(self.grads[:, new_t].unsqueeze(1),
                                  self.grads.index_select(1, indx), self.margin)
                    new_norm = torch.norm(self.grads[:, new_t], 2)
                    #print('norm: %f' % (norm.item()))
                    #print('new norm: %f' % (new_norm.item()))
                    self.grads[:, new_t].copy_(self.grads[:, new_t] / new_norm * norm)
                    # before overwrite, to avoid gradient explosion, renormalize the gradient
                    # so that new gradient has the same l2 norm as previous
                    # it can be proved theoretically that this does not violate the non-negativity
                    # of inner product between memory and gradient
                    # copy gradients back
                    overwrite_grad(self.parameters, self.grads[:, new_t],
                                   self.grad_dims)
            self.opt.step()
        # after training, store into memory

        # when storing into memory, we use the correct task label
        # Update ring buffer storing examples from current task
        if t != self.old_task:
            # new task, clear mem_cnt
            self.observed_tasks.append(t)
            self.old_task = t
            self.mem_cnt[t] = 0
        self.remember(x, t, y)
