import torch, torch.nn as nn
import snntorch as snn

class Net_old(nn.Module):
    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        num_inputs = 784  # number of inputs
        num_hidden = 300  # number of hidden neurons
        num_outputs = 10  # number of classes (i.e., output neurons)

        beta1 = 0.9  # global decay rate for all leaky neurons in layer 1
        beta2 = torch.rand((num_outputs),
                           dtype=torch.float)  # independent decay rate for each leaky neuron in layer 2: [0, 1)

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta1)  # not a learnable decay rate
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2, learn_beta=True)  # learnable decay rate

    def forward(self, x):
        mem1 = self.lif1.init_leaky()  # reset/init hidden states at t=0
        mem2 = self.lif2.init_leaky()  # reset/init hidden states at t=0
        spk2_rec = []  # record output spikes
        mem2_rec = []  # record output hidden states

        for step in range(self.num_steps):  # loop over time
            cur1 = self.fc1(x.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)  # record spikes
            mem2_rec.append(mem2)  # record membrane

        return torch.stack(spk2_rec), torch.stack(mem2_rec)


class Net_old(nn.Module):
    def __init__(self, num_steps, num_inputs, num_hidden, num_outputs, beta=0.95):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):

            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


class Net(nn.Module):

    def __init__(self, num_steps, num_inputs, num_hidden, num_outputs, beta=0.95):
        super().__init__()
        self.num_steps = num_steps

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

class SparseNet(nn.Module):

    def __init__(self, num_steps, num_inputs, num_hidden, num_outputs, beta=0.95):
        super().__init__()
        # Initialize layers
        self.num_steps = num_steps
        self.fc1 = SparseLayer(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = SparseLayer(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)


    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

from custom_layers import SparseLayer, NearestNeighborSparseLayer


class HNN(nn.Module):
    def __init__(self, num_steps, in_features, num_hidden, out_features, beta=0.95, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.num_steps = num_steps
        self.in_features = in_features
        self.out_features = out_features
        self.device = device


        beta1 = 0.9  # global decay rate for all leaky neurons in layer 1
        beta2 = torch.rand((num_hidden), dtype=torch.float)  # independent decay rate for each leaky neuron in layer 2: [0, 1)
        beta3 = torch.rand((num_hidden), dtype=torch.float)  # independent decay rate for each leaky neuron in layer 2: [0, 1)
        beta4 = torch.rand((out_features), dtype=torch.float)  # independent decay rate for each leaky neuron in layer 2: [0, 1)

        # Initialize layers

        self.ec = SparseLayer(in_features+out_features, num_hidden)
        self.ec_lif = snn.Leaky(beta=beta1)  # not a learnable decay rate

        self.dg = NearestNeighborSparseLayer(num_hidden, num_hidden, sparsity=0.33)
        self.dg_lif = snn.Leaky(beta=beta2, learn_beta=True)  # learnable decay rate

        self.ca3 = SparseLayer(num_hidden, num_hidden, sparsity=0.6)
        self.ca3_lif = snn.Leaky(beta=beta3, learn_beta=True)  # learnable decay rate

        self.ca1 = SparseLayer(num_hidden, out_features, sparsity=0.6)
        self.ca1_lif = snn.Leaky(beta=beta4, learn_beta=True)  # learnable decay rate

        self.forward_bool = True
        self.input_shape = ()


    def forward(self, x):
        mem1 = self.ec_lif.init_leaky()  # reset/init hidden states at t=0
        mem2 = self.dg_lif.init_leaky()  # reset/init hidden states at t=0
        mem3 = self.ca3_lif.init_leaky()  # reset/init hidden states at t=0
        mem4 = self.ca1_lif.init_leaky()  # reset/init hidden states at t=0
        spk4_rec = []  # record output spikes
        mem4_rec = []  # record output hidden states



        for step in range(self.num_steps):  # loop over time
            flatten_tensor = x.flatten(1)

            if self.forward_bool or self.input_shape != flatten_tensor.shape:
                # print("here")
                self.forward_bool = False
                self.input_shape = flatten_tensor.shape
                self.feedback = torch.zeros(flatten_tensor.shape[0], self.out_features).to(self.device)

            ec_input = torch.cat((flatten_tensor, self.feedback), dim=1)
            cur1 = self.ec(ec_input)
            spk1, mem1 = self.ec_lif(cur1, mem1)

            cur2 = self.dg(spk1)
            spk2, mem2 = self.dg_lif(cur2, mem2)

            cur3 = self.ca3(torch.add(spk1, spk2))
            spk3, mem3 = self.ca3_lif(cur3, mem3)

            cur4 = self.ca1(torch.add(spk1, spk3))
            spk4, mem4 = self.ca1_lif(cur4, mem4)

            self.feedback = spk4.clone().detach()

            spk4_rec.append(spk4)  # record spikes
            mem4_rec.append(mem4)  # record membrane

        return torch.stack(spk4_rec), torch.stack(mem4_rec)