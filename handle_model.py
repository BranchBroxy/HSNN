import snntorch as snn
import torch
import torch.nn as nn

class handle_model():
    def __init__(self, num_steps, num_inputs, num_hidden, num_outputs, model, train_dataloader, test_dataloader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.model = model

        self.num_steps = num_steps
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs


        self.training_acc = []
        self.training_acc_with_epoch = []
        self.training_loss = []
        self.name_of_model = model.__name__
        if train_dataloader.batch_size != test_dataloader.batch_size:
            raise Exception("Train Dataloader batch size and Test Dataloader batch size dont match")
        else:
            self.batch_size = train_dataloader.batch_size

        self.dtype = torch.float

    def run(self, epochs=1, learning_rate=5e-4, loss_fn=nn.CrossEntropyLoss()):
        self.model = self.model(self.num_steps, self.num_inputs, self.num_hidden, self.num_outputs).to(self.device)
        self.epochs = epochs
        self.learinig_rate = learning_rate
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        # clear previously stored gradients
        self.optimizer.zero_grad()
        # weight update
        self.optimizer.step()

        # training loop
        self.loss_hist = []
        self.test_loss_hist = []
        self.counter = 0

        for self.epoch in range(self.epochs):
            print(f"Epoch {self.epoch + 1} of {self.name_of_model}\n-------------------------------")
            self.train()
            self.test()
            self.complet_test()


    def train(self):
        self.iter_counter = 0
        train_batch = iter(self.train_dataloader)
        n_total_steps = len(self.train_dataloader)
        # Minibatch training loop
        for batch, (data, targets) in enumerate(train_batch):
            data = data.to(self.device)
            targets = targets.to(self.device)
            self.data = data
            self.targets = targets

            # forward pass
            self.model.train()
            spk_rec, mem_rec = self.model(data.view(self.batch_size, -1))


            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=self.dtype, device=self.device)
            for step in range(self.num_steps):
                loss_val += self.loss_fn(mem_rec[step], targets)

            # Gradient calculation + weight update
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            # Store loss history for future plotting
            self.loss_hist.append(loss_val.item())
            if (batch + 1) % 100 == 0:
                print(f'Epoch [{self.epoch + 1}/{ self.epochs}], Step [{batch + 1}/{n_total_steps}], Loss: {loss_val.item():.4f}')
                #print(self.model.sl1.weight)
                #print(model.sl2.weight)
                #print(model.sl2.connections)

    def test(self):
        # Test set
        with torch.no_grad():
            self.model.eval()
            test_data, test_targets = next(iter(self.test_dataloader))
            test_data = test_data.to(self.device)
            test_targets = test_targets.to(self.device)

            # Test set forward pass
            test_spk, test_mem = self.model(test_data.view(self.batch_size, -1))
            import numpy as np
            _, idx = test_spk.sum(dim=0).max(1)
            self.acc = np.mean((test_targets == idx).detach().cpu().numpy())

            # Test set loss
            test_loss = torch.zeros((1), dtype=self.dtype, device=self.device)
            for step in range(self.num_steps):
                test_loss += self.loss_fn(test_mem[step], test_targets)
            self.test_loss_hist.append(test_loss.item())
            print(f"Test Error: \nAccuracy: {(100 * self.acc):>0.1f}%, Avg loss: {test_loss.item():>8f} \n")
            self.training_acc.append(100 * self.acc)
            self.training_acc_with_epoch.append([self.epoch + 1, 100 * self.acc])



    def plot_loss(self):
        # Plot Loss
        import matplotlib.pyplot as plt
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        plt.plot(self.loss_hist)
        plt.plot(self.test_loss_hist)
        plt.title("Loss Curves")
        plt.legend(["Train Loss", "Test Loss"])
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()


    def batch_accuracy(self, data, targets, train=False):
        import numpy as np
        output, _ = self.model(data.view(self.batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        self.acc = np.mean((targets == idx).detach().cpu().numpy())

        if train:
            print(f"Train set accuracy for a single minibatch: {self.acc * 100:.2f}%")
        else:
            print(f"Test set accuracy for a single minibatch: {self.acc * 100:.2f}%")



    def complet_test(self):
        total = 0
        correct = 0

        # drop_last switched to False to keep all samples
        test_loader = self.test_dataloader

        with torch.no_grad():
            self.model.eval()
            for data, targets in test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                # forward pass
                test_spk, _ = self.model(data.view(data.size(0), -1))

                # calculate total accuracy
                _, predicted = test_spk.sum(dim=0).max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f"Total correctly classified test set images: {correct}/{total}")
        self.test_acc = 100 * correct / total
        print(f"Test Set Accuracy: {self.test_acc:.2f}%")
