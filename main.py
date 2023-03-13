import torch, torch.nn as nn
import snntorch as snn

def init_function():
    # Use a breakpoint in the code line below to debug your script.
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

def compare_models_robustness(*models: nn.Module, noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) -> None:
    from handle_model import handle_model
    from get_data import add_noise_to_mnist_dataset, load_mnist
    batch_size = 100
    train_loader, test_loader = load_mnist(batch_size)
    acc_list_per_noise_level = []
    # noise_levels = [0, 0.5, 1]

    # Network Architecture
    num_inputs = 28 * 28
    num_hidden = 1000
    num_outputs = 10
    # Temporal Dynamics
    num_steps = 25
    beta = 0.95

    for noise_level in noise_levels:
        print(f"Experiment with Noise Level: {noise_level}")
        noisy_train_dataset = add_noise_to_mnist_dataset(train_loader.dataset, noise_level=noise_level)
        noisy_train_loader = torch.utils.data.DataLoader(dataset=noisy_train_dataset, batch_size=batch_size, shuffle=False)

        noisy_test_dataset = add_noise_to_mnist_dataset(test_loader.dataset, noise_level=noise_level)
        noisy_test_loader = torch.utils.data.DataLoader(dataset=noisy_test_dataset, batch_size=batch_size, shuffle=False)

        model_handlers = []
        model_names = ["Noise Level"]
        for model in models:
            model_handlers.append(handle_model(num_steps, num_inputs, num_hidden, num_outputs, model, noisy_train_loader, noisy_test_loader))
            model_names.append(model.__name__)
            #models[0].__class__.__name__
        acc_list = []
        for model_runner in model_handlers:
            model_runner.run(epochs=10)
            acc_list.append(model_runner.test_acc)

        list_to_append = []
        list_to_append.append(noise_level)
        for acc in acc_list:
            list_to_append.append(acc)

        acc_list_per_noise_level.append(list_to_append)

    import pandas as pd
    df = pd.DataFrame(acc_list_per_noise_level)

    df.columns = model_names
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    figure, axes = plt.subplots()
    axes.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.grid()
    axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.ylabel("Accuracy in %")
    plt.xlabel("Noise Level")
    plt.title("Accuracy comparison of NN with MNIST")
        # df = pd.DataFrame(data=acc_list, columns=["Epoch", "Accuracy"])

    for index, row in df.iterrows():
        print(row)
    for i in range(df.shape[1]-1):
        sns.lineplot(data=df, x=df.iloc[:, 0], y=df.iloc[:, i+1], ax=axes, label=df.columns[i+1], marker="*",
                     markersize=8)

    plt.legend(loc='lower right')
        # axes.legend(labels=["Acc1", "Acc2"])
    plt.savefig("Compare.png")
    plt.close()

def snn_torch_four():
    from handle_model import handle_model
    from models import Net, SparseNet, HNN
    from get_data import load_mnist
    train_loader, test_loader = load_mnist(128)


    #net_model_runner = handle_model(num_steps, num_inputs, num_hidden, num_outputs, Net, train_loader, test_loader)
    #sparse_model_runner = handle_model(num_steps, num_inputs, num_hidden, num_outputs, SparseNet, train_loader, test_loader)
    compare_models_robustness(HNN, Net, SparseNet)

if __name__ == '__main__':
    # snn_test_one()
    snn_torch_four()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
