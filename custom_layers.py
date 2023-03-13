import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLayer(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5, bias=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(SparseLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.use_bias = bias
        self.device = device

        # Berechnen der maximalen Anzahl an Verbindungen zwischen den Eingangs- und Ausgangsneuronen.
        max_connections = int(in_features * out_features * sparsity)

        # Erstellen eines 2D-Tensors, der alle möglichen Verbindungen zwischen den Eingangs- und Ausgangsneuronen enthält.
        connections = torch.zeros(in_features, out_features)

        # Zufälliges Auswählen von max_connections Verbindungen und Markieren als aktiv.
        active_indices = torch.randperm(in_features * out_features)[:max_connections]
        connections.view(-1)[active_indices] = 1

        # Speichern der Verbindungen als PyTorch Parameter.
        self.connections = nn.Parameter(connections, requires_grad=False)

        # Erstellen der Gewichtungsmatrix als PyTorch Parameter.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device), requires_grad=True)

        # Initialisierung der Gewichte mit Xavier-Initialisierung.
        nn.init.xavier_uniform_(self.weight)

        # Optional: Erstellen des Bias-Parameters.
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).to(device), requires_grad=True)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Anwenden der Verbindungen auf die Eingabe.
        x = x.matmul(self.connections * self.weight.t())

        # Optional: Hinzufügen des Biases.
        if self.use_bias:
            x = x + self.bias

        return x


class NearestNeighborSparseLayer(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5, bias=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(NearestNeighborSparseLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.use_bias = bias
        self.device = device

        # Berechnen der maximalen Anzahl an Verbindungen zwischen den Eingangs- und Ausgangsneuronen.
        max_connections = int(in_features * out_features * sparsity)

        # Erstellen eines 2D-Tensors, der alle möglichen Verbindungen zwischen den Eingangs- und Ausgangsneuronen enthält.
        connections = torch.zeros(in_features, out_features)

        # Zufälliges Auswählen von max_connections Verbindungen und Markieren als aktiv.
        active_indices = torch.randperm(in_features * out_features)[:max_connections]
        connections.view(-1)[active_indices] = 1

        # Erstellen der Verbindungen zwischen benachbarten Input-Neuronen.
        nearest_neighbors = torch.zeros(in_features, in_features)

        for i in range(in_features):
            for j in range(i - 1, i + 2):
                if j >= 0 and j < in_features:
                    nearest_neighbors[i, j] = 1

        # Speichern der Verbindungen als PyTorch Parameter.
        self.connections = nn.Parameter(connections, requires_grad=True)

        # Speichern der Verbindungen zwischen benachbarten Input-Neuronen als PyTorch Parameter.
        self.nearest_neighbors = nn.Parameter(nearest_neighbors, requires_grad=True)

        # Erstellen der Gewichtungsmatrix als PyTorch Parameter.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device), requires_grad=True)

        # Initialisierung der Gewichte mit Xavier-Initialisierung.
        nn.init.xavier_uniform_(self.weight)

        # Optional: Erstellen des Bias-Parameters.
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).to(device), requires_grad=True)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Anwenden der Verbindungen auf die Eingabe.
        connections = self.connections * self.nearest_neighbors
        x = x.matmul(connections * self.weight.t())

        # Optional: Hinzufügen des Biases.
        if self.use_bias:
            x = x + self.bias

        return x


class SelfConnectedSparseLayer(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5, self_connection=0.5, bidirectional=True, bias=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(SelfConnectedSparseLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.self_connection = self_connection
        self.bidirectional = bidirectional
        self.use_bias = bias
        self.device = device

        # Berechnen der maximalen Anzahl an Verbindungen zwischen den Eingangs- und Ausgangsneuronen.
        max_connections = int(in_features * out_features * sparsity)

        # Erstellen eines 2D-Tensors, der alle möglichen Verbindungen zwischen den Eingangs- und Ausgangsneuronen enthält.
        connections = torch.zeros(in_features, out_features)

        # Zufälliges Auswählen von max_connections Verbindungen und Markieren als aktiv.
        active_indices = torch.randperm(in_features * out_features)[:max_connections]
        connections.view(-1)[active_indices] = 1

        # Erstellen der Verbindungen zwischen benachbarten Input-Neuronen.
        nearest_neighbors = torch.zeros(in_features, in_features)

        for i in range(in_features):
            for j in range(i - 1, i + 2):
                if j >= 0 and j < in_features:
                    nearest_neighbors[i, j] = 1

        # Selbstverbindungen
        self_connections = torch.zeros(in_features, in_features)
        num_self_connections = int(in_features * self_connection)

        for i in range(in_features):
            self_indices = torch.randperm(in_features)[:num_self_connections]
            self_connections[i, self_indices] = 1

        # Speichern der Verbindungen als PyTorch Parameter.
        self.connections = nn.Parameter(connections, requires_grad=True)

        # Speichern der Verbindungen zwischen benachbarten Input-Neuronen als PyTorch Parameter.
        self.nearest_neighbors = nn.Parameter(nearest_neighbors, requires_grad=True)

        # Speichern der Selbstverbindungen als PyTorch Parameter.
        self.self_connections = nn.Parameter(self_connections, requires_grad=True)

        # Erstellen der Gewichtungsmatrix als PyTorch Parameter.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device), requires_grad=True)

        # Initialisierung der Gewichte mit Xavier-Initialisierung.
        nn.init.xavier_uniform_(self.weight)

        # Optional: Erstellen des Bias-Parameters.
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).to(device), requires_grad=True)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Anwenden der Verbindungen auf die Eingabe.
        connections = self.connections * self.nearest_neighbors

        if self.bidirectional:
            connections = connections + self.connections.t() * self.self_connections

        x = x.matmul(connections * self.weight.t())

        # Optional: Hinzufügen des Biases.
        if self.use_bias:
            x = x + self.bias

        return x


class SparseLayerV2(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5, bias=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(SparseLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.use_bias = bias
        self.device = device

        # Erstellen der Verbindungsmatrix als Tensor.
        self.connections = nn.Parameter(torch.zeros(in_features, out_features), requires_grad=True)

        # Erstellen der Gewichtungsmatrix als PyTorch-Parameter.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device), requires_grad=True)

        # Initialisierung der Gewichte mit Xavier-Initialisierung.
        nn.init.xavier_uniform_(self.weight)

        # Optional: Erstellen des Bias-Parameters.
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).to(device), requires_grad=True)
            nn.init.zeros_(self.bias)

        # Berechnen der maximalen Anzahl an Verbindungen zwischen den Eingangs- und Ausgangsneuronen.
        self.max_connections = int(in_features * out_features * sparsity)

    def forward(self, x):
        # Aktualisieren der Verbindungsmatrix.
        active_indices = torch.topk(self.connections.view(-1), self.max_connections).indices
        self.connections.data = torch.zeros_like(self.connections)
        self.connections.view(-1)[active_indices] = 1

        # Anwenden der Verbindungen auf die Eingabe.
        x = x.matmul(self.connections * self.weight.t())

        # Optional: Hinzufügen des Biases.
        if self.use_bias:
            x = x + self.bias

        return x