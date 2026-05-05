import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import random
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from Networks import AF_Tools
from Networks.GetData import *
import torch.optim as optim
from Utils.Tools import write_data_in_file
from Networks.ResNet18_MNIST import resnet18_mnist
from torchattacks import FGSM, PGD


Activation_Functions = AF_Tools.Activation_Functions_
operators = AF_Tools.operators_
operators_p = AF_Tools.operators_p_

EPOCHS = 10
LAYERS = 4
destination_path = '/workspace/SARAF/save/res18_mnist/res18_mnist_'
result_file_name = 'results_ResNet_MNIST_TS_base.csv'
trace = False


def train_model(model, data_loaders, dataset_sizes, criterion, optimizer, device):
    model = model.to(device)
    model.train()

    for epoch in range(EPOCHS):
        epoch_time = time.time()
        running_loss, running_corrects, total_checked = 0.0, 0, 0
        error_found = False

        for indx, (inputs, labels) in enumerate(data_loaders['train']):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss_total = loss.item() * inputs.size(0)
                running_loss += batch_loss_total
                running_corrects += torch.sum(preds == labels.data)
                total_checked += inputs.size(0)
            except Exception as e:
                error_found = True
                print(' **************************** error ****************************')
                print(str(e))
                break

        if error_found:
            return -1
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() * 100 / total_checked
        print('correct   :  ', running_corrects, '         dataset size  :   ', dataset_sizes['train'])
        print('Epoch {}/{} -  Train Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, EPOCHS, epoch_loss, epoch_acc))
        print('time : ', time.time() - epoch_time)

    return 0


def test_model(model, data_loaders, criterion, device, dataset_sizes):
    model.eval()
    running_loss, running_corrects, total_checked = 0.0, 0, 0

    for indx, (inputs, labels) in enumerate(data_loaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_checked += inputs.size(0)

    test_loss = running_loss / dataset_sizes['test']
    test_acc = running_corrects.double() * 100 / total_checked

    return test_loss, test_acc.item()



def get_attacks():
    return {'FGSM': FGSM, 'PGD': PGD}


def atk_by_name(atkname):
    attacks = get_attacks()
    return attacks[atkname]



def extract_features(model, data_loader, device, num_batches=20):
    features = []
    labels_list = []

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hook = model.avgpool.register_forward_hook(get_activation('avgpool'))
    model.eval()
    with torch.no_grad():
        for indx, (images, labels) in enumerate(data_loader):
            if indx >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
            feat = activation['avgpool'].squeeze(-1).squeeze(-1)
            features.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())

    hook.remove()
    return np.concatenate(features, axis=0), np.concatenate(labels_list, axis=0)


def extract_features_perturbed(model, data_loader, device, attack, num_batches=20):
    features = []
    labels_list = []

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hook = model.avgpool.register_forward_hook(get_activation('avgpool'))
    model.eval()
    for indx, (images, labels) in enumerate(data_loader):
        if indx >= num_batches:
            break
        perturbed = attack(images, labels).to(device)
        with torch.no_grad():
            _ = model(perturbed)
        feat = activation['avgpool'].squeeze(-1).squeeze(-1)
        features.append(feat.cpu().numpy())
        labels_list.append(labels.numpy())

    hook.remove()
    return np.concatenate(features, axis=0), np.concatenate(labels_list, axis=0)


def plot_tsne(features, labels, title, ax):
    print(f"Running t-SNE for: {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(features)
    scatter = ax.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=labels, cmap='tab10', alpha=0.6, s=10
    )
    ax.set_title(title, fontsize=12)
    return scatter


def generate_tsne_comparison(original_model, ts_best_model, data_loaders, device, atk_name, eps):
    attack_class = atk_by_name(atk_name)
    atk_original = attack_class(original_model, eps=eps)
    atk_ts = attack_class(ts_best_model, eps=eps)

    print("\n=== Generating t-SNE Visualization ===")

    print("Extracting features: Original model (clean)...")
    feat_original, labels = extract_features(original_model, data_loaders['test'], device)

    print("Extracting features: Original model (perturbed)...")
    feat_perturbed, _ = extract_features_perturbed(original_model, data_loaders['test'], device, atk_original)

    print("Extracting features: Tabu Search optimized model (perturbed)...")
    feat_ts, _ = extract_features_perturbed(ts_best_model, data_loaders['test'], device, atk_ts)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    s1 = plot_tsne(feat_original, labels, "(a) Original ResNet-18", axes[0])
    s2 = plot_tsne(feat_perturbed, labels, "(b) Perturbed ResNet-18", axes[1])
    s3 = plot_tsne(feat_ts, labels, "(c) ResNet-18 + Tabu Search", axes[2])

    axes[2].legend(*s3.legend_elements(), title="Classes", loc="upper right", markerscale=2)
    plt.suptitle(f"t-SNE Visualization — Attack: {atk_name}, ε={eps:.4f}", fontsize=14)
    plt.tight_layout()

    save_path = f'../save/tsne_ts_{atk_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"t-SNE plot saved to: {save_path}")
    plt.show()


def evaluate_robustness_at_eps(model, data_loader, device, attack_class, eps):
    model.eval()
    atk = attack_class(model, eps=eps)
    correct = 0
    total = 0

    for indx, (images, labels) in enumerate(data_loader):
        perturbed = atk(images, labels).to(device)
        with torch.no_grad():
            outputs = model(perturbed)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

    return 100.0 * correct / total


def generate_perturbation_comparison(original_model, ts_best_model, data_loaders, device):
    print("\n=== Generating Perturbation Comparison Plot ===")

    eps_values = [i / 255 for i in range(1, 11)]
    eps_labels = list(range(1, 11))
    attack_names = ['FGSM', 'PGD']

    results = {
        'default': {'FGSM': [], 'PGD': []},
        'tabu':    {'FGSM': [], 'PGD': []}
    }

    for atk_name in attack_names:
        attack_class = atk_by_name(atk_name)
        print(f"\nEvaluating {atk_name}...")

        for eps in eps_values:
            print(f"  epsilon = {eps:.4f}")

            acc_default = evaluate_robustness_at_eps(
                original_model, data_loaders['test'], device, attack_class, eps
            )
            acc_tabu = evaluate_robustness_at_eps(
                ts_best_model, data_loaders['test'], device, attack_class, eps
            )

            results['default'][atk_name].append(acc_default)
            results['tabu'][atk_name].append(acc_tabu)
            print(f"    Default: {acc_default:.2f}%  |  Tabu Search: {acc_tabu:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, atk_name in enumerate(attack_names):
        ax = axes[i]
        ax.plot(eps_labels, results['default'][atk_name],
                marker='s', color='blue', linewidth=1.5, markersize=6,
                label='Default (ReLU)', linestyle='-')
        ax.plot(eps_labels, results['tabu'][atk_name],
                marker='o', color='red', linewidth=1.5, markersize=6,
                label='Tabu Search', linestyle='-')
        ax.set_title(atk_name, fontsize=13)
        ax.set_xlabel('Perturbation', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_xticks(eps_labels)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='upper center', ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("Comparison of ResNet-18 accuracy: Default (ReLU) vs Tabu Search",
                 fontsize=13, y=1.06)
    plt.tight_layout()

    save_path = '../save/perturbation_comparison_ts.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPerturbation comparison plot saved to: {save_path}")
    plt.show()


class TabuSearch:
    def __init__(self, initial_state, data_loaders, dataset_sizes, atk_,
                 max_iterations=20, tabu_tenure=7, num_neighbors=5):
        """
        max_iterations : number of main loop iterations
        tabu_tenure    : how many iterations a state stays in the tabu list
        num_neighbors  : how many neighbors to generate and evaluate each iteration
        """
        self.state = initial_state
        self.data_loaders = data_loaders
        self.dataset_sizes = dataset_sizes
        self.atk, self.eps = atk_
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.num_neighbors = num_neighbors
        self.best_state = initial_state
        self.best_energy = float('inf')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.convergence_curve = []

        # Tabu list stores (state_tuple, expiry_iteration)
        # A state is tabu until expiry_iteration is reached
        self.tabu_list = {}

    def is_tabu(self, state, current_iteration):
        """Check if a state is currently in the tabu list"""
        state_key = tuple(state)
        if state_key in self.tabu_list:
            # Still tabu if expiry hasn't been reached
            if self.tabu_list[state_key] > current_iteration:
                return True
            else:
                # Expired — remove from tabu list
                del self.tabu_list[state_key]
        return False

    def add_to_tabu(self, state, current_iteration):
        """Add a state to the tabu list with an expiry iteration"""
        state_key = tuple(state)
        self.tabu_list[state_key] = current_iteration + self.tabu_tenure

    def perturb_state(self, state):
        """Generate a neighboring state by randomly changing one element"""
        new_state = state.copy()
        loc = random.randint(0, len(state) - 1)
        if loc in range(0, LAYERS):
            new_state[loc] = random.choices(operators_p, weights=operators_p)[0]
            if new_state[loc] < 2:
                new_state[loc + LAYERS * 2] = 0
        elif loc in range(LAYERS, LAYERS * 2):
            new_state[loc] = random.randint(0, len(Activation_Functions) - 1)
        elif loc in range(LAYERS * 2, LAYERS * 3):
            new_state[loc] = random.randint(0, len(Activation_Functions) - 1)
            if new_state[loc - LAYERS * 2] < 2:
                new_state[loc - LAYERS * 2] = random.randint(2, 4)
        return new_state

    def get_neighbors(self, state):
        """Generate num_neighbors different neighboring states"""
        neighbors = []
        for _ in range(self.num_neighbors):
            neighbor = self.perturb_state(state)
            neighbors.append(neighbor)
        return neighbors

    def energy(self, state):
        """Train and evaluate a model for a given state. Returns energy (lower = better)."""
        total_time = time.time()
        model = resnet18_mnist(num_classes=10, states_=state).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        fname = destination_path + '_'.join(map(str, state)) + '.pth'

        train_time = time.time()
        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
        else:
            res = train_model(model, self.data_loaders, self.dataset_sizes, criterion, optimizer, self.device)
            if res < 0:
                return 100
            torch.save(model.state_dict(), fname)
        train_time = time.time() - train_time

        test_time = time.time()
        test_loss, test_acc = test_model(model, self.data_loaders, criterion, self.device, self.dataset_sizes)
        test_time = time.time() - test_time

        correct = 0
        total = 0
        eval_time = time.time()
        model.eval()
        attack = atk_by_name(self.atk)
        a_t_k = attack(model, eps=self.eps)

        for indx, (images, labels) in enumerate(self.data_loaders['test']):
            perturb_images = a_t_k(images, labels).to(self.device)
            outputs = model(perturb_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(self.device)).sum()

        eval_time = time.time() - eval_time
        total_time = time.time() - total_time
        rob_acc = 100 * float(correct) / total

        write_data_in_file(result_file_name, (state, self.atk, self.eps, rob_acc, test_acc))
        print(state, self.atk, self.eps, test_acc, rob_acc)
        print('total time:', total_time, ' train time:', train_time,
              ' eval time:', eval_time, 'test time:', test_time)

        return (100 - rob_acc)

    def run(self):
        """
        Main Tabu Search loop:
        1. Generate num_neighbors neighbors of current state
        2. Evaluate all non-tabu neighbors
        3. Pick the best non-tabu neighbor (aspiration: allow tabu if it beats best ever)
        4. Add current state to tabu list
        5. Move to best neighbor
        6. Repeat
        """
        current_state = self.state
        current_energy = self.energy(current_state)
        self.best_state = current_state.copy()
        self.best_energy = current_energy
        self.convergence_curve.append(100 - self.best_energy)

        print(f"\nInitial energy: {current_energy:.4f} | Robustness: {100 - current_energy:.4f}%")

        for iteration in range(self.max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.max_iterations} ===")
            print(f"Tabu list size: {len(self.tabu_list)}")

            # Generate neighbors
            neighbors = self.get_neighbors(current_state)

            best_neighbor = None
            best_neighbor_energy = float('inf')

            for neighbor in neighbors:
                neighbor_energy = self.energy(neighbor)

                # Aspiration criterion: allow tabu state if it beats the best known solution
                if self.is_tabu(neighbor, iteration):
                    if neighbor_energy < self.best_energy:
                        print(f"  Aspiration criterion triggered! Tabu state accepted.")
                        best_neighbor = neighbor
                        best_neighbor_energy = neighbor_energy
                else:
                    # Non-tabu: accept if it's the best non-tabu neighbor so far
                    if neighbor_energy < best_neighbor_energy:
                        best_neighbor = neighbor
                        best_neighbor_energy = neighbor_energy

            if best_neighbor is None:
                # All neighbors are tabu and none beat best — pick least bad tabu neighbor
                print("  All neighbors tabu, picking least bad...")
                all_energies = [(self.energy(n), n) for n in neighbors]
                best_neighbor_energy, best_neighbor = min(all_energies, key=lambda x: x[0])

            # Add current state to tabu list before moving
            self.add_to_tabu(current_state, iteration)

            # Move to best neighbor
            current_state = best_neighbor
            current_energy = best_neighbor_energy

            # Update best solution
            if current_energy < self.best_energy:
                self.best_state = current_state.copy()
                self.best_energy = current_energy
                print(f"  New best found! Robustness: {100 - self.best_energy:.4f}%")

            self.convergence_curve.append(100 - self.best_energy)
            print(f"  Current robustness: {100 - current_energy:.4f}%")
            print(f"  Best robustness so far: {100 - self.best_energy:.4f}%")

        return self.best_state, self.best_energy

    def plot_convergence(self, atk_name):
        """Plot convergence curve: best robustness accuracy vs iteration"""
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(self.convergence_curve)), self.convergence_curve,
                 marker='o', color='red', linewidth=2, markersize=4)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Robustness Accuracy (%)', fontsize=12)
        plt.title(f'Tabu Search Convergence Curve — {atk_name}', fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_path = f'../save/convergence_curve_ts_{atk_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence curve saved to: {save_path}")
        plt.show()



if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    data_loaders_, dataset_sizes_ = get_mnist_data(10000)
    atk_eps_s = [("FGSM", 1/255)]

    for atk_eps_ in atk_eps_s:
        ch_size_1 = LAYERS * 3
        state_ = np.random.choice(len(Activation_Functions), ch_size_1, replace=True)

        ts = TabuSearch(
            initial_state=state_,
            data_loaders=data_loaders_,
            dataset_sizes=dataset_sizes_,
            atk_=atk_eps_,
            max_iterations=20,     # number of main iterations
            tabu_tenure=7,         # how long a state stays tabu
            num_neighbors=5       # neighbors evaluated per iteration
        )

        state_, e = ts.run()
        print()
        print("Final state: ", state_)
        print("Final energy:", e)
        print("Best robustness: {:.4f}%".format(100 - e))


        ts.plot_convergence(atk_eps_[0])

     
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Baseline model: operator index 3 = value 1 (plain f(x)), activation 0 = ReLU
        default_state = [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]
        original_model = resnet18_mnist(num_classes=10, states_=default_state).to(device)

        # Best Tabu Search model with saved weights
        ts_best_model = resnet18_mnist(num_classes=10, states_=state_).to(device)
        fname_best = destination_path + '_'.join(map(str, state_)) + '.pth'
        if os.path.isfile(fname_best):
            ts_best_model.load_state_dict(torch.load(fname_best))
        else:
            print("Warning: best model weights not found, using untrained model")


        generate_tsne_comparison(
            original_model,
            ts_best_model,
            data_loaders_,
            device,
            atk_eps_[0],
            atk_eps_[1]
        )


        generate_perturbation_comparison(
            original_model,
            ts_best_model,
            data_loaders_,
            device
        )