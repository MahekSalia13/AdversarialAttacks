import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import random
import torch.nn as nn
import time
import csv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from Networks import AF_Tools
from Networks.GetData import *
import torch.optim as optim
from Networks.AlexNet_CIFAR import *
from torchattacks import FGSM, PGD


Activation_Functions = AF_Tools.Activation_Functions_
operators = AF_Tools.operators_
operators_p = AF_Tools.operators_p_

EPOCHS = 20
LAYERS = 4
destination_path = '../save/cifar10/cifar10_'
result_file_name = 'ILS_CIFAR_FGSM.csv'
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
        print('Epoch {}/{} - Train Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, EPOCHS, epoch_loss, epoch_acc))
        print('Time taken for epoch:', time.time() - epoch_time)

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


def generate_perturbation_comparison(original_model, ils_best_model, data_loaders, device):
    print("\n=== Generating Perturbation Comparison Plot ===")

    eps_values = [i / 255 for i in range(1, 11)]
    eps_labels = list(range(1, 11))
    attack_names = ['FGSM', 'PGD']

    results = {
        'default': {'FGSM': [], 'PGD': []},
        'ils':     {'FGSM': [], 'PGD': []}
    }

    for atk_name in attack_names:
        attack_class = atk_by_name(atk_name)
        print(f"\nEvaluating {atk_name}...")

        for eps in eps_values:
            print(f"  epsilon = {eps:.4f}")

            acc_default = evaluate_robustness_at_eps(
                original_model, data_loaders['test'], device, attack_class, eps
            )
            acc_ils = evaluate_robustness_at_eps(
                ils_best_model, data_loaders['test'], device, attack_class, eps
            )

            results['default'][atk_name].append(acc_default)
            results['ils'][atk_name].append(acc_ils)
            print(f"    Default: {acc_default:.2f}%  |  ILS: {acc_ils:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, atk_name in enumerate(attack_names):
        ax = axes[i]
        ax.plot(eps_labels, results['default'][atk_name],
                marker='s', color='blue', linewidth=1.5, markersize=6,
                label='Default (ReLU)', linestyle='-')
        ax.plot(eps_labels, results['ils'][atk_name],
                marker='o', color='red', linewidth=1.5, markersize=6,
                label='ILS', linestyle='-')
        ax.set_title(atk_name, fontsize=13)
        ax.set_xlabel('Perturbation', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_xticks(eps_labels)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='upper center', ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("Comparison of AlexNet accuracy on CIFAR-10: Default (ReLU) vs ILS",
                 fontsize=13, y=1.06)
    plt.tight_layout()

    save_path = '../save/perturbation_comparison_ils_cifar.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPerturbation comparison plot saved to: {save_path}")
    plt.show()


class IteratedLocalSearch:
    def __init__(self, initial_state, data_loaders, dataset_sizes, atk_, max_iterations=50):
        self.state = initial_state
        self.data_loaders = data_loaders
        self.dataset_sizes = dataset_sizes
        self.atk, self.eps = atk_
        self.max_iterations = max_iterations
        self.best_state = initial_state
        self.best_energy = float('inf')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.convergence_curve = []

    def local_search(self, state):
        best_state = state
        best_energy = self.energy(state)

        for _ in range(5):
            new_state = self.perturb_state(state)
            new_energy = self.energy(new_state)
            if new_energy < best_energy:
                best_state, best_energy = new_state, new_energy

        return best_state, best_energy

    def perturb_state(self, state):
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

    def energy(self, state):
        total_time = time.time()
        model = AlexNet_CIFAR(num_classes=10, states_=state).to(self.device)
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

        with open(result_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([state, self.atk, self.eps, rob_acc, test_acc])

        print(state, self.atk, self.eps, test_acc, rob_acc)
        print('total time:', total_time, ' train time:', train_time,
              ' eval time:', eval_time, 'test time:', test_time)

        return 100 - rob_acc

    def run(self):
        current_state = self.state
        current_energy = self.energy(current_state)
        self.best_state, self.best_energy = current_state, current_energy
        self.convergence_curve.append(100 - self.best_energy)

        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration+1}/{self.max_iterations}")
            new_state = self.perturb_state(current_state)
            new_state, new_energy = self.local_search(new_state)

            if new_energy < self.best_energy:
                self.best_state, self.best_energy = new_state, new_energy

            current_state, current_energy = new_state, new_energy
            self.convergence_curve.append(100 - self.best_energy)
            print(f"Best robustness so far: {100 - self.best_energy:.4f}%")

        return self.best_state, self.best_energy

    def plot_convergence(self, atk_name):
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(self.convergence_curve)), self.convergence_curve,
                 marker='o', color='blue', linewidth=2, markersize=4)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Robustness Accuracy (%)', fontsize=12)
        plt.title(f'ILS Convergence Curve (CIFAR-10) — {atk_name}', fontsize=13)
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_path = f'../save/convergence_curve_ils_cifar_{atk_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence curve saved to: {save_path}")
        plt.show()

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    data_loaders_, dataset_sizes_ = get_cifar10_data_32(512, 256)
    atk_eps_s = [("FGSM", 1/255)]

    # Initialize CSV
    with open(result_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['State', 'Attack', 'Epsilon', 'Robust Accuracy', 'Test Accuracy'])

    for atk_eps_ in atk_eps_s:
        ch_size_1 = LAYERS * 3
        state_ = np.random.choice(len(Activation_Functions), ch_size_1, replace=True)
        ils = IteratedLocalSearch(state_, data_loaders_, dataset_sizes_, atk_eps_, max_iterations=50)
        state_, e = ils.run()
        print()
        print("Final state: ", state_)
        print("Best robustness: {:.4f}%".format(100 - e))

        
        ils.plot_convergence(atk_eps_[0])

     
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Baseline model: untrained default ReLU AlexNet
        default_state = [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]
        original_model = AlexNet_CIFAR(num_classes=10, states_=default_state).to(device)

        # Train baseline if not already saved
        fname_default = destination_path + '_'.join(map(str, default_state)) + '.pth'
        if os.path.isfile(fname_default):
            original_model.load_state_dict(torch.load(fname_default))
        else:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(original_model.parameters(), lr=0.001)
            train_model(original_model, data_loaders_, dataset_sizes_, criterion, optimizer, device)
            torch.save(original_model.state_dict(), fname_default)

        # Best ILS model with saved weights
        ils_best_model = AlexNet_CIFAR(num_classes=10, states_=state_).to(device)
        fname_best = destination_path + '_'.join(map(str, state_)) + '.pth'
        if os.path.isfile(fname_best):
            ils_best_model.load_state_dict(torch.load(fname_best))
        else:
            print("Warning: best model weights not found, using untrained model")

           
        generate_perturbation_comparison(
            original_model,
            ils_best_model,
            data_loaders_,
            device
        )