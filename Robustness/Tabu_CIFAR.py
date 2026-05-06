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
from Networks import AF_Tools
from Networks.GetData import *
import torch.optim as optim
from Networks.AlexNet_CIFAR import AlexNet_CIFAR
from torchattacks import FGSM, PGD

Activation_Functions = AF_Tools.Activation_Functions_
operators = AF_Tools.operators_
operators_p = AF_Tools.operators_p_

EPOCHS = 20
LAYERS = 7
destination_path = '../save/cifar10/cifar10'
result_file_name = 'Tabu_CIFAR_FGSM.csv'
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
                print('Error occurred during training batch:', str(e))
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


class TabuSearch:
    def __init__(self, initial_state, data_loaders, dataset_sizes, atk_,
                 max_iterations=20, tabu_tenure=7, num_neighbors=5):
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
        self.tabu_list = {}

    def is_tabu(self, state, current_iteration):
        state_key = tuple(state)
        if state_key in self.tabu_list:
            if self.tabu_list[state_key] > current_iteration:
                return True
            else:
                del self.tabu_list[state_key]
        return False

    def add_to_tabu(self, state, current_iteration):
        state_key = tuple(state)
        self.tabu_list[state_key] = current_iteration + self.tabu_tenure

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

    def get_neighbors(self, state):
        neighbors = []
        for _ in range(self.num_neighbors):
            neighbor = self.perturb_state(state)
            neighbors.append(neighbor)
        return neighbors

    def energy(self, state):
        model = AlexNet_CIFAR(num_classes=10, states_=state).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        fname = destination_path + '_'.join(map(str, state)) + '.pth'

        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
        else:
            res = train_model(model, self.data_loaders, self.dataset_sizes, criterion, optimizer, self.device)
            if res < 0:
                return 100
            torch.save(model.state_dict(), fname)

        test_loss, test_acc = test_model(model, self.data_loaders, criterion, self.device, self.dataset_sizes)

        correct = 0
        total = 0
        model.eval()
        attack = atk_by_name(self.atk)
        a_t_k = attack(model, eps=self.eps)

        for indx, (images, labels) in enumerate(self.data_loaders['test']):
            perturb_images = a_t_k(images, labels).to(self.device)
            outputs = model(perturb_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(self.device)).sum()

        rob_acc = 100 * float(correct) / total

        with open(result_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([state, self.atk, self.eps, rob_acc, test_acc])

        print(state, self.atk, self.eps, test_acc, rob_acc)
        return (100 - rob_acc)

    def run(self):
        current_state = self.state
        current_energy = self.energy(current_state)
        self.best_state = current_state.copy()
        self.best_energy = current_energy
        self.convergence_curve.append(100 - self.best_energy)

        print(f"\nInitial energy: {current_energy:.4f} | Robustness: {100 - current_energy:.4f}%")

        for iteration in range(self.max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.max_iterations} ===")
            print(f"Tabu list size: {len(self.tabu_list)}")

            neighbors = self.get_neighbors(current_state)
            best_neighbor = None
            best_neighbor_energy = float('inf')

            for neighbor in neighbors:
                neighbor_energy = self.energy(neighbor)

                if self.is_tabu(neighbor, iteration):
                    if neighbor_energy < self.best_energy:
                        print(f"  Aspiration criterion triggered! Tabu state accepted.")
                        best_neighbor = neighbor
                        best_neighbor_energy = neighbor_energy
                else:
                    if neighbor_energy < best_neighbor_energy:
                        best_neighbor = neighbor
                        best_neighbor_energy = neighbor_energy

            if best_neighbor is None:
                print("  All neighbors tabu, picking least bad...")
                all_energies = [(self.energy(n), n) for n in neighbors]
                best_neighbor_energy, best_neighbor = min(all_energies, key=lambda x: x[0])

            self.add_to_tabu(current_state, iteration)
            current_state = best_neighbor
            current_energy = best_neighbor_energy

            if current_energy < self.best_energy:
                self.best_state = current_state.copy()
                self.best_energy = current_energy
                print(f"  New best found! Robustness: {100 - self.best_energy:.4f}%")

            self.convergence_curve.append(100 - self.best_energy)
            print(f"  Current robustness: {100 - current_energy:.4f}%")
            print(f"  Best robustness so far: {100 - self.best_energy:.4f}%")

        return self.best_state, self.best_energy

    def plot_convergence(self, atk_name):
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(self.convergence_curve)), self.convergence_curve,
                 marker='o', color='red', linewidth=2, markersize=4)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Robustness Accuracy (%)', fontsize=12)
        plt.title(f'Tabu Search Convergence Curve — {atk_name}', fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_path = f'../save/convergence_curve_tabu_cifar_{atk_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence curve saved to: {save_path}")
        plt.show()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    data_loaders_, dataset_sizes_ = get_cifar10_data_32(512, 256)
    atk_eps_s = [("FGSM", 1 / 255)]

    with open(result_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['State', 'Attack', 'Epsilon', 'Robust Accuracy', 'Test Accuracy'])

    for atk_eps_ in atk_eps_s:
        ch_size_1 = LAYERS * 3
        state_ = np.random.choice(len(Activation_Functions), ch_size_1, replace=True)

        ts = TabuSearch(
            initial_state=state_,
            data_loaders=data_loaders_,
            dataset_sizes=dataset_sizes_,
            atk_=atk_eps_,
            max_iterations=20,
            tabu_tenure=7,
            num_neighbors=5
        )

        state_, e = ts.run()
        print("\nFinal state:", state_)
        print("Final energy:", e)
        print("Best robustness: {:.4f}%".format(100 - e))

        ts.plot_convergence(atk_eps_[0])