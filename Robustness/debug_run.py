import numpy as np
import torch
from simanneal import Annealer
import random
import torch.nn as nn
import time
import os
import os.path
import csv
import torch.optim as optim

from Networks import AF_Tools
from Networks.GetData import *
from Networks.ResNet18_MNIST import resnet18_mnist
from Utils.Tools import write_data_in_file
from torchattacks import FGSM, PGD

Activation_Functions = AF_Tools.Activation_Functions_
operators = AF_Tools.operators_
operators_p = AF_Tools.operators_p_

EPOCHS = 0
LAYERS = 4
ch_size_1 = LAYERS * 3
trace = False

# -----------------------------
# experiment info
# -----------------------------
MODEL_NAME = "ResNet"      # or "AlexNet"
DATASET_NAME = "MNIST"     # or "CIFAR"
ALGORITHM_NAME = "SA"      # or "base", "VNS", etc.

destination_path = f"save/{MODEL_NAME.lower()}_{DATASET_NAME.lower()}/{MODEL_NAME.lower()}_{DATASET_NAME.lower()}_"

def make_result_filename(model_name, dataset_name, attack_name, algorithm_name):
    return f"results_{model_name}_{dataset_name}_{attack_name}_{algorithm_name}.csv"

def append_result_to_csv(filename, row):
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)

        # write header only once
        if not file_exists:
            writer.writerow([
                "model",
                "dataset",
                "attack",
                "algorithm",
                "eps",
                "state",
                "clean_test_acc",
                "robust_acc"
            ])

        writer.writerow(row)


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
                print('**************************** error ****************************')
                print(str(e))
                break

        if error_found:
            return -1

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() * 100 / total_checked
        print('correct   :  ', running_corrects, ' dataset size  :   ', dataset_sizes['train'])
        print('Epoch {}/{} - Train Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, EPOCHS, epoch_loss, epoch_acc))
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
    return {
        'FGSM': FGSM,
        'PGD': PGD,
    }


def atk_by_name(atkname):
    attacks = get_attacks()
    return attacks[atkname]


class robust_simannealer(Annealer):
    def __init__(self, state, data_loaders, dataset_sizes, atk_):
        self.dataloaders, self.dataset_sizes, (self.atk, self.eps) = data_loaders, dataset_sizes, atk_
        self.p1 = range(0, LAYERS)
        self.p2 = range(LAYERS, LAYERS * 2)
        self.p3 = range(LAYERS * 2, LAYERS * 3)
        super(robust_simannealer, self).__init__(state)

    def move(self):
        moves_count = 2
        for i in range(moves_count):
            loc = random.randint(0, len(self.state) - 1)
            if loc in self.p1:
                self.state[loc] = random.choices(operators_p, weights=operators_p)[0]
                if self.state[loc] < 2:
                    self.state[loc + LAYERS * 2] = 0

            elif loc in self.p2:
                self.state[loc] = random.randint(0, len(Activation_Functions) - 1)

            elif loc in self.p3:
                self.state[loc] = random.randint(0, len(Activation_Functions) - 1)
                if self.state[loc - LAYERS * 2] < 2:
                    self.state[loc - LAYERS * 2] = random.randint(2, 4)

    def energy(self):
        total_time = time.time()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if trace:
            print('Individual : ', self.state)

        model = resnet18_mnist(num_classes=10, states_=self.state).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        fname = destination_path + '_'.join(map(str, self.state)) + '.pth'

        train_time = time.time()
        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
        else:
            res = train_model(model, self.dataloaders, self.dataset_sizes, criterion, optimizer, device)
            if res < 0:
                return 100
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            torch.save(model.state_dict(), fname)
        train_time = time.time() - train_time

        test_time = time.time()
        test_loss, test_acc = test_model(model, self.dataloaders, criterion, device, self.dataset_sizes)
        test_time = time.time() - test_time

        correct = 0
        total = 0

        eval_time = time.time()
        model.eval()

        attack = atk_by_name(self.atk)
        a_t_k = attack(model, eps=self.eps)

        for indx, (images, labels) in enumerate(self.dataloaders['test']):
            images = images.to(device)
            labels = labels.to(device)

            perturb_images = a_t_k(images, labels)
            outputs = model(perturb_images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        eval_time = time.time() - eval_time
        total_time = time.time() - total_time

        rob_acc = 100 * float(correct) / total

        result_file_name = make_result_filename(
            MODEL_NAME,
            DATASET_NAME,
            self.atk,
            ALGORITHM_NAME
        )

        append_result_to_csv(result_file_name, [
            MODEL_NAME,
            DATASET_NAME,
            self.atk,
            ALGORITHM_NAME,
            self.eps,
            list(map(int, self.state)),
            test_acc,
            rob_acc
        ])

        print(self.state, self.atk, self.eps, test_acc, rob_acc)
        print('total time : ', total_time, ' train time : ', train_time, ' eval time : ', eval_time, ' test time : ', test_time)

        return 100 * (1 - rob_acc)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    data_loaders_, dataset_sizes_ = get_mnist_data(500)

    atk_eps_s = [("FGSM", 1/255)]

    for atk_eps_ in atk_eps_s:
        state_ = np.random.choice(len(Activation_Functions), ch_size_1, replace=True)
        tsp = robust_simannealer(state_, data_loaders_, dataset_sizes_, atk_eps_)
        tsp.steps = 1
        tsp.copy_strategy = "deepcopy"
        state_, e = tsp.anneal()

        print("Final state:", state_)
        print("Final energy:", e)