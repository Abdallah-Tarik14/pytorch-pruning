import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time

# Function to calculate FLOPs (Floating Point Operations)
def count_flops(model, input_size=(1, 3, 32, 32)):
    # This is a simplified FLOPs counter and might not be perfectly accurate for all layers.
    # For more accurate FLOPs, consider using libraries like `thop` or `ptflops`.
    
    flops = 0
    # Create a dummy input tensor
    input_tensor = torch.randn(input_size)

    def conv_hook(module, input, output):
        # For convolutional layers, FLOPs = 2 * Cin * Cout * K_h * K_w * H_out * W_out
        # (2 for multiply-add operation)
        batch_size, cin, hin, win = input[0].size()
        cout, k_h, k_w = module.out_channels, module.kernel_size[0], module.kernel_size[1]
        h_out, w_out = output.size(2), output.size(3)
        
        # Consider bias if present
        bias_flops = cout * h_out * w_out if module.bias is not None else 0

        flops += 2 * cin * cout * k_h * k_w * h_out * w_out + bias_flops

    def linear_hook(module, input, output):
        # For linear layers, FLOPs = 2 * in_features * out_features
        flops += 2 * module.in_features * module.out_features
        # Consider bias if present
        bias_flops = module.out_features if module.bias is not None else 0
        flops += bias_flops

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model(input_tensor)

    for hook in hooks:
        hook.remove()

    return flops

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        # Adjust the first convolutional layer for 3-channel 32x32 input (CIFAR-10)
        # VGG16 expects 224x224, so we need to adapt or train from scratch.
        # For simplicity, we'll keep the pretrained weights and let the model adapt.
        # The original VGG16 classifier is for ImageNet (1000 classes).
        # We need to replace the classifier for CIFAR-10 (10 classes).

        # Freeze feature layers initially
        for param in self.features.parameters():
            param.requires_grad = False

        # Calculate the input features for the classifier based on CIFAR-10 input size
        # A 32x32 image going through VGG16 features will result in a 1x1x512 output
        # if the last pooling layer is adaptive average pooling to 1x1.
        # Original VGG16 uses MaxPool2d with kernel_size=2, stride=2.
        # For 32x32 input, after 5 blocks of conv+pool, the size becomes 32/2^5 = 1.
        # So, the input to the classifier will be 512 * 1 * 1 = 512.
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()
    
    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data


        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if args.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune             

class PrunningFineTuner_VGG16:
    def __init__(self, model):
        self.train_data_loader = dataset.loader()
        self.test_data_loader = dataset.test_loader()

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model) 
        self.model.train()

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            if args.use_cuda:
                batch = batch.cuda()
            output = self.model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
        
        accuracy = float(correct) / total
        print("Accuracy :", accuracy)
        
        self.model.train()
        return accuracy

    def train(self, optimizer = None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.0001, momentum=0.9)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")
        

    def train_batch(self, optimizer, batch, label, rank_filters):

        if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer = None, rank_filters = False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)
        
    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        #Get the accuracy before prunning
        self.test()
        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        # Adjust pruning strategy for CIFAR-10. Prune a smaller percentage per iteration.
        # Let\'s try pruning 10% of filters in each iteration until 50% of filters are pruned.
        pruning_percentage_per_iteration = 0.05 # Prune 5% of total filters per iteration
        total_pruning_percentage = 0.5 # Prune up to 50% of total filters

        num_filters_to_prune_per_iteration = int(number_of_filters * pruning_percentage_per_iteration)
        iterations = int(total_pruning_percentage / pruning_percentage_per_iteration)

        print(f"Number of pruning iterations: {iterations} (pruning {pruning_percentage_per_iteration*100}% of filters per iteration)")

        for i in range(iterations):
            print(f"Pruning iteration {i+1}/{iterations}")
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

            print("Layers that will be prunned", layers_prunned)
            print("Pruning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=args.use_cuda)

            self.model = model
            if args.use_cuda:
                self.model = self.model.cuda()

            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from pruning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epoches = 5) # Reduced epochs for faster iteration


        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=10) # Final fine-tuning
        torch.save(model.state_dict(), "model_prunned")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--use-cuda", action="store_true", default=False, help="Use NVIDIA GPU acceleration")    
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args

if __name__ == '__main__':
    args = get_args()

    if args.train:
        model = ModifiedVGG16Model()
        if args.use_cuda:
            model = model.cuda()
        fine_tuner = PrunningFineTuner_VGG16(model)
        fine_tuner.train(epoches=20) # Increased initial training epochs
        torch.save(model, "model")

    elif args.prune:
        original_model = torch.load("model", map_location=lambda storage, loc: storage)
        if args.use_cuda:
            original_model = original_model.cuda()
        
        # Evaluate original model
        print("\n--- Original Model Metrics ---")
        original_fine_tuner = PrunningFineTuner_VGG16(original_model)
        original_accuracy = original_fine_tuner.test()
        original_params = count_parameters(original_model)
        original_flops = count_flops(original_model)
        print(f"Accuracy: {original_accuracy:.4f}")
        print(f"Parameters: {original_params}")
        print(f"FLOPs: {original_flops}")

        # Prune the model
        model = torch.load("model", map_location=lambda storage, loc: storage)
        if args.use_cuda:
            model = model.cuda()
        fine_tuner = PrunningFineTuner_VGG16(model)
        fine_tuner.prune()

        # Evaluate pruned model
        print("\n--- Pruned Model Metrics ---")
        pruned_accuracy = fine_tuner.test()
        pruned_params = count_parameters(model)
        pruned_flops = count_flops(model)
        print(f"Accuracy: {pruned_accuracy:.4f}")
        print(f"Parameters: {pruned_params}")
        print(f"FLOPs: {pruned_flops}")

        # Comparison
        print("\n--- Model Comparison ---")
        print(f"Original Accuracy: {original_accuracy:.4f}, Pruned Accuracy: {pruned_accuracy:.4f}")
        print(f"Original Parameters: {original_params}, Pruned Parameters: {pruned_params} (Reduction: {((original_params - pruned_params) / original_params) * 100:.2f}%)")
        print(f"Original FLOPs: {original_flops}, Pruned FLOPs: {pruned_flops} (Reduction: {((original_flops - pruned_flops) / original_flops) * 100:.2f}%)")


