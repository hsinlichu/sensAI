import numpy as np
import torch
import contextlib
import torch.nn.functional as F

def apoz_scoring(activation):
    """
    Calculate the Average Percentage of Zeros Score of the feature map activation layer output
    """
    activation = (activation.abs() <= 0.005).float()
    if activation.dim() == 4:
        featuremap_apoz_mat = activation.mean(dim=(0, 2, 3))
    elif activation.dim() == 2:
        featuremap_apoz_mat = activation.mean(dim=(0, 1))
    else:
        raise ValueError(
            f"activation_channels_avg: Unsupported shape: {activation.shape}")
    return featuremap_apoz_mat.mul(100).cpu()


def avg_scoring(activation):
    activation = activation.abs()
    if activation.dim() == 4:
        featuremap_avg_mat = activation.mean(dim=(0, 2, 3))
    elif activation.dim() == 2:
        featuremap_avg_mat = activation.mean(dim=(0, 1))
    else:
        raise ValueError(
            f"activation_channels_avg: Unsupported shape: {activation.shape}")
    return featuremap_avg_mat.cpu()


class ActivationRecord:
    def __init__(self, model, arch):
        self.apoz_scores_by_layer = []
        self.avg_scores_by_layer = []
        self.num_batches = 0
        self.layer_idx = 0
        self._candidates_by_layer = None
        self._model = model
        # switch to evaluate mode
        self._model.eval()
        self._model.apply(lambda m: m.register_forward_hook(self._hook))
        self.arch = arch

    def parse_activation(self, feature_map):
        apoz_score = apoz_scoring(feature_map).numpy()
        avg_score = avg_scoring(feature_map).numpy()

        if self.num_batches == 0:
            self.apoz_scores_by_layer.append(apoz_score)
            self.avg_scores_by_layer.append(avg_score)
        else:
            self.apoz_scores_by_layer[self.layer_idx] += apoz_score
            self.avg_scores_by_layer[self.layer_idx] += avg_score
        self.layer_idx += 1

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for score in self.apoz_scores_by_layer:
            score /= self.num_batches
        for score in self.avg_scores_by_layer:
            score /= self.num_batches

    def record_batch(self, *args, **kwargs):
        # reset layer index
        self.layer_idx = 0
        with torch.no_grad():
            # output is not used
            _ = self._model(*args, **kwargs)
        self.num_batches += 1

    def _hook(self, module, input, output):
        """Apply a hook to RelU layer"""
        if self.arch == "shufflenetv2":
            if module.__class__.__name__ == 'BatchNorm2d':
                self.parse_activation(F.relu(output))
        else:
            if module.__class__.__name__ == 'ReLU':
                self.parse_activation(output)

    def generate_pruned_candidates(self):
        num_layers = len(self.apoz_scores_by_layer)
        thresholds = [73] * num_layers
        avg_thresholds = [0.01] * num_layers

        candidates_by_layer = []
        for layer_idx, (apoz_scores, avg_scores) in enumerate(zip(self.apoz_scores_by_layer, self.avg_scores_by_layer)):
            if self.arch == "mobilenetv2":
                apoz_scores = torch.Tensor(apoz_scores)
                avg_scores = torch.Tensor(avg_scores)
                avg_candidates = [idx for idx, score in enumerate(
                    avg_scores) if score >= avg_thresholds[layer_idx]]
                candidates = [(idx,float(score)) for idx, score in enumerate(apoz_scores) if score >= thresholds[layer_idx]]
                candidates = sorted(candidates, key = lambda x: x[1])[:int(len(candidates)/2)]
                candidates = [x[0] for x in candidates]
            else:
                apoz_scores = torch.Tensor(apoz_scores)
                avg_scores = torch.Tensor(avg_scores)
                print(avg_scores)
                avg_candidates = [idx for idx, score in enumerate(
                    avg_scores) if score >= avg_thresholds[layer_idx]]
                candidates = [x[0] for x in apoz_scores.gt(
                    thresholds[layer_idx]).nonzero().tolist()]
            difference_candidates = list(
                    set(candidates).difference(set(avg_candidates)))
            candidates_by_layer.append(difference_candidates)
            """
            DEBUG: Printing out remaining neuron IDs
            all_neuron = [idx for idx, score in enumerate(avg_scores)]
            remaining = list(set(all_neuron)-set(difference_candidates))
            print("\nThose remaining neuron index for layer ", layer_idx)
            print(remaining)
            """
        print(
            f"Total pruned candidates: {sum(len(l) for l in candidates_by_layer)}")
        return candidates_by_layer
