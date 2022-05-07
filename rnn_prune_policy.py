import numpy as np
import torch
import contextlib
import torch.nn.functional as F

def apoz_scoring(activation):
    """
    Calculate the Average Percentage of Zeros Score of the feature map activation layer output
    """
    activation = (activation.abs() <= 0.05).float()
    if activation.dim() == 2:
        featuremap_apoz_mat = activation.mean(dim=(0, 1))
    else:
        raise ValueError(
            f"activation_channels_avg: Unsupported shape: {activation.shape}")
    return featuremap_apoz_mat.mul(100).cpu()


def avg_scoring(activation):
    activation = activation.abs()
    if activation.dim() == 2:
        featuremap_avg_mat = activation.mean(dim=(0, 1))
    else:
        raise ValueError(
            f"activation_channels_avg: Unsupported shape: {activation.shape}")
    return featuremap_avg_mat.cpu()


class DiffRecord:
    def __init__(self, model, arch):
        self.apoz_hx_by_timestep = []
        self.avg_hx_by_timestep = []
        self.apoz_cx_by_timestep = []
        self.avg_cx_by_timestep = []
        self.num_batches = 0
        self.time_step = 0
        self._candidates_by_layer = None
        self._model = model
        # switch to evaluate mode
        self._model.eval()
        self._model.apply(lambda m: m.register_forward_hook(self._hook))
        self.arch = arch
        self.last_hx = torch.randn(64, model.hidden_size)
        self.last_cx = torch.randn(64, model.hidden_size)


    def parse_activation(self, output_hc):
        hx = output_hc[0]
        cx = output_hc[1]

        if self.time_step>0:
            diff_hx = hx-self.last_hx
            diff_cx = cx-self.last_cx
            # diff_hx = (hx-self.last_hx)/self.last_hx
            # diff_cx = (cx-self.last_cx)/self.last_cx
            apoz_score_hx = apoz_scoring(diff_hx)
            apoz_score_cx = apoz_scoring(diff_cx)
            avg_score_hx = avg_scoring(diff_hx)
            avg_score_cx = avg_scoring(diff_cx)
            if self.num_batches == 0:
                self.apoz_hx_by_timestep.append(apoz_score_hx)
                self.apoz_cx_by_timestep.append(apoz_score_cx)
                self.avg_hx_by_timestep.append(avg_score_hx)
                self.avg_cx_by_timestep.append(avg_score_cx)
            else:
                self.apoz_hx_by_timestep[self.time_step-1]+=apoz_score_hx
                self.apoz_cx_by_timestep[self.time_step-1]+=apoz_score_cx
                self.avg_hx_by_timestep[self.time_step-1]+=avg_score_hx
                self.avg_cx_by_timestep[self.time_step-1]+=avg_score_cx
        self.last_hx = hx
        self.last_cx = cx    
        self.time_step += 1

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for score in self.apoz_hx_by_timestep:
            score /= self.num_batches
        for score in self.apoz_cx_by_timestep:
            score /= self.num_batches
        for score in self.avg_hx_by_timestep:
            score /= self.num_batches
        for score in self.avg_cx_by_timestep:
            score /= self.num_batches

    def record_batch(self, *args, **kwargs):
        # reset layer index
        self.time_step = 0
        with torch.no_grad():
            # output is not used
            _ = self._model(*args, **kwargs)
        self.num_batches += 1

    def _hook(self, module, input, output):
        """Apply a hook to LSTMCell layer"""
        if module.__class__.__name__ == 'LSTMCell':
            self.parse_activation(output)

    def showActivation(self):
        print(">>>>>>>>>>>Activation<<<<<<<<<<<<")
        print("apoz_hx_by_timestep.size = "+str(len(self.apoz_hx_by_timestep)))
        print(np.mean(self.apoz_hx_by_timestep))
        print("apoz_cx_by_timestep.size = "+str(len(self.apoz_cx_by_timestep)))
        print(np.mean(self.apoz_cx_by_timestep))
        print("avg_hx_by_timestep.size = "+str(len(self.avg_hx_by_timestep)))
        print(np.mean(self.avg_hx_by_timestep))
        print("avg_cx_by_timestep.size = "+str(len(self.avg_cx_by_timestep)))
        print(np.mean(self.avg_cx_by_timestep))
        # print("avg_scores_by_layer.size = "+str(len(self.avg_scores_by_layer)))

    def generate_pruned_candidates(self):
        num_timestep = len(self.apoz_hx_by_timestep)
        apoz_thresholds_hx = [np.mean(self.apoz_hx_by_timestep)] * num_timestep
        avg_thresholds_hx = [np.mean(self.avg_hx_by_timestep)] * num_timestep
        apoz_thresholds_cx = [np.mean(self.apoz_cx_by_timestep)] * num_timestep
        avg_thresholds_cx = [np.mean(self.avg_cx_by_timestep)] * num_timestep
        self.showActivation()

        candidates_by_timestep = []
        for time_idx in range(num_timestep):
            apoz_score_hx = self.apoz_hx_by_timestep[time_idx]
            avg_score_hx = self.avg_hx_by_timestep[time_idx]
            apoz_score_cx = self.apoz_cx_by_timestep[time_idx]
            avg_score_cx = self.avg_cx_by_timestep[time_idx]
            if apoz_score_hx>apoz_thresholds_hx[time_idx] and avg_score_hx<avg_thresholds_hx[time_idx]:
            	if apoz_score_cx>apoz_thresholds_cx[time_idx] and avg_score_cx<avg_thresholds_cx[time_idx]:
                	candidates_by_timestep.append(time_idx+1)
        print("Total pruned candidates: "+ str(len(candidates_by_timestep)))
        return candidates_by_timestep






    
