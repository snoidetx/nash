from main.shapley.helpers.helper_freeshap import NTKProbePS


def get_ntk_utility(probe: NTKProbePS, train_indices, val_set, per_sample=True):
    val_loss, ps_val_loss, val_acc, ps_val_acc = probe.kernel_regression(train_indices, val_set)
    if per_sample:
        return ps_val_acc, ps_val_loss
    else:
        return val_acc, val_loss
