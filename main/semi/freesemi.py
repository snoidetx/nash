from typing_extensions import override
from dataclasses import dataclass, field, fields, MISSING
from main.utils.roots import setup_roots
from main.utils.saveload import load
import numpy as np

setup_roots()

from ntk import *
from probe import *
from dvutils.Data_Shapley import *
from main.semi.helpers.helper_freesemi import *


def free_semivalue(weights, **kwargs):

    logging.getLogger().setLevel(logging.INFO)

    from torch.multiprocessing import set_start_method, set_sharing_strategy
    import torch.multiprocessing as mp

    set_start_method("spawn")
    set_sharing_strategy("file_system")

    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    #parser = HfArgumentParser(DshapArguments)
    #args, remaining_argv = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    #sys.argv = [sys.argv[0]] + remaining_argv
    # run_yaml_experiment(args.yaml_path, args.just_cache_data, args.num_dp, args.val_sample_num, args.tmc_iter, args.seed, args.file_path)
    
    init_args = {}
    for fld in fields(DshapArguments):
        if fld.name in kwargs:
            init_args[fld.name] = kwargs[fld.name]
        elif fld.default is not MISSING:
            init_args[fld.name] = fld.default
        else:
            raise ValueError(f"No default provided for {fld.name}, and not in kwargs.")

    args = DshapArguments(**init_args)
    args.parallel = kwargs.get('parallel', None)

    try:
        results = load(args.file_path + f"results_{args.seed}seed.pkl")
    except:
        run_yaml_experiment_ps(weights, args.yaml_path, args.just_cache_data, args.dataset_name, args.num_dp, args.val_sample_num,
                               args.tmc_iter, args.seed, args.tmc_seed, args.file_path, args.prompt, args.approximate,
                               args.run_shapley, args.per_point, args.posion, args.signgd, args.early_stopping, args.parallel)
        
        for p in mp.active_children():
            p.terminate()

        results = load(args.file_path + f"results_{args.seed}seed.pkl")

    sampled_idx = results['sampled_idx']
    mapping = np.arange(len(sampled_idx))
    for i in range(len(mapping)):
        mapping[sampled_idx[i]] = i

    ps_shapleys = results['dv_result'][:, 1, :][mapping]
    shapleys = ps_shapleys.mean(axis=1)
    return shapleys, ps_shapleys
