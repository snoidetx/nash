from typing import List, Any
from typing_extensions import override
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import os
import numpy as np
from main.utils.saveload import save, load

from main.utils.roots import setup_roots

setup_roots()

from ntk import *
from probe import *
from dvutils.Data_Shapley import *


class NTKProbePS(NTKProbe):
    yaml_tag = '!NTKProbePS'


    @override
    def kernel_regression(self, train_indices, test_set):
        # perform kernel regression for given train dataset and test dataset
        # select a proper submatrix from the full ntk matrix
        k_train = self.ntk[:, train_indices[:, None], train_indices]
        y_train = self.train_labels[train_indices]

        # construct the kernel matrix for the train set
        if self.correction:
            kr_model = NTKRegression_correction_multiclass(k_train, y_train, self.num_labels,
                                                           train_logits=self.train_logits[train_indices],
                                                           test_logits=self.test_logits)
        else:
            if self.approximate_ntk == 'diagonal':
                kr_model = fastNTKRegression(k_train, y_train, self.num_labels, batch_size=50)
            elif self.approximate_ntk == 'inv':
                if len(train_indices) % 500 == 0:
                    self.pre_inv = None
                kr_model = shapleyNTKRegression(k_train, y_train, self.num_labels, self.pre_inv)
            else:
                kr_model = NTKRegression(k_train, y_train, self.num_labels)

        k_test = self.ntk[:, self.ntk.size(2):, :]
        k_test = k_test[:, :, train_indices]
        # improvement note: 500 can be adjustable
        if self.approximate_ntk == 'inv' and len(train_indices) >= 500:
            test_preds, pre_inv = kr_model(k_test, return_inv=True)
            self.pre_inv = pre_inv
        else:
            test_preds = kr_model(k_test)
        test_preds = test_preds.to('cpu')
        test_labels = torch.tensor([i['label'] for i in test_set])

        # Compute accuracy on train and test sets
        ps_test_acc = (test_preds.argmax(dim=1) == test_labels).float().numpy()
        test_acc = ps_test_acc.mean()
        ps_test_loss = F.cross_entropy(test_preds, test_labels, reduction='none').numpy()
        test_loss = ps_test_loss.mean()
        # sanity check
        if test_loss > 1:
            print("bad kernel regression")
            print(
                f"train loss:, {F.cross_entropy(kr_model(k_train), y_train, reduction='mean').item()}, test loss: {test_loss}, test acc: {test_acc}")
        return test_loss, ps_test_loss, test_acc, ps_test_acc


    def kernel_regression_with_external_pre_inv(self, train_indices, test_set, pre_inv):
        # perform kernel regression for given train dataset and test dataset
        # select a proper submatrix from the full ntk matrix
        k_train = self.ntk[:, train_indices[:, None], train_indices]
        y_train = self.train_labels[train_indices]

        # construct the kernel matrix for the train set
        if self.correction:
            kr_model = NTKRegression_correction_multiclass(k_train, y_train, self.num_labels,
                                                           train_logits=self.train_logits[train_indices],
                                                           test_logits=self.test_logits)
        else:
            if self.approximate_ntk == 'diagonal':
                kr_model = fastNTKRegression(k_train, y_train, self.num_labels, batch_size=50)
            elif self.approximate_ntk == 'inv':
                if len(train_indices) % 500 == 0:
                    pre_inv = None
                kr_model = shapleyNTKRegression(k_train, y_train, self.num_labels, pre_inv)
            else:
                kr_model = NTKRegression(k_train, y_train, self.num_labels)

        k_test = self.ntk[:, self.ntk.size(2):, :]
        k_test = k_test[:, :, train_indices]
        # improvement note: 500 can be adjustable
        if self.approximate_ntk == 'inv' and len(train_indices) >= 500:
            test_preds, pre_inv = kr_model(k_test, return_inv=True)
            #self.pre_inv = pre_inv
        else:
            test_preds = kr_model(k_test)
        test_preds = test_preds.to('cpu')
        test_labels = torch.tensor([i['label'] for i in test_set])

        # Compute accuracy on train and test sets
        ps_test_acc = (test_preds.argmax(dim=1) == test_labels).float().numpy()
        test_acc = ps_test_acc.mean()
        ps_test_loss = F.cross_entropy(test_preds, test_labels, reduction='none').numpy()
        test_loss = ps_test_loss.mean()
        # sanity check
        if test_loss > 1:
            print("bad kernel regression")
            print(
                f"train loss:, {F.cross_entropy(kr_model(k_train), y_train, reduction='mean').item()}, test loss: {test_loss}, test acc: {test_acc}")
        return test_loss, ps_test_loss, test_acc, ps_test_acc, pre_inv


class FreeShap(Fast_Data_Shapley):
    yaml_tag = '!FreeShap'


    def __init__(self, dataset, probe_model, num_metric):
        super().__init__(dataset, probe_model, num_metric)


    def _accumulate_vec(self, idxs, marginal_contribs):
        # sv_result: [N, M, V], marginal_contribs: [N, M, V], idxs: permutation (len N)
        self.sv_result[idxs, :, :] += (1.0 / self.tmc_iteration) * marginal_contribs[idxs, :, :]


    @override
    def tmc_one_iteration(self, early_stopping=False, tolerance=0.05):
        def _tmc_compute(idxs, marginal_contribs):
            for metric_idx in range(self.num_metric):
                for idx in idxs:
                    self.sv_result[idx][metric_idx] += (1.0 / self.tmc_iteration) * marginal_contribs[idx][metric_idx]
        
        idxs = np.random.permutation(self.n_participants)
        marginal_contribs = np.zeros([self.n_participants, self.num_metric, self.n_val]) # per-sample

        truncation_counter = 0
        new_score = self.get_null_score()
        new_ps_score = self.get_ps_null_score()
        selected_idx = []
        tmp_inspect = np.zeros([self.n_participants, self.num_metric])
        for n, idx in tqdm(enumerate(idxs), leave=False):
            old_score = new_score
            old_ps_score = new_ps_score
            selected_idx.append(idx)

            new_v_entropy, new_ps_v_entropy, new_acc, new_ps_acc = self.probe_model.kernel_regression(np.array(selected_idx), self.val_set)
            new_score = np.array([-new_v_entropy, new_acc])
            new_ps_score = np.array([-new_ps_v_entropy, new_ps_acc])

            tmp_inspect[n] = new_score
            marginal_contribs[idx] = (new_ps_score - old_ps_score)
            if early_stopping:
                distance_to_full_score = np.abs(new_score - self.get_full_score())
                if (distance_to_full_score <= tolerance * np.abs(self.get_full_score() - self.get_null_score())).all():
                    truncation_counter += 1
                    if truncation_counter > 1:
                        print(n)
                        break
                else:
                    truncation_counter = 0

        self._accumulate_vec(idxs=idxs, marginal_contribs=marginal_contribs)
        self.probe_model.pre_inv = None
        return idxs, marginal_contribs


    def tmc_one_iteration_parallelizable(self, idxs, early_stopping=False, tolerance=0.05):
        def _tmc_compute(idxs, marginal_contribs):
            for metric_idx in range(self.num_metric):
                for idx in idxs:
                    self.sv_result[idx][metric_idx] += (1.0 / self.tmc_iteration) * marginal_contribs[idx][metric_idx]
        
        #idxs = np.random.permutation(self.n_participants)
        marginal_contribs = np.zeros([self.n_participants, self.num_metric, self.n_val]) # per-sample

        truncation_counter = 0
        new_score = self.get_null_score()
        new_ps_score = self.get_ps_null_score()
        selected_idx = []
        tmp_inspect = np.zeros([self.n_participants, self.num_metric])
        pre_inv = None
        for n, idx in tqdm(enumerate(idxs), leave=False):
            old_score = new_score
            old_ps_score = new_ps_score
            selected_idx.append(idx)

            new_v_entropy, new_ps_v_entropy, new_acc, new_ps_acc, pre_inv = self.probe_model.kernel_regression_with_external_pre_inv(np.array(selected_idx), self.val_set, pre_inv)
            new_score = np.array([-new_v_entropy, new_acc])
            new_ps_score = np.array([-new_ps_v_entropy, new_ps_acc])

            tmp_inspect[n] = new_score
            marginal_contribs[idx] = (new_ps_score - old_ps_score)
            if early_stopping:
                distance_to_full_score = np.abs(new_score - self.get_full_score())
                if (distance_to_full_score <= tolerance * np.abs(self.get_full_score() - self.get_null_score())).all():
                    truncation_counter += 1
                    if truncation_counter > 1:
                        #print(n)
                        break
                else:
                    truncation_counter = 0

        #self._accumulate_vec(idxs=idxs, marginal_contribs=marginal_contribs)
        self.probe_model.pre_inv = None
        return idxs, marginal_contribs


    def get_ps_null_score(self):
        null_score = self.get_null_score()
        null_ps_score = np.repeat(null_score[:, None], self.n_val, axis=1)
        return null_ps_score

    
    @override
    def get_full_score(self):
        """To compute the performance on grand coalition"""
        try:
            self.full_score
        except:
            selected_idx = np.arange(self.n_participants)
            v_entropy, _, acc, _ = self.probe_model.kernel_regression(selected_idx, self.val_set)
            self.full_score = np.array([-v_entropy, acc])
        self.probe_model.pre_inv = None
        return self.full_score


    @override
    def run(self,
            data_idx,
            val_data_idx,
            val_split="val",
            iteration=1000,
            method="tmc",
            metric="accu",
            use_cache_ntk=False,
            prompt=True,
            seed=2023,
            num_dp=5000,
            checkpoint=False,
            per_point=False,
            early_stopping=False,
            tolerance=0.05,
            perm_batch_size=128,
            parallel=40):
        """Compute the sv with different method"""
        print("start to compute shapley value")
        if seed != 2023:
            np.random.seed(seed)
        self.data_idx = data_idx
        self.n_participants = len(data_idx)
        self.n_val = len(list(self.dataset._load_data(self.dataset.dev_str)))
        self.tmc_iteration = iteration
        self.sv_result = np.zeros([self.n_participants, self.num_metric, self.n_val])
        if per_point:
            self.sv_result = np.zeros([self.n_participants, self.num_metric, len(val_data_idx)])
        self.mc_cache = []

        train_set = self.dataset.get_idx_dataset(data_idx, split="train")
        val_set = self.dataset.get_idx_dataset(val_data_idx, split=val_split)

        self.train_set = train_set
        self.val_set = val_set

        self.metric = metric
        if method == "tmc":
            # prepare the ntk matrix for the full dataset
            if not use_cache_ntk:
                self.initialize_ntk()
            full_score = self.get_full_score()

            if parallel is not None:
                print("Start parallel with {} workers".format(parallel))
                print("Computing for {} data".format(self.n_participants))
                ss = np.random.SeedSequence(seed)
                child_seeds = ss.spawn(iteration) 

                ###
                # define a worker to avoid pickling self repeatedly
                N_PARTICIPANTS = self.n_participants
                NUM_METRIC = self.num_metric
                N_VAL = self.n_val
                NULL_SCORE = self.get_null_score()
                NULL_PS_SCORE = self.get_ps_null_score()
                PROBE_MODEL = self.probe_model
                VAL_SET = self.val_set
                FULL_SCORE = self.get_full_score()

                def _tmc_worker(child_seed, early_stopping=False, tolerance=0.05):
                    rng = np.random.Generator(np.random.PCG64(child_seed))
                    idxs = rng.permutation(N_PARTICIPANTS)
                    marginal_contribs = np.zeros([N_PARTICIPANTS, NUM_METRIC, N_VAL]) # per-sample

                    truncation_counter = 0
                    new_score = NULL_SCORE
                    new_ps_score = NULL_PS_SCORE
                    selected_idx = []
                    tmp_inspect = np.zeros([N_PARTICIPANTS, NUM_METRIC])
                    pre_inv = None
                    for n, idx in enumerate(idxs):
                        old_score = new_score
                        old_ps_score = new_ps_score
                        selected_idx.append(idx)

                        new_v_entropy, new_ps_v_entropy, new_acc, new_ps_acc, pre_inv = PROBE_MODEL.kernel_regression_with_external_pre_inv(np.array(selected_idx), VAL_SET, pre_inv)
                        new_score = np.array([-new_v_entropy, new_acc])
                        new_ps_score = np.array([-new_ps_v_entropy, new_ps_acc])

                        tmp_inspect[n] = new_score
                        marginal_contribs[idx] = (new_ps_score - old_ps_score)
                        if early_stopping:
                            distance_to_full_score = np.abs(new_score - FULL_SCORE)
                            if (distance_to_full_score <= tolerance * np.abs(FULL_SCORE - NULL_SCORE)).all():
                                truncation_counter += 1
                                if truncation_counter > 1:
                                    print(n)
                                    break
                            else:
                                truncation_counter = 0

                    #self._accumulate_vec(idxs=idxs, marginal_contribs=marginal_contribs)
                    PROBE_MODEL.pre_inv = None
                    return idxs, marginal_contribs
                ###

                BACKEND="threading"
                MAX_NBYTES="64M"
                BATCH_SIZE="auto"
                #perms = [np.random.permutation(self.n_participants) for _ in range(iteration)]
                with tqdm_joblib(tqdm(total=iteration, desc="Permutations", leave=False)):
                    result = Parallel(n_jobs=parallel, backend=BACKEND, batch_size=BATCH_SIZE, max_nbytes=MAX_NBYTES)(
                                delayed(_tmc_worker)(child_seeds[p], early_stopping=early_stopping, tolerance=tolerance)
                                    for p in range(iteration))

                for r in result:
                    self._accumulate_vec(idxs=r[0], marginal_contribs=r[1])

            else:
                for curr_iter in tqdm(range(iteration), desc='[TMC iterations]'):
                    self.tmc_one_iteration(early_stopping=early_stopping, tolerance=tolerance)

            sv_result = self.sv_result
        elif method == "exact":
            self.exact_method(metric)
            sv_result = self.exact_sv_from_mem()
        return sv_result


def run_yaml_experiment_ps(yaml_path, just_cache_data, dataset_name, num_dp, val_sample_num, tmc_iter, seed, tmc_seed, file_path,
                           prompt, approximate, run_shapley, per_point, posion, signgd, early_stopping, parallel):
    """
    Runs an experiment as configured by a yaml config file
    """
    import os
    os.environ["OMP_NUM_THREADS"] = '16'
    os.environ["OPENBLAS_NUM_THREADS"] = '16'
    os.environ["MKL_NUM_THREADS"] = '16'

    # Check if the folder exists, if not, create it
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f"Folder '{file_path}' created.")
    else:
        print(f"Folder '{file_path}' already exists.")

    # Set global torch seed for model initialization etc.
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Take constructed classes from yaml
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args['dataset']
    probe_model = yaml_args['probe_com']
    dshap_com = yaml_args['dshap_com']
    if prompt:
        probe_model.model.init(list_dataset.label_word_list)
    if approximate != "none":
        probe_model.approximate(approximate)
        if approximate == "inv":
            probe_model.normalize_ntk()
    if signgd:
        probe_model.signgd()

    if just_cache_data:
        print("Data caching done. Exiting...")
        return

    # sample a set of data points to conduct data valuation
    np.random.seed(seed)
    random.seed(seed)
    if dataset_name == "sst2":
        dataset = load_dataset("nyu-mll/glue", "sst2")
    elif dataset_name == "mr":
        dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    elif dataset_name == "subj":
        dataset = load_dataset("SetFit/subj")
    elif dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
    elif dataset_name == "rte":
        dataset = load_dataset("nyu-mll/glue", "rte")
    elif dataset_name == "mnli":
        dataset = load_dataset("nyu-mll/glue", "mnli")
    elif dataset_name == "hans":
        dataset = load_dataset("hans")
    elif dataset_name == "mrpc":
        dataset = load_dataset("nyu-mll/glue", "mrpc")
    # Sample 10 data points from the dataset
    train_data = dataset['train']
    train_data = train_data.map(lambda example, idx: {'idx': idx}, with_indices=True)
    train_data = train_data.shuffle(seed).select(range(min(train_data.num_rows, num_dp)))
    sampled_idx = train_data['idx']
    if dataset_name == "mnli":
        val_num = dataset['validation_matched'].num_rows
    elif dataset_name in ["subj", "mrpc", "ag_news"]:
        val_num = dataset['test'].num_rows
    else:
        val_num = dataset['validation'].num_rows
    if val_sample_num > val_num:
        sampled_val_idx = np.arange(val_num)
    else:
        sampled_val_idx = np.random.choice(np.arange(val_num), val_sample_num, replace=False).tolist()

    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    # data Shapley with entk
    print(f"{file_path}{dataset_name}_{model_name}_ntk_seed{seed}_num{num_dp}_sign{signgd}.pkl")
    val_split = "test" if dataset_name in ["subj", "mrpc", "ag_news"] else "val"
    try:
        #with open(f"{file_path}{dataset_name}_{model_name}_ntk_seed{seed}_num{num_dp}_sign{signgd}.pkl", "rb") as f:
        #    ntk = pickle.load(f)
        ntk = load(file_path + f"ntk_{seed}seed.pkl")
        print("++++++++++++++++++++++++++++++++++++using cached ntk++++++++++++++++++++++++++++++++++++")
        probe_model.get_cached_ntk(ntk)
        probe_model.get_train_labels(list_dataset.get_idx_dataset(sampled_idx, split="train"))
    except:
        print("++++++++++++++++++++++++++++++++++no cached ntk, computing+++++++++++++++++++++++++++++++++++")
        train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
        val_set = list_dataset.get_idx_dataset(sampled_val_idx, split=val_split)
        # Given that train_loader and val_loader are provided in run(), prepare datasets
        # Set parameters for ntk computation
        # compute ntk matrix
        ntk = probe_model.compute_ntk(train_set, val_set)
        # save the ntk matrix
        save(ntk, file_path + f"ntk_{seed}seed.pkl")
        #with open(f"{file_path}{dataset_name}_{model_name}_ntk_seed{seed}_num{num_dp}_sign{signgd}.pkl", "wb") as f:
        #    pickle.dump(ntk, f)
        print("++++++++++++++++++++++++++++++++++saving ntk cache+++++++++++++++++++++++++++++++++++")
    if run_shapley:
        checkpoint=True
        if per_point:
            checkpoint=False
        print("early_stopping", early_stopping)
        print("Start computing Shapley...")
        dv_result = dshap_com.run(data_idx=sampled_idx, val_data_idx=sampled_val_idx, val_split=val_split, iteration=tmc_iter,
                                  use_cache_ntk=True, prompt=prompt, seed=tmc_seed, num_dp=num_dp,
                                  checkpoint=checkpoint, per_point=per_point, early_stopping=early_stopping, parallel=parallel)

        mc_com = np.array(dshap_com.mc_cache)
        if per_point:
            result_dict = {'dv_result': dv_result,  # entropy, accuracy
                           'sampled_idx': sampled_idx}
            with open(f"{file_path}{dataset_name}_{model_name}_shapley_result_seed{seed}_num{num_dp}_appro{approximate}_sign{signgd}_earlystop{early_stopping}_tmc{tmc_seed}_iter{tmc_iter}.pkl",
                      "wb") as f:
                pickle.dump(result_dict, f)
        else:
            ac_com = np.array(dshap_com.ac_cache)
            result_dict = {'dv_result': dv_result,  # entropy, accuracy
                           'mc_com': mc_com,
                           'ac_com': ac_com,
                           'sampled_idx': sampled_idx}
            save(result_dict, file_path + f"results_{seed}seed.pkl")
            #with open(f"{file_path}{dataset_name}_{model_name}_shapley_result_seed{seed}_num{num_dp}_appro{approximate}_sign{signgd}_posion{posion}_earlystop{early_stopping}_tmc{tmc_seed}_iter{tmc_iter}.pkl", "wb") as f:
            #    pickle.dump(result_dict, f)
