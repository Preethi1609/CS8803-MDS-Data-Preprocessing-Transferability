import numpy as np
import pandas as pd
import utils
from .experiment_utils import set_random_seed, load_data, build_data, grid_search, makedir, save_result
from model import LogisticRegression
from pipeline.diffprep_flex_pipeline import DiffPrepFlexPipeline
from pipeline.diffprep_fix_pipeline import DiffPrepFixPipeline
import torch
import torch.nn as nn
from trainer.diffprep_trainer import DiffPrepSGD
from utils import SummaryWriter
from .experiment_utils import min_max_normalize
from copy import deepcopy

class DiffPrepExperiment(object):
    """Run auto prep with one set of hyper parameters"""
    def __init__(self, data_dir, dataset, prep_space, model_name, method):
        self.data_dir = data_dir
        self.dataset = dataset
        self.prep_space = prep_space
        self.model_name = model_name
        self.method = method

    def run(self, params, verbose=True):        
        X, y = load_data(self.data_dir, self.dataset)
        X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y, random_state=params["split_seed"])
        
        # pre norm for diffprep flex
        if self.method == "diffprep_flex":
            X_train, X_val, X_test = min_max_normalize(X_train, X_val, X_test)
            params["patience"] = 10
            params["num_epochs"] = 3000

        # set random seed
        set_random_seed(params)

        ## transform pipeline
        # define and fit first step
        if self.method == "diffprep_fix":
            prep_pipeline = DiffPrepFixPipeline(self.prep_space, temperature=params["temperature"],
                                             use_sample=params["sample"],
                                             diff_method=params["diff_method"],
                                             init_method=params["init_method"])
        elif self.method == "diffprep_flex":
            prep_pipeline = DiffPrepFlexPipeline(self.prep_space, temperature=params["temperature"],
                            use_sample=params["sample"],
                            diff_method=params["diff_method"],
                            init_method=params["init_method"])
        else:
            raise Exception("Wrong auto prep method")

        prep_pipeline.init_parameters(X_train, X_val, X_test)
        print("Train size: ({}, {})".format(X_train.shape[0], prep_pipeline.out_features))

        # model
        input_dim = prep_pipeline.out_features
        output_dim = len(set(y.values.ravel()))

        # model = TwoLayerNet(input_dim, output_dim)
        set_random_seed(params)
        if self.model_name == "log":
            model = LogisticRegression(input_dim, output_dim)
        else:
            raise Exception("Wrong model")

        model = model.to(params["device"])

        # loss
        loss_fn = nn.CrossEntropyLoss()

        # optimizer
        model_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params["model_lr"],
            weight_decay=params["weight_decay"],
            momentum=params["momentum"]
        )
        
        if params["prep_lr"] is None:
            prep_lr = params["model_lr"]
        else:
            prep_lr = params["prep_lr"]
    
        prep_pipeline_optimizer = torch.optim.Adam(
            prep_pipeline.parameters(),
            lr=prep_lr,
            betas=(0.5, 0.999),
            weight_decay=params["weight_decay"]
        )

        # scheduler
        # model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, patience=patience, factor=0.1, threshold=0.001)
        prep_pipeline_scheduler = None
        model_scheduler = None

        if params["logging"]:
            logger = SummaryWriter()
        else:
            logger = None

        diff_prep = DiffPrepSGD(prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
                    model_scheduler, prep_pipeline_scheduler, params, writer=logger)

        result, best_model = diff_prep.fit(X_train, y_train, X_val, y_val, X_test, y_test)
        return result, best_model, logger

def run_diffprep(data_dir, dataset, result_dir, prep_space, params, model_name, method):
    print("Dataset:", dataset, "Diff Method:", params["diff_method"], method)

    sample = "sample" if params["sample"] else "nosample"
    diff_prep_exp = DiffPrepExperiment(data_dir, dataset, prep_space, model_name, method)
    best_result, best_model, best_logger, best_params = grid_search(diff_prep_exp, deepcopy(params))
    save_result(best_result, best_model, best_logger, best_params, result_dir, save_model=False)
    print("DiffPrep Finished. val acc:", best_result["best_val_acc"], "test acc", best_result["best_test_acc"])