import argparse
import os
import sys

import torch
import torchvision
from avalanche.models import IncrementalClassifier
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
)

from benchmarks.generate_scenario import generate_benchmark
from utils.competition_plugins import (
    GPUMemoryChecker,
    TimeChecker
)

from strategies.my_plugin import MyPlugin
from strategies.my_strategy import MyStrategy
from strategies.lwf_unlabelled import LwFUnlabelled
from strategies.xder_extramodel import xder_extramodel
from utils.generic import set_random_seed, FileOutputDuplicator, evaluate
from utils.short_text_logger import ShortTextLogger

from models.resnet18 import load_resnet18

def main(args):
    # --- Generate Benchmark
    benchmark = generate_benchmark(args.config_file)

    # --- Setup model and Device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")

    # --- Initialize Model
    set_random_seed()
    # model = torchvision.models.resnet18()
    model = load_resnet18()
    # This classification head increases its size automatically in avalanche with the number of
    # annotated samples. If you modify the network structure adapt accordingly
    # model.fc = IncrementalClassifier(512, 2, masking=False)
    model.fc = torch.nn.Linear(512, 100)

    # --- Logger and metrics
    # Adjust logger to personal taste
    base_results_dir = os.path.join("results", f"{os.path.splitext(args.config_file)[0]}_{args.run_name}")
    os.makedirs(base_results_dir, exist_ok=True)
    preds_file = os.path.join(base_results_dir, f"pred_{args.config_file}")

    sys.stdout = FileOutputDuplicator(sys.stdout, os.path.join(base_results_dir, "log.txt"), "w")
    sys.stderr = FileOutputDuplicator(sys.stderr, os.path.join(base_results_dir, "error.txt"), "w")
    text_logger = ShortTextLogger(file=sys.stdout)
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=False, stream=False),
        loss_metrics(minibatch=False, epoch=True, experience=False, stream=False),
        loggers=[text_logger],
    )

    # --- Competition Plugins -> check
    # DO NOT REMOVE OR CHANGE THESE PLUGINS:
    competition_plugins = [
        GPUMemoryChecker(max_allowed=8000),
        TimeChecker(max_allowed=600) # 600
    ]

    # --- Your Plugins
    plugins = [
        # Implement your own plugins or use predefined ones
        MyPlugin()
    ]

    # --- Strategy
    cl_strategy = xder_extramodel(model=model,
                             optimizer=torch.optim.Adam(model.parameters(), lr=4e-4),
                             criterion=CrossEntropyLoss(),
                             train_mb_size=64, # 128
                             train_epochs=1,
                             eval_mb_size=256,
                             device=device,
                             plugins=competition_plugins + plugins,
                             evaluator=eval_plugin)

    # --- Sequence of incremental training tasks/experiences
    for exp_idx, (train_exp, unl_ds) in enumerate(zip(benchmark.train_stream, benchmark.unlabelled_stream)):
        # train on current experience / task, head is automatically expanded by monitoring the
        cl_strategy.train(train_exp, unlabelled_ds=unl_ds, num_workers=args.num_workers)
        
        # --- Make prediction on test-set samples
        # preds, gts = evaluate(benchmark.test_stream[0].dataset, cl_strategy.model, device, exp_idx, preds_file, num_workers=args.num_workers)
        # models = [cl_strategy.model, cl_strategy.sdp_model, ]
        models = cl_strategy.model
        # for prevmodel in cl_strategy.prev_model_list:
        #     models.append(prevmodel)
        
        preds, gts = evaluate(benchmark.test_stream[0].dataset, models, device, exp_idx, preds_file, num_workers=args.num_workers)
        # must comment these evaluation lines to create submission files
        # get valid data (except distractor classes)
        # start_t = time.time()
        distractors = list(range(100, 130))
        valid_idx = torch.tensor([int(gt) not in distractors for gt in gts])
        valid_gts = gts[valid_idx]
        valid_preds = preds[valid_idx]
        
        
        cur_cls = train_exp.classes_in_this_experience
        seen_cls.extend(cur_cls)
        seen_cls = list(set(seen_cls))

        # final accuracy
        final_acc = torch.sum(valid_gts == valid_preds) / len(valid_gts)
        
        # accuracy of current task
        cur_idx = torch.tensor([int(gt) in cur_cls for gt in valid_gts])
        cur_preds = valid_preds[cur_idx]
        cur_gts = valid_gts[cur_idx]
        cur_acc = torch.sum(cur_gts == cur_preds) / len(cur_gts)
        
        # accuracy of all the seen classes
        seen_idx = torch.tensor([int(gt) in seen_cls for gt in valid_gts])
        seen_preds = valid_preds[seen_idx]
        seen_gts = valid_gts[seen_idx]
        seen_acc = torch.sum(seen_gts == seen_preds) / len(seen_gts)

        print(f"TEST - EXP {exp_idx}: Accuracy on current exp: {(100*cur_acc):.3f}% | Accuracy on all seen cls: {(100*seen_acc):.3f}% | Final Accuracy: {(100*final_acc):.3f}%")

    print(f"Predictions saved in {preds_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--config_file", type=str, default="scenario_1.pkl")
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    main(args)
