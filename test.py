import argparse
import time

import clip
import torch
import torch.nn.functional as F

from utils import (
    build_test_data_loader,
    clip_classifier,
    cls_acc,
    get_clip_logits,
    get_res_logits,
)


def get_arguments():
    parser = argparse.ArgumentParser(description="CARPRT test-time prompt reweighting (CLIP).")
    parser.add_argument(
        '--config',
        dest='config',
        default=None,
        help='Optional; reserved for YAML configs (not used by this script).',
    )
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Dataset id(s), slash-separated, e.g. 'caltech101' or 'I/A'.")
    parser.add_argument(
        '--data-root',
        dest='data_root',
        type=str,
        default='/data/gpfs/projects/punim2161/datasets',
        help='Root directory of benchmark datasets.',
    )
    parser.add_argument(
        '--backbone',
        dest='backbone',
        type=str,
        choices=['RN50', 'ViT-B/16'],
        required=True,
        help='CLIP backbone.',
    )
    parser.add_argument('--temp', dest='temp', type=float, default=1.0, help='Temperature for softmax over prompt weights.')
    return parser.parse_args()


def get_matrix(logits, num_class, num_prompt):
    max_values, max_indices = torch.max(logits, dim=1)
    max_values = max_values.float()

    sum_matrix = torch.zeros((num_prompt, num_class), dtype=torch.float32, device=logits.device)
    for p in range(num_prompt):
        sum_matrix[p].scatter_add_(0, max_indices[p], max_values[p])

    counts_matrix = torch.zeros((num_prompt, num_class), dtype=torch.long, device=logits.device)
    for p in range(num_prompt):
        counts_matrix[p].scatter_add_(0, max_indices[p], torch.ones_like(max_values[p], dtype=torch.long))

    return sum_matrix, counts_matrix


def run_test_carprt(loader, clip_model, text_feature, temp):
    with torch.no_grad():
        num_prompt, num_class, _ = text_feature.shape
        device = text_feature.device

        carprt_weight = torch.zeros((num_prompt, num_class), dtype=torch.float32, device=device)
        carprt_count = torch.zeros((num_prompt, num_class), dtype=torch.long, device=device)

        start_time = time.time()

        for images, _target in loader:
            images = images.to(device)
            logits = get_clip_logits(images, clip_model, text_feature)

            sum_matrix, count_matrix = get_matrix(logits, num_class, num_prompt)

            carprt_weight += sum_matrix
            carprt_count += count_matrix

        carprt_safe_count = torch.where(carprt_count == 0, 1, carprt_count)
        carprt_weight = carprt_weight / carprt_safe_count
        carprt_weight = F.softmax(carprt_weight / temp, dim=0)

        estimate_time = time.time() - start_time

    accuracies_carprt = []
    for images, target in loader:
        images = images.to(device)
        target = target.to(device)
        logits = get_res_logits(images, clip_model, text_feature, carprt_weight)
        acc = cls_acc(logits, target)
        accuracies_carprt.append(acc)

    return sum(accuracies_carprt) / len(accuracies_carprt), estimate_time


def main():
    args = get_arguments()
    _ = args.config

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        print("---- The temperature: {}. ----\n".format(args.temp))

        text_feature = clip_classifier(classnames, template, clip_model)
        acc_wpe, estimate_time = run_test_carprt(test_loader, clip_model, text_feature, args.temp)

        print("---- CARPRT's test accuracy: {:.2f}. ----\n".format(acc_wpe))
        print("---- Estimate weight time: {:.4f} seconds. ----\n".format(estimate_time))


if __name__ == "__main__":
    main()
