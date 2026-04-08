"""Utilities for CARPRT evaluation (CLIP logits, data loading)."""

import clip
import torch
from tabulate import tabulate

from datasets import build_dataset
from datasets.utils import build_data_loader


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings *= clip_model.logit_scale.exp()
            clip_weights.append(class_embeddings)

        text_feature = torch.stack(clip_weights, dim=1).cuda()
    return text_feature


def get_clip_logits(images, clip_model, text_feature):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits_matrix = torch.einsum('pcd,nd -> pcn', text_feature, image_features)

        return logits_matrix


def get_res_logits(images, clip_model, text_feature, weights):
    with torch.no_grad():
        data_type = clip_model.dtype
        weights = weights.type(data_type)
        weights = weights.cuda()
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = torch.einsum('pc,pcd -> cd', weights, text_feature)

        logits = image_features @ text_features.t()

        return logits


def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        dataset = build_dataset("imagenet", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=512, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['A', 'V', 'R', 'S']:
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=512, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101', 'dtd', 'eurosat', 'fgvc', 'food101', 'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=512, is_train=False, tfm=preprocess, shuffle=True)
    elif dataset_name in ['cifar10', 'imcifar10', 'cifar100', 'imcifar100']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=512, is_train=False, tfm=preprocess, shuffle=True)
    else:
        raise ValueError("Dataset is not from the chosen list")

    table = []
    table.append(["Dataset", dataset_name])
    table.append(["Classes", f"{dataset.num_classes}"])
    table.append(["Test Size", f"{len(dataset.test)}"])

    print(tabulate(table))

    return test_loader, dataset.classnames, dataset.template
