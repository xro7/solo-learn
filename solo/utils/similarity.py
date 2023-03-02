import torch
from torchmetrics.functional import retrieval_average_precision, retrieval_precision, retrieval_recall
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
import numpy as np


def normalized_average_rank(N: int, ranks: List, return_average_rank=False, remove_identity=True):
    if remove_identity:
        ranks = ranks[1:]
        N = N - 1
    Nrel = len(ranks)
    average_rank = np.mean(ranks)
    nar = (1 / (N * Nrel)) * (np.sum(np.array(ranks)) - ((Nrel * (Nrel + 1)) / 2))
    if return_average_rank:
        return average_rank, nar
    return nar


def f_score(precision, recall):
    score = 2 * (precision * recall) / (precision + recall)
    return score


def open_image(path, aug=None, size=50):
    return np.array(Image.open(path).convert('RGB'))


def plt_images(images, labels=None, n=25, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    if n == 1:
        plt.imshow(images[0])
        plt.axis('off')
    else:
        for i, image in enumerate(images[:n]):
            plt.subplot(5, 10, i + 1)
            plt.imshow(image)
            plt.axis('off')
            if labels is not None:
                plt.title(labels[i])
    fig.tight_layout()
    plt.show()


def evaluate(query_indices, embeddings, targets, paths, retrievers, metric_K, K, verbose=False, plot=False,
             remove_first=True):
    for retriever_class in retrievers:
        nars = []
        average_precision = []
        recall = []
        precision = []

        N = embeddings.shape[0]
        retriever = retriever_class(d_vector=embeddings.shape[1])
        retriever.create_index(embeddings)

        for query_idx in tqdm(query_indices):

            query = embeddings[query_idx]
            query = np.expand_dims(query, 0)
            target = targets[query_idx]
            if verbose:
                print(target)
            if plot:
                plt_images([open_image(paths[query_idx])], labels=target, n=25, figsize=(10, 5))

            if targets.ndim > 1:  # one hot:
                similar_indices = np.where(np.argmax(targets, axis=1) == np.argmax(target))[0]
            else:
                similar_indices = np.where(targets == target)[0]

            if plot:
                plt_images([open_image(paths[i]) for i in similar_indices[:25]], labels=similar_indices[:25], n=25,
                           figsize=(10, 5))
            if verbose:
                print(f'Similar indices length {len(similar_indices)}')

            distances, predicted_indices = retriever.search(query, K, return_distances=True)
            # predicted_indices_of_similar = predicted_indices[boolean_ranking]

            if len(predicted_indices) == N:
                ranks = np.arange(0, N)[np.isin(predicted_indices, similar_indices)]
                nar = normalized_average_rank(N, ranks, return_average_rank=True, remove_identity=True)
                if verbose:
                    print(f'NAR {nar}')
                nars.append(nar[1])

            if plot:
                labels = np.round(distances[:25], 3)
                plt_images([open_image(paths[i]) for i in predicted_indices[:25]], labels=labels, n=25,
                           figsize=(10, 5))
            boolean_ranking = np.isin(predicted_indices, similar_indices)
            ranking = torch.tensor(boolean_ranking)
            if remove_first:
                ranking = ranking[1:]
                similar_images = len(similar_indices) - 1
            else:
                similar_images = len(similar_indices)
            if verbose:
                print(f'Ranking vector length {len(ranking)}')
                print(f'Number of correct@{metric_K} in ranking {ranking[:metric_K].sum()}')
            # map = retrieval_average_precision(torch.linspace(1, 0, N, dtype=torch.float), ranking)
            map_at_k = retrieval_average_precision(torch.linspace(1, 0, metric_K, dtype=torch.float),
                                                   ranking[:metric_K])
            if verbose:
                print(f'AP@{metric_K} {map_at_k}')
            average_precision.append(map_at_k)

            p_at_k = retrieval_precision(torch.linspace(1, 0, len(ranking), dtype=torch.float), ranking,
                                         k=similar_images)
            precision.append(p_at_k)
            if verbose:
                print(f'P@Rel {p_at_k}')

            # if len(ranking) >= N-1:
            r_at_k = ranking[:metric_K].sum() / similar_images
            # r_at_k = retrieval_recall(torch.linspace(1, 0, len(ranking), dtype=torch.float), ranking, k=metric_K_)
            if verbose:
                print(f'R@{metric_K} {r_at_k}')
            recall.append(r_at_k)

        print(retriever_class)
        if len(nars) > 0:
            print(f'Overall NAR {np.mean(nars)}')
        else:
            print('NAR cannot be computed when retrieving only top K predictions')
        print(f'Overall MAP@{metric_K} {np.mean(average_precision)}')
        print(f'Overall P@Rel {np.mean(precision)}')
        if len(recall) > 0:
            print(f'Overall R@{metric_K} {np.mean(recall)}')
        else:
            print(f'R@{metric_K} cannot be computed when retrieving only top K predictions')

        return {'NAR': np.mean(nars), f'MAP@{metric_K}': np.mean(average_precision), 'P@Rel': np.mean(precision),
                f'R@{metric_K}': np.mean(recall), 'f1': f_score(np.mean(precision), np.mean(recall))}


class SimilarityModule:

    def __init__(self, d_vector, distance='euclidean') -> None:
        self.d_vector = d_vector
        self.distance = distance

    def create_index(self, data):
        pass

    def add_to_index(self, vector):
        pass

    def remove_from_index(self, idx):
        pass

    def search(self, query, K):
        pass


class SimpleSimilarityModule(SimilarityModule):

    def __init__(self, d_vector, distance='euclidean') -> None:
        super().__init__(d_vector, distance)
        self.index = None

    def create_index(self, data):
        self.index = data

    def add_to_index(self, vector):
        raise NotImplementedError

    def search(self, query, K=None, return_distances=False):

        if self.distance == 'euclidean':
            distances = np.linalg.norm(self.index - query, ord=2, axis=1)
            predicted_indices = np.argsort(distances)

            if K is None:
                K = len(predicted_indices)
            if return_distances:
                distances = distances[predicted_indices]
                return distances[:K], predicted_indices[:K]
            return predicted_indices[:K]
        else:
            raise NotImplementedError
