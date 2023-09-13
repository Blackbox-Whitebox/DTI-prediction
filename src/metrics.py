from utils import get_predictions_from_similarity, get_predictions_from_prob

def accuracy_from_contrastive_model(p, y):
    p1, p2 = p
    p = get_predictions_from_similarity(p1, p2)
    accuracy = (p == y).float().mean()

    return accuracy


def accuracy_from_classification_model(p, y):
    p = p.argmax(-1)
    accuracy = (p==y).float().mean()
    return accuracy