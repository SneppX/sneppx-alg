import numpy as np

from SneppX_ALG import metrics
from SneppX_ALG.metrics import classification as C
from SneppX_ALG.metrics import detection as D
from SneppX_ALG.metrics import ranking as R
from SneppX_ALG.metrics import regression as G
from SneppX_ALG.metrics import segmentation as S
from SneppX_ALG.metrics import seq2seq as T


def test_confusion_matrix():
    cm = C.confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])
    assert cm.tolist() == [[2, 0], [1, 1]]


def test_accuracy_score():
    assert C.accuracy_score([0, 1, 2, 2], [0, 1, 1, 2]) == 0.75
    assert C.accuracy_score([1, 1], [1, 1]) == 1.0


def test_top_k_accuracy():
    scores = [[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.1, 0.2, 0.7]]
    assert C.top_k_accuracy_score([0, 1, 2], scores, k=1) == 1.0
    assert C.top_k_accuracy_score([1, 2, 0], scores, k=1) == 0.0


def test_precision_recall_f1_binary():
    yt = [1, 1, 0, 0, 1]
    yp = [1, 0, 0, 0, 1]
    assert C.precision_score(yt, yp) == 1.0
    assert abs(C.recall_score(yt, yp) - 2 / 3) < 1e-9
    assert abs(C.f1_score(yt, yp) - 0.8) < 1e-9


def test_precision_recall_fscore_support():
    p, r, f, s = C.precision_recall_fscore_support([1, 1, 0, 0, 1], [1, 0, 0, 0, 1])
    assert p.shape == (2,)
    assert s.tolist() == [2, 3]


def test_roc_auc_score():
    auc = C.roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert abs(auc - 0.75) < 1e-9
    assert C.roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_matthews_and_kappa():
    assert C.matthews_corrcoef([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0
    assert C.cohen_kappa_score([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0


def test_balanced_accuracy():
    assert abs(C.balanced_accuracy_score([0, 1, 1], [0, 1, 0]) - 0.75) < 1e-9


def test_average_precision_score():
    ap = C.average_precision_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert 0.0 < ap <= 1.0


def test_regression_mse_mae_r2():
    assert G.mean_squared_error([1, 2, 3], [1, 2, 3]) == 0.0
    assert abs(G.mean_squared_error([1, 2, 3], [2, 2, 2]) - 2 / 3) < 1e-9
    assert G.r2_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert abs(G.r2_score([1, 2, 3], [2, 2, 2]) - 0.0) < 1e-9


def test_regression_extras():
    assert abs(G.mean_absolute_percentage_error([2, 4], [1, 2]) - 0.5) < 1e-9
    assert abs(G.huber_loss([0.0], [2.0], delta=1.0) - 1.5) < 1e-9
    assert G.root_mean_squared_error([0, 0], [0, 0]) == 0.0
    assert G.max_error([1, 2], [1, 5]) == 3.0


def test_box_iou_and_nms():
    assert D.box_iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
    assert D.box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    assert abs(D.box_iou([0, 0, 2, 2], [1, 0, 3, 2]) - 1 / 3) < 1e-9
    keep = D.nms([[0, 0, 1, 1], [0.1, 0.1, 1.1, 1.1]], [0.9, 0.8], 0.5)
    assert keep == [0]


def test_voc_average_precision():
    ap = D.voc_average_precision([0.0, 0.5, 1.0], [1.0, 1.0, 1.0])
    assert abs(ap - 1.0) < 1e-9


def test_segmentation_metrics():
    yt = [0, 0, 1, 1]
    yp = [0, 0, 1, 1]
    assert S.pixel_accuracy(yt, yp) == 1.0
    assert S.mean_iou(yt, yp) == 1.0
    assert S.dice_coefficient(yt, yp) == 1.0
    assert S.frequency_weighted_iou(yt, yp) == 1.0


def test_ranking_metrics():
    rel = [1, 0, 1, 0, 0]
    assert R.precision_at_k(rel, 1) == 1.0
    assert R.precision_at_k(rel, 5) == 0.4
    assert R.recall_at_k(rel, 2) == 0.5
    assert R.hit_rate_at_k(rel, 1) == 1.0
    assert R.ndcg_at_k([1, 1, 0], 3) == 1.0
    assert R.mean_reciprocal_rank([[0, 1, 0], [1, 0, 0]]) == 0.75


def test_seq2seq_metrics():
    assert T.edit_distance("kitten", "sitting") == 3
    assert abs(T.sentence_bleu("the cat sat", "the cat sat") - 1.0) < 1e-6
    assert abs(T.rouge_l("a b c", "a b c")["fmeasure"] - 1.0) < 1e-9
    assert T.word_error_rate("a b c", "a b c") == 0.0
    assert abs(T.chrf("hello", "hello") - 1.0) < 1e-9
    assert T.exact_match("a b", "a b") == 1.0
    assert T.corpus_bleu([["a", "b", "c"]], [["a", "b", "c"]]) > 0.9
