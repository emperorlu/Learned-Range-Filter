from DeepBloom import DeepBloom
from DeeperBloom import DeeperBloom
from Model import Model
from AlmostPerfectModel import AlmostPerfectModel
from PerfectModel import PerfectModel
from GRUModel import GRUModel
from AlwaysNoModel import AlwaysNoModel
import json
import random
import string
from utils import * 

## Get data to train on
with open('../data/dataset.json', 'r') as f:
    dataset = json.load(f)

positives = dataset['positives']
negatives = dataset['negatives']

def test_almost_perfect_model():
    fp_rate = .05

    train_dev_negatives = negatives[:int(.90 * len(negatives))]
    test_negatives = negatives[int(.90 * len(negatives)):]
    print("Number train, dev", len(train_dev_negatives))
    print("Number test", len(test_negatives))
    data = Data(positives, train_dev_negatives)

    # this parameter is not related to fp_rate
    db = DeepBloom(AlmostPerfectModel(.2), data, fp_rate)
    
    for positive in data.positives:
        assert(db.check(positive))

    false_positives = 0.0
    for negative in data.negatives:
        if db.check(negative):
            false_positives += 1
    print("Train/dev false Positive Rate: " + str(100* false_positives / len(train_dev_negatives)) + "%")

    false_positives = 0.0
    for neg in test_negatives:
        if db.check(neg):
            false_positives += 1
    print("Test false positive rate: ", str(100* false_positives / len(test_negatives)) + "%")

def test_gru_model(positives, negatives, model, train_dev_fraction=0.9, deeper_bloom=False, fp_rate=0.01, fp_fractions=None):
    train_dev_negatives = negatives[:int(train_dev_fraction * len(negatives))]
    test_negatives = negatives[int(train_dev_fraction * len(negatives)):]
    print("Number train, dev", len(train_dev_negatives))
    print("Number test", len(test_negatives))
    print("Number positives ", len(positives))

    data = Data(positives, train_dev_negatives)
    if not deeper_bloom:
        db = DeepBloom(model, data, fp_rate)
        print("Params needed", db.model.model.count_params())
    else:
        db = DeeperBloom(model, data, fp_rate, fp_fractions=fp_fractions)
        total = 0.0
        for i in range(db.k):
            print("Params needed for model", i, db.models[i].model.count_params())
            total += db.models[i].model.count_params()
        print("Total params", total)
    print("Bloom filter bits needed", db.bloom_filter.size)
    # for positive in positives:
    #     assert(db.check(positive))

    # false_positives = 0.0
    # for negative in data.negatives:
    #     if db.check(negative):
    #         false_positives += 1
    # print("Train/dev false Positive Rate: " + str(false_positives / len(train_dev_negatives)))

    false_positives = 0.0
    for neg in test_negatives:
        if db.check(neg):
            false_positives += 1
    print("Test false positive rate: ", str(false_positives / len(test_negatives)))


def test_deeper_bloom(positives, negatives):
    fp_rate = 0.01

    train_dev_negatives = negatives[:int(.9 * len(negatives))]
    test_negatives = negatives[int(.9 * len(negatives)):]
    print("Number train, dev", len(train_dev_negatives))
    print("Number test", len(test_negatives))
    print("Number positives ", len(positives))

    data = Data(positives, train_dev_negatives)

    db = DeeperBloom([AlmostPerfectModel(.2), AlmostPerfectModel(.2), AlmostPerfectModel(.2), AlmostPerfectModel(.2)], data, fp_rate)
    print("Bloom filter bits needed", db.bloom_filter.size)
    # for positive in positives:
    #     assert(db.check(positive))

    false_positives = 0.0
    for negative in data.negatives:
        if db.check(negative):
            false_positives += 1
    print("Train/dev false Positive Rate: " + str(false_positives / len(train_dev_negatives)))

    false_positives = 0.0
    for neg in test_negatives:
        if db.check(neg):
            false_positives += 1
    print("Test false positive rate: ", str(false_positives / len(test_negatives)))

if __name__=='__main__':
    # test_almost_perfect_model()
    # test_gru_model(positives, negatives)
    test_deeper_bloom(positives, negatives)
