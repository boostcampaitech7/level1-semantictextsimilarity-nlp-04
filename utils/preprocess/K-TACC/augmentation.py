# Making Augset
import pandas as pd
from BERT_augmentation import BERT_Augmentation
from adverb_augmentation import AdverbAugmentation
from aeda import aeda
from tqdm import tqdm
from multiprocessing import Pool
import joblib
from functools import partial
import numpy as np

tqdm.pandas()

BERT_aug = BERT_Augmentation()
random_masking_replacement = BERT_aug.random_masking_replacement
random_masking_insertion = BERT_aug.random_masking_insertion
adverb_aug = AdverbAugmentation()
adverb_gloss_replacement = adverb_aug.adverb_gloss_replacement

orig_train = pd.read_json("sts/datasets/klue-sts-v1.1_train.json")


def apply_random_masking_replacement(x, ratio=0.15):
    return random_masking_replacement(x, ratio=ratio)


# dataset

random_masking_replacement_train = orig_train.copy()
pool = joblib.Parallel(n_jobs=8, prefer="threads")
mapper = joblib.delayed(apply_random_masking_replacement)
tasks = [
    mapper(row) for i, row in random_masking_replacement_train["sentence_1"].items()
]
random_masking_replacement_train["sentence_1"] = pool(tqdm(tasks))

tasks = [
    mapper(row) for i, row in random_masking_replacement_train["sentence_2"].items()
]
random_masking_replacement_train["sentence_2"] = pool(tqdm(tasks))

random_masking_replacement_augset = pd.concat(
    [orig_train, random_masking_replacement_train]
)
random_masking_replacement_augset.drop_duplicates(
    ["sentence_1", "sentence_2"], inplace=True
)
print(len(random_masking_replacement_augset))

random_masking_replacement_augset.reset_index().to_json(
    "sts/datasets/klue-sts-v1.1_train_random_masking_replacement_augset_span_0.15_nospan.json"
)


# random insertion
def apply_random_masking_insertion(x, ratio=0.15):
    return random_masking_insertion(x, ratio=ratio)


ratio = 0.15
random_masking_insertion_train = orig_train.copy()
pool = joblib.Parallel(n_jobs=8, prefer="threads")
mapper = joblib.delayed(apply_random_masking_insertion)
tasks = [mapper(row) for i, row in random_masking_insertion_train["sentence_1"].items()]
random_masking_insertion_train["sentence_1"] = pool(tqdm(tasks))

tasks = [mapper(row) for i, row in random_masking_insertion_train["sentence_2"].items()]
random_masking_insertion_train["sentence_2"] = pool(tqdm(tasks))

random_masking_insertion_augset = pd.concat(
    [orig_train, random_masking_insertion_train]
)
random_masking_insertion_augset.drop_duplicates(
    ["sentence_1", "sentence_2"], inplace=True
)
print(len(random_masking_insertion_augset))
random_masking_insertion_augset.reset_index().to_json(
    "sts/datasets/klue-sts-v1.1_train_random_masking_insertion_augset_span_0.15_nospan.json"
)


# # adverb
adverb_train = orig_train.copy()


def apply_adverb_gloss_replacement(df, ratio):
    df["sentence_1"] = df["sentence_1"].progress_apply(
        lambda x: adverb_gloss_replacement(x)
    )
    df["sentence_2"] = df["sentence_2"].progress_apply(
        lambda x: adverb_gloss_replacement(x)
    )
    return df


adverb_train["sentence_1"] = adverb_train["sentence_1"].progress_apply(
    lambda x: adverb_gloss_replacement(x)
)
adverb_train["sentence_2"] = adverb_train["sentence_2"].progress_apply(
    lambda x: adverb_gloss_replacement(x)
)
adverb_augset = pd.concat([orig_train, adverb_train])
adverb_augset.drop_duplicates(["sentence_1", "sentence_2"], inplace=True)
print(len(adverb_augset))
adverb_augset.reset_index().to_json(
    "sts/datasets/klue-sts-v1.1_train_adverb_augset.json"
)


# # koreda
from koreda import synonym_replacement, random_deletion, random_swap, random_insertion
from aeda import aeda

# synonym_replacement
sr_train = orig_train.copy()
sr_train["sentence_1"] = sr_train["sentence_1"].apply(
    lambda x: " ".join(synonym_replacement(x.split(), 1))
)
sr_train["sentence_2"] = sr_train["sentence_2"].apply(
    lambda x: " ".join(synonym_replacement(x.split(), 1))
)
sr_augset = pd.concat([orig_train, sr_train])
sr_augset.drop_duplicates(["sentence_1", "sentence_2"], inplace=True)
print(len(sr_augset))
sr_augset.reset_index().to_json("sts/datasets/klue-sts-v1.1_train_sr_augset.json")

# random_deletion
rd_train = orig_train.copy()
rd_train["sentence_1"] = rd_train["sentence_1"].apply(
    lambda x: " ".join(random_deletion(x.split(), 0.15))
)
rd_train["sentence_2"] = rd_train["sentence_2"].apply(
    lambda x: " ".join(random_deletion(x.split(), 0.15))
)
rd_augset = pd.concat([orig_train, rd_train])
rd_augset.drop_duplicates(["sentence_1", "sentence_2"], inplace=True)
print(len(rd_augset))
rd_augset.reset_index().to_json("sts/datasets/klue-sts-v1.1_train_rd_augset.json")

# random_swap
rs_train = orig_train.copy()
rs_train["sentence_1"] = rs_train["sentence_1"].apply(
    lambda x: " ".join(random_swap(x.split(), 1))
)
rs_train["sentence_2"] = rs_train["sentence_2"].apply(
    lambda x: " ".join(random_swap(x.split(), 1))
)
rs_augset = pd.concat([orig_train, rs_train])
rs_augset.drop_duplicates(["sentence_1", "sentence_2"], inplace=True)
print(len(rs_augset))
rs_augset.reset_index().to_json("sts/datasets/klue-sts-v1.1_train_rs_augset.json")

# random_insertion
ri_train = orig_train.copy()
ri_train["sentence_1"] = ri_train["sentence_1"].apply(
    lambda x: " ".join(random_insertion(x.split(), 1))
)
ri_train["sentence_2"] = ri_train["sentence_2"].apply(
    lambda x: " ".join(random_insertion(x.split(), 1))
)
ri_augset = pd.concat([orig_train, ri_train])
ri_augset.drop_duplicates(["sentence_1", "sentence_2"], inplace=True)
print(len(ri_augset))
ri_augset.reset_index().to_json("sts/datasets/klue-sts-v1.1_train_ri_augset.json")

# aeda
aeda_train = orig_train.copy()
aeda_train["sentence_1"] = aeda_train["sentence_1"].apply(lambda x: aeda(x))
aeda_train["sentence_2"] = aeda_train["sentence_2"].apply(lambda x: aeda(x))
aeda_augset = pd.concat([orig_train, aeda_train])
aeda_augset.drop_duplicates(["sentence_1", "sentence_2"], inplace=True)
print(len(aeda_augset))
aeda_augset.reset_index().to_json("sts/datasets/klue-sts-v1.1_train_aeda_augset.json")
