import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix
import os
import torch
import argparse
import numpy as np
import textattack
import csv
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
from keras import backend as K
from tensorflow.keras import layers
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from utils.load_model_command import load_model
from utils.attack_commands import load_attack
from utils.detect_commands import abstract_extract_reattacks_features_with_check_word
from textattack.attack_results import SkippedAttackResult
from datetime import datetime
from textattack import Attacker
from textattack import AttackArgs
from textattack import Attacker
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


CHECKPOINT_STEP = 100
ADV_LABEL = 1
ORG_LABEL = 0
HIDDEN_SIZE = 64
EPOCHS = 50
PATIENCE = 5
IS_SAME = 0
IS_DIFF = 1
IS_MIXED = 2
MAX_WORD_PADDING = 24
MISCLASSIFIED_LABEL = 1
CORRECT_CLASSIFIED_LABEL = 0

TRAIN_SET_ARGS = {
        "sst2": ("glue","sst2","train"),
        "imdb": ("imdb",None,"train"),
        "ag_news": ("ag_news",None,"train"),
        "cola": ("glue", "cola", "train"),
        "mrpc": ("glue", "mrpc", "train"),
        "qnli": ("glue", "qnli", "train"),
        "rte": ("glue", "rte", "train"),
        "wnli": ("glue", "wnli", "train"),
        "mr": ("rotten_tomatoes", None, "train"),
        "snli": ("snli", None, "train"),
        "yelp": ("yelp_polarity", None, "train"),
        }
TEST_SET_ARGS = {
        "sst2": ("glue","sst2","validation"),
        "imdb": ("imdb",None,"test"),
        "ag_news": ("ag_news",None,"test"),
        "cola": ("glue", "cola", "validation"),
        "mrpc": ("glue", "mrpc", "validation"),
        "qnli": ("glue", "qnli", "validation"),
        "rte": ("glue", "rte", "validation"),
        "wnli": ("glue", "wnli", "validation"),
        "mr": ("rotten_tomatoes", None, "test"),
        "snli": ("snli", None, "test"),
        "yelp": ("yelp_polarity", None, "test"),
        }

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle



def load_dataset_from_huggingface(dataset_name):
    dataset = textattack.datasets.HuggingFaceDataset(
        *dataset_name, shuffle=False
    )
    return dataset

class Sample:
    def __init__(self, text, detect_features, detect_features_reattack, defense_features, predict_label, defense_label):
        self.text = text
        self.detect_features= detect_features
        self.detect_features_reattack= detect_features_reattack
        self.defense_features = defense_features
        self.predict_label = predict_label
        self.defense_label = defense_label




def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def create_arguments():
    """ Create all arguments
    Args:        
    Returns:      
        args (argparse.ArgumentParser):
            all aguments for TRUST

    """
    parser = argparse.ArgumentParser(description='OneWORD')
    parser.add_argument('--dataset',
                        help='A dataset',
                        default="sst2")                        
    parser.add_argument('--attack',
                        help='An attack',
                       default="pwws")
    parser.add_argument('--target',
                        help='Target model',
                        default="cnn-sst2")
    parser.add_argument('--auxiliary_attacks',
                        nargs="*", 
                        help='List of attacks for defense',
                        default = ['pwws','deepwordbug'])
    parser.add_argument('--supporters',
                        nargs="*", 
                        help='List of support models.',
                        default = ['roberta-base-sst2'])
    parser.add_argument('--num_train',
                        help='Number train sample',
                        type=int,
                        default=1000)
    parser.add_argument('--num_dev',
                        help='Number dev sample',
                        type=int,
                        default=100)
    parser.add_argument('--num_test',
                        help='Number test samples',
                        type=int,
                        default=100)
    parser.add_argument('--reattack_type',
                        help='Target model',
                        default="multi_attacks_DL")
    parser.add_argument('--num_check_word',
                        help='Number of processed words',
                        type=int,
                        default=9999)
    parser.add_argument('--num_node',
                        help='Number dev sample',
                        type=int,
                        default=512)
    parser.add_argument('--epochs',
                        help='Number of epochs',
                        type=int,
                        default=1000)
    parser.add_argument('--patience',
                        help='Number of patience',
                        type=int,
                        default=100)
    parser.add_argument('--batch_size',
                        help='Number test samples',
                        type=int,
                        default=32)
    parser.add_argument('--word_ratio',
                        help='word_ratio',
                        type=float,
                        default=1.0)
    parser.add_argument('--detect_ratio',
                        help='ratio for detect adv and org',
                        type=float,
                        default=1.0)
    parser.add_argument('--log_prefix',
                        help='dummy value',
                        default="unknown")
    parser.add_argument('--checkpoint_step',
                        help='number of checkpoint step',
                        type=int,
                        default=10)
    parser.add_argument('--coding',
                        help='coding',
                        default="no")
    parser.add_argument('--handle_exception',
                        help='handle_exception',
                        default="no")
    args = parser.parse_args()
    return args



def read_data(file_name):
    with (open(file_name, 'rb')) as inp:
        org_texts, adv_texts, ground_truths = zip(*pickle.load(inp))
    return org_texts, adv_texts, ground_truths


def save_data(dataset, attack_results, file_name, num_examples_offset = 0):
    org_texts = []
    adv_texts = []
    ground_truths = []
    for i, result in enumerate(attack_results):
        if isinstance(result, SkippedAttackResult):
            __, ground_truth = dataset[i + num_examples_offset]
            org_text = result.original_text()
            adv_text = None
            org_texts.append(org_text)
            adv_texts.append(adv_text)
            ground_truths.append(ground_truth)
        else:
            __, ground_truth = dataset[i + num_examples_offset]
            org_text = result.original_text()
            adv_text = result.perturbed_text()
            org_texts.append(org_text)
            adv_texts.append(adv_text)
            ground_truths.append(ground_truth)
            
    save_object(zip(org_texts,adv_texts, ground_truths), file_name)     



def print_and_log(text):
    print(text)
    supporters_name = '_'.join(args.supporters)
    auxiliary_attacks_name = '_'.join(args.auxiliary_attacks)
    if args.coding == 'no':
        filename = f"{args.log_prefix}_{args.reattack_type}_train-{args.num_train}_dev-{args.num_dev}_test-{args.num_test}_supporters_{supporters_name}_attack_for_defense_{auxiliary_attacks_name}.txt"
    else: # yes
        filename = f"test_coding_only_{args.log_prefix}_{args.reattack_type}_train-{args.num_train}_dev-{args.num_dev}_test-{args.num_test}_supporters_{supporters_name}_attack_for_defense_{auxiliary_attacks_name}.txt"
    filename = filename.replace('/', '_')
    filename = f"{args.dataset}/{args.target}/{args.attack}/results/" + filename
    with open(filename, "a+") as f:
        f.write(text + "\n")


args = create_arguments()



def load_sst2_dataset(file_name = "data/sst2/test.tsv"):
    tsv_file = open(file_name)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    data = []
    is_header = True
    for row in read_tsv:
        if is_header:
            is_header = False
        else:
            data.append((row[0], int(row[1])))
    tsv_file.close()
    dataset = textattack.datasets.Dataset(data)
    return dataset



def extract(text, target, supporters, auxiliary_attacks, batch_size = args.batch_size, word_ratio = args.word_ratio, detect_ratio = args.detect_ratio):
    sample = None
    if text != None:
        # predict_label, defense_label, detect_features, defense_features = abstract_extract_features(text, target, supporters, attack_for_defense, batch_size, word_ratio, detect_ratio)
        # predict_label, defense_label, detect_features, detect_features_reattack, defense_features = abstract_extract_high_conf_reattack_features(text, target, supporters, attack_for_defense, batch_size, word_ratio, detect_ratio)
        # predict_label, defense_label, detect_features, detect_features_reattack, defense_features = abstract_extract_reattacks_features(text, target, supporters, auxiliary_attacks, batch_size, word_ratio, detect_ratio)
        if (args.handle_exception == "yes"):
            predict_label, defense_label, detect_features, detect_features_reattack, defense_features = abstract_extract_reattacks_features_with_check_word(text, target, supporters, auxiliary_attacks, args.num_check_word, batch_size, word_ratio, detect_ratio, handle_exception = True)
        else:
            predict_label, defense_label, detect_features, detect_features_reattack, defense_features = abstract_extract_reattacks_features_with_check_word(text, target, supporters, auxiliary_attacks, args.num_check_word, batch_size, word_ratio, detect_ratio, handle_exception = False)
        sample = Sample(text, detect_features,detect_features_reattack, defense_features, predict_label, defense_label)
    return sample



def extract_and_save(org_texts, adv_texts, ground_truths, feature_file, target, supporters, auxiliary_attacks, batch_size = args.batch_size, word_ratio = args.word_ratio, detect_ratio = args.detect_ratio):
    previous_org_data, previous_adv_data, previous_ground_truths = [], [], []
    if os.path.exists(feature_file):
        previous_org_data, previous_adv_data, previous_ground_truths = load_all_data(feature_file) 
        previous_org_data = list(previous_org_data)
        previous_adv_data = list(previous_adv_data)
        previous_ground_truths = list(previous_ground_truths)
        print_and_log(f"load_dataset with len = {len(previous_org_data)}")
    start_index = len(previous_org_data)

    start=datetime.now()
    for index in range(start_index,len(org_texts)):
        org_sample = extract(org_texts[index], target, supporters, auxiliary_attacks, batch_size = args.batch_size, word_ratio = args.word_ratio, detect_ratio = args.detect_ratio) 
        previous_org_data.append(org_sample)
        adv_sample = extract(adv_texts[index], target, supporters, auxiliary_attacks, batch_size = args.batch_size, word_ratio = args.word_ratio, detect_ratio = args.detect_ratio)
        previous_adv_data.append(adv_sample)
        previous_ground_truths.append(ground_truths[index])
        print(f"processed {index}/{len(org_texts)}")
        if index != 0 and index % args.checkpoint_step == 0:
            save_all_data(previous_org_data, previous_adv_data, previous_ground_truths,feature_file)            
    print(f"number of data = {len(previous_adv_data)}")            

    extract_features_time = datetime.now()-start
    print_and_log(f"total extract features time = {extract_features_time}") 
    number_of_sample = (len(org_texts) - start_index) * 2
    print_and_log(f"number of sample = {number_of_sample}")
    if number_of_sample != 0:
        print_and_log(f"average extract features time = {extract_features_time / number_of_sample}") 
    save_all_data(previous_org_data, previous_adv_data, previous_ground_truths,feature_file)            



def save_all_data(org_data, adv_data, ground_truths, filename):
    save_object(zip(org_data,adv_data, ground_truths), filename)     


def load_all_data(file_name):
    with (open(file_name, 'rb')) as inp:
        org_data, adv_data, ground_truths = zip(*pickle.load(inp))
    return org_data, adv_data, ground_truths






def evaluate_model(model, X, y):
    scores = dict()
    predict = model.predict(X)
    predict = 1 * (predict.squeeze() >= 0.5)
    f_score = f1_score(y, predict, average='binary')
    acccuracy = accuracy_score(y, predict)

    tn, fp, fn, tp = confusion_matrix(y, predict, labels=[0, 1]).ravel()
    print(tn, fp, fn, tp)  # 1 1 1 1
    scores["F1"] = f_score
    scores["accuracy"] = acccuracy
    scores["tpr"] = tp/(tp+fn)
    scores["fpr"] = fp/(fp+tn)
    return scores


def print_history(history, kind = "accuracy"):
    values = history.history[kind]
    final_values = []
    for value in values:
        final_values.append(round(value, 3))
    print_and_log(f"{kind} : {','.join(map(str, final_values))}")

     
def Reattack_MLP_train_classifier(X_train, y_train, X_dev, y_dev) :
    TEST_EPOCHS = args.epochs

    TEST_NUM_NODE = args.num_node # TEMP

    TEST_PATIENCE = args.patience

    BATCH_SIZE = 64

    FEATURES = X_train.shape[1]

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=TEST_PATIENCE, restore_best_weights=True)

    model = tf.keras.Sequential()
    model.add(layers.Dense(TEST_NUM_NODE, activation='relu', input_shape=(FEATURES,)))
    model.add(layers.Dense(TEST_NUM_NODE, activation='relu'))
    model.add(layers.Dense(TEST_NUM_NODE, activation='relu'))
    model.add(layers.Dense(TEST_NUM_NODE, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_dev,y_dev),epochs=TEST_EPOCHS, batch_size = BATCH_SIZE, callbacks=[callback])
    print(history)
    val_acc_per_epoch = history.history['val_accuracy']
    print_history(history, "accuracy")
    print_history(history, "val_accuracy")
    print_history(history, "loss")
    print_history(history, "val_loss")
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print_and_log(f"best_epoch = {best_epoch}")
    scores = model.evaluate(X_dev, y_dev)
    print(scores)
    scores = evaluate_model(model, X_train, y_train)
    print_and_log(f"train_performance = {scores}")
    scores = evaluate_model(model, X_dev, y_dev)
    print_and_log(f"dev_performance = {scores}")

    return model


def convert_to_histogram_features(detect_features, is_support):
    # detect_features = detect_features[M:]
    # detect_features = detect_features[:M]
    histogram_features = []
    for word in detect_features:
        if not is_support:
            N = len(word)
            M = N // 2
            word = word[:M]
        histogram_features.append(sum(word) / len(word))
    histogram_features = sorted(histogram_features, reverse = True)
    return histogram_features


def convert_to_multiple_histogram_features(features, feature_type, is_support):
    final_features = []
    if feature_type == IS_SAME:
        indexes = [0]
    elif feature_type == IS_DIFF:
        indexes = [1]
    elif feature_type == IS_MIXED:
        indexes = [0,1]
    for index in indexes:
        # print(f"feature = {feature}")
        histogram_features = convert_to_histogram_features(features[index], is_support) 
        # print(f"histogram_features = {histogram_features}")
        final_features.append(histogram_features)
    return final_features



def generate_histogram_features(org_data, adv_data, train_ground_truths, feature_type, is_support):
    org_features = []
    targets = []
    for org_sample, adv_sample, ground_truth in zip(org_data, adv_data, train_ground_truths):
        if adv_sample == None or ( adv_sample != None and adv_sample.predict_label != ground_truth and org_sample.predict_label == ground_truth):

            histogram_features = convert_to_multiple_histogram_features(org_sample.detect_features, feature_type, is_support) 

            org_features.append(histogram_features)
            targets.append(ORG_LABEL)
            if adv_sample != None and adv_sample.predict_label != ground_truth:
                histogram_features = convert_to_multiple_histogram_features(adv_sample.detect_features, feature_type, is_support) 
                org_features.append(histogram_features)
                targets.append(ADV_LABEL)
    return org_features, targets


def find_max_word(features):
    max = 0
    for feature in features:
        for sample in feature:
            if len(sample) > max:
                max = len(sample)
    return max


def histogram_padding(X, max_word, value = -1.0):
    features = []
    for x in X:
        x_final = pad_sequences(x, maxlen = max_word, value = value, padding='post', dtype="float32")
        features.append(x_final)
    return features




def histogram_merge(features):
    X = []
    for feature in features:
        merge = feature.flatten()
        X.append(merge)
    X = np.array(X)
    return X


def generate_train_features(org_data, adv_data, train_ground_truths, feature_type, is_support):
    features, targets = generate_histogram_features(org_data, adv_data, train_ground_truths, feature_type, is_support) 

    max_word = find_max_word(features) 

    y = np.array(targets)

    features = histogram_padding(features, max_word) 
    X = histogram_merge(features) 

    return X, y, max_word



def generate_features(org_data, adv_data, train_ground_truths, max_word, feature_type, is_support):
    features, targets = generate_histogram_features(org_data, adv_data, train_ground_truths, feature_type, is_support)

    y = np.array(targets)

    features = histogram_padding(features, max_word)
    X = histogram_merge(features)

    return X, y



def Reattack_LSTM_train_adv(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, feature_type, is_support):    
    X_train, y_train, max_word = generate_train_features(train_org_data, train_adv_data, train_ground_truths, feature_type, is_support) 
    X_dev, y_dev = generate_features(dev_org_data, dev_adv_data, dev_ground_truths, max_word, feature_type, is_support) 

    classifier = Reattack_MLP_train_classifier(X_train, y_train, X_dev, y_dev) # temp 
    
    return classifier, max_word   



def reattack_validation(adv_classifier, max_word, test_org_data, test_adv_data, test_ground_truths, feature_type, is_support):

    X_test, y_test = generate_features(test_org_data, test_adv_data, test_ground_truths, max_word,  feature_type, is_support)
    scores = adv_classifier.evaluate(X_test, y_test, return_dict=True)
    print_and_log(f"Evaluation = {scores}")

    scores = evaluate_model(adv_classifier, X_test, y_test)
    print_and_log(f"test_performance = {scores}")    

    return scores



def split(data, ratio):
    N = len(data)
    num_train = int(N * ratio)
    train = data[:num_train]
    dev = data[num_train:]
    return train, dev


def detection_process_ratio(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type, is_support):

    if dev_org_data == None:
        RATIO = 0.9
        train_org_data, dev_org_data = split(train_org_data, ratio = RATIO) 
        train_adv_data, dev_adv_data = split(train_adv_data, ratio = RATIO)
        train_ground_truths, dev_ground_truths = split(train_ground_truths, ratio = RATIO)

    adv_classifier, max_word = Reattack_LSTM_train_adv(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, feature_type, is_support) 

    return reattack_validation(adv_classifier, max_word, test_org_data, test_adv_data, test_ground_truths, feature_type, is_support)



def detection_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type, is_support):
    return detection_process_ratio(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type, is_support)

    



def get_features_file_path(kind = "test"):
    supporters_name = '_'.join(args.supporters)
    auxiliary_attacks_name = '_'.join(args.auxiliary_attacks)
    assert kind == "test" or kind == "dev" or kind == "train", "only support test, dev, or train"
    if kind == "test":
        filename = f"4_priority_checkpoint_{args.num_check_word}_test_{args.num_test}_supporters_{supporters_name}_auxiliary_attacks_{auxiliary_attacks_name}_features.pkl"
    elif kind == "dev":
        filename = f"4_priority_checkpoint_{args.num_check_word}_dev-{args.num_dev}_supporters_{supporters_name}_auxiliary_attacks_{auxiliary_attacks_name}_features.pkl"
    else:
        filename = f"4_priority_checkpoint_{args.num_check_word}_train-{args.num_train}_supporters_{supporters_name}_auxiliary_attacks_{auxiliary_attacks_name}_features.pkl"
    filename = filename.replace('/', '_')
    filename = "features/" + filename 
    return f'{args.dataset}/{args.target}/{args.attack}/{filename}'


def overall_evaluate(result):
    final_score = {}
    for score in result:
        for name, value in score.items():
            if not (name in final_score):
                final_score[name] = [value]
            else:
                final_score[name].append(value)

    for name, value in final_score.items():
        final_score[name] = sum(value) / len(value)
    return final_score


def filter_dataset(dataset, skip_indexes = {8304,8305}, shuffle = False):
    data = []    
    for index, sample in enumerate(dataset):
        if not index in skip_indexes:
            item, ground_truth = sample
            key = list(item.keys())[0]
            data.append((item[key], ground_truth))

    new_dataset = textattack.datasets.Dataset(data, shuffle = shuffle)
    return new_dataset




def generate_data():
    if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}/features'):
        os.makedirs(f'{args.dataset}/{args.target}/{args.attack}/features')
    if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}/results'):
        os.makedirs(f'{args.dataset}/{args.target}/{args.attack}/results')
    if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}/data'):
        os.makedirs(f'{args.dataset}/{args.target}/{args.attack}/data')
    test_feature_file = get_features_file_path(kind = "test") 
    dev_feature_file = get_features_file_path(kind = "dev")
    train_feature_file = get_features_file_path(kind = "train")
    target = load_model(args.target)
    supporters = []
    attack = load_attack(args.attack, target) 
    for supporter_name in args.supporters:
        supporters.append(load_model(supporter_name)) 
    auxiliary_attacks = []
    for attack_for_defense in args.auxiliary_attacks:
        auxiliary_attacks.append(load_attack(attack_for_defense, target)) 


    if (args.num_dev != 0):
        if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}/data/dev-{args.num_dev}_data.pkl'):
            start=datetime.now()
            if (args.dataset == "sst2"):
                dataset = load_dataset_from_huggingface(TEST_SET_ARGS[args.dataset]) 
            else:
                dataset = load_dataset_from_huggingface(TRAIN_SET_ARGS[args.dataset])
            attack_args = AttackArgs(num_examples=args.num_dev)
            attacker = Attacker(attack, dataset, attack_args)
            attack_results = attacker.attack_dataset()
            attack_time = datetime.now()-start
            if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}'):
                os.makedirs(f'{args.dataset}/{args.target}/{args.attack}')
            print_and_log(f'attack time = {attack_time}')
            
            count = 0
            for result in attack_results:
                if not isinstance(result, SkippedAttackResult):
                    count += 1
            print_and_log(f'{args.dataset}/{args.target}/{args.attack}')
            print_and_log(f'number of attack samples = {count}')

            save_data(dataset, attack_results, f'{args.dataset}/{args.target}/{args.attack}/data/dev-{args.num_dev}_data.pkl') 
        org_texts, adv_texts, ground_truths = read_data(f'{args.dataset}/{args.target}/{args.attack}/data/dev-{args.num_dev}_data.pkl')         

        extract_and_save(org_texts, adv_texts, ground_truths, dev_feature_file, target, supporters, auxiliary_attacks, batch_size = args.batch_size, word_ratio = args.word_ratio, detect_ratio = args.detect_ratio) 

        print_and_log('*' * 80)
        

    if (args.num_test != 0):
        if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}/data/test-{args.num_test}_data.pkl'):
            start=datetime.now()
            if (args.dataset != "sst2"):
                dataset = load_dataset_from_huggingface(TEST_SET_ARGS[args.dataset]) 
            else:
                dataset = load_sst2_dataset()
            attack_args = AttackArgs(num_examples=args.num_test)
            attacker = Attacker(attack, dataset, attack_args)
            attack_results = attacker.attack_dataset()
            attack_time = datetime.now()-start
            if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}'):
                os.makedirs(f'{args.dataset}/{args.target}/{args.attack}')
            print_and_log(f'attack time = {attack_time}')
            
            count = 0
            for result in attack_results:
                if not isinstance(result, SkippedAttackResult):
                    count += 1
            print_and_log(f'{args.dataset}/{args.target}/{args.attack}')
            print_and_log(f'number of attack samples = {count}')

            save_data(dataset, attack_results, f'{args.dataset}/{args.target}/{args.attack}/data/test-{args.num_test}_data.pkl')
        org_texts, adv_texts, ground_truths = read_data(f'{args.dataset}/{args.target}/{args.attack}/data/test-{args.num_test}_data.pkl')        

        extract_and_save(org_texts, adv_texts, ground_truths, test_feature_file, target, supporters, auxiliary_attacks, batch_size = args.batch_size, word_ratio = args.word_ratio, detect_ratio = args.detect_ratio)


        print_and_log('*' * 80)
    if (args.num_train != 0):
        if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}/data/train-{args.num_train}_data.pkl'):
            start=datetime.now()
            dataset = load_dataset_from_huggingface(TRAIN_SET_ARGS[args.dataset])
            if (args.dataset == 'sst2' and args.attack in {"iga", "pso", "faster-alzantot"}):
                dataset = filter_dataset(dataset)
            attack_args = AttackArgs(num_examples=args.num_train)
            attacker = Attacker(attack, dataset, attack_args)
            attack_results = attacker.attack_dataset()
            attack_time = datetime.now()-start
            if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}'):
                os.makedirs(f'{args.dataset}/{args.target}/{args.attack}')
            print_and_log(f'attack time = {attack_time}')
            
            count = 0
            for result in attack_results:
                if not isinstance(result, SkippedAttackResult):
                    count += 1
            print_and_log(f'{args.dataset}/{args.target}/{args.attack}')
            print_and_log(f'number of attack samples = {count}')
            
            save_data(dataset, attack_results, f'{args.dataset}/{args.target}/{args.attack}/data/train-{args.num_train}_data.pkl') 
        org_texts, adv_texts, ground_truths = read_data(f'{args.dataset}/{args.target}/{args.attack}/data/train-{args.num_train}_data.pkl')        

        extract_and_save(org_texts, adv_texts, ground_truths, train_feature_file, target, supporters, auxiliary_attacks, batch_size = args.batch_size, word_ratio = args.word_ratio, detect_ratio = args.detect_ratio) 
        print_and_log('*' * 80)



def get_data():

    test_feature_file = get_features_file_path(kind = "test") 
    dev_feature_file = get_features_file_path(kind = "dev")
    train_feature_file = get_features_file_path(kind = "train")
    train_org_data, train_adv_data, train_ground_truths = load_all_data(train_feature_file)
    dev_org_data, dev_adv_data, dev_ground_truths = None, None, None
    if args.num_dev != 0:
        dev_org_data, dev_adv_data, dev_ground_truths = load_all_data(dev_feature_file) 
    test_org_data, test_adv_data, test_ground_truths = load_all_data(test_feature_file)
    print_and_log(f'{args.dataset}/{args.target}/{args.attack}')
    print_and_log(f'attack = {args.attack}')
    print_and_log(f'target = {args.target}')

    print_and_log(f'num_train = {args.num_train}')        
    print_and_log(f'num_test = {args.num_test}')        
    print_and_log(f'num_dev = {args.num_dev}')        
    print_and_log(f'batch_size = {args.batch_size}')
    print_and_log(f'supporters  = {args.supporters}')
    print_and_log(f'auxiliary_attacks  = {args.auxiliary_attacks}')
    print_and_log(f'num_node  = {args.num_node}')
    print_and_log(f'epochs  = {args.epochs}')
    # print_and_log(f'PATIENCE  = {PATIENCE}')
    return train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths


def restoration_generate_histogram_features(org_data, adv_data, train_ground_truths, feature_type, is_support):
    features = []
    targets = []
    for org_sample, adv_sample, ground_truth in zip(org_data, adv_data, train_ground_truths):
        histogram_features = convert_to_multiple_histogram_features(org_sample.detect_features, feature_type, is_support)

        features.append(histogram_features)
        if org_sample.predict_label == ground_truth:
            targets.append(CORRECT_CLASSIFIED_LABEL)
        else:
            targets.append(MISCLASSIFIED_LABEL)

        if adv_sample != None and adv_sample.predict_label != ground_truth:
            histogram_features = convert_to_multiple_histogram_features(adv_sample.detect_features, feature_type, is_support)
            features.append(histogram_features)
            # histogram_features = convert_to_histogram_features(adv_sample.detect_features_reattack)
            # reattack_features.append(histogram_features)
            targets.append(MISCLASSIFIED_LABEL)
    return features, targets



def restoration_generate_train_features(org_data, adv_data, train_ground_truths, feature_type, is_support):
    features, targets = restoration_generate_histogram_features(org_data, adv_data, train_ground_truths, feature_type, is_support)

    max_word = find_max_word(features)

    y = np.array(targets)

    features = histogram_padding(features, max_word)
    X = histogram_merge(features)

    return X, y, max_word


def restoration_generate_features(org_data, adv_data, train_ground_truths, max_word, feature_type, is_support):
    features, targets = restoration_generate_histogram_features(org_data, adv_data, train_ground_truths, feature_type, is_support) 

    y = np.array(targets)

    features = histogram_padding(features, max_word) 
    X = histogram_merge(features) 

    return X, y



def restoration_Reattack_LSTM_train_adv(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, feature_type, is_support):    
    X_train, y_train, max_word = restoration_generate_train_features(train_org_data, train_adv_data, train_ground_truths, feature_type, is_support)
    X_dev, y_dev = restoration_generate_features(dev_org_data, dev_adv_data, dev_ground_truths, max_word, feature_type, is_support) 

    classifier = Reattack_MLP_train_classifier(X_train, y_train, X_dev, y_dev) # temp 

    return classifier, max_word   


def generate_adv_histogram_features(org_data, adv_data, train_ground_truths, feature_type, is_support):
    features = []
    targets = []
    for org_sample, adv_sample, ground_truth in zip(org_data, adv_data, train_ground_truths):
        if adv_sample != None and adv_sample.predict_label != ground_truth:
            histogram_features = convert_to_multiple_histogram_features(adv_sample.detect_features, feature_type, is_support) 
            features.append(histogram_features)
            targets.append(MISCLASSIFIED_LABEL)
        else:
            histogram_features = convert_to_multiple_histogram_features(org_sample.detect_features, feature_type, is_support)

            features.append(histogram_features)
            if org_sample.predict_label == ground_truth:
                targets.append(CORRECT_CLASSIFIED_LABEL)
            else:
                targets.append(MISCLASSIFIED_LABEL)

    return features, targets


def generate_adv_test_features(org_data, adv_data, train_ground_truths, max_word, feature_type, is_support):
    features, targets = generate_adv_histogram_features(org_data, adv_data, train_ground_truths, feature_type, is_support)

    y = np.array(targets)

    features = histogram_padding(features, max_word)
    X = histogram_merge(features)

    return X, y


def generate_clean_histogram_features(org_data, adv_data, train_ground_truths, feature_type, is_support):
    features = []
    targets = []
    for org_sample, adv_sample, ground_truth in zip(org_data, adv_data, train_ground_truths):

        histogram_features = convert_to_multiple_histogram_features(org_sample.detect_features, feature_type, is_support)

        features.append(histogram_features)
        if org_sample.predict_label == ground_truth:
            targets.append(CORRECT_CLASSIFIED_LABEL)
        else:
            targets.append(MISCLASSIFIED_LABEL)
    return features, targets


def generate_clean_test_features(org_data, adv_data, train_ground_truths, max_word, feature_type, is_support):
    features, targets = generate_clean_histogram_features(org_data, adv_data, train_ground_truths, feature_type, is_support)

    y = np.array(targets)
    features = histogram_padding(features, max_word)
    X = histogram_merge(features)
    return X, y




def restoration_reattack_validation(adv_classifier, max_word, test_org_data, test_adv_data, test_ground_truths, feature_type, is_support):
    X_test, y_test = generate_adv_test_features(test_org_data, test_adv_data, test_ground_truths, max_word, feature_type, is_support) 
    scores = adv_classifier.evaluate(X_test, y_test, return_dict=True)
    print_and_log(f"Evaluation adv text = {scores}")

    adv_scores = evaluate_model(adv_classifier, X_test, y_test)
    print_and_log(f"test_performance  adv text = {adv_scores}")    

    X_test, y_test = generate_clean_test_features(test_org_data, test_adv_data, test_ground_truths, max_word, feature_type, is_support)
    scores = adv_classifier.evaluate(X_test, y_test, return_dict=True)
    print_and_log(f"Evaluation clean text = {scores}")

    org_scores = evaluate_model(adv_classifier, X_test, y_test)
    print_and_log(f"test_performance  clean text = {org_scores}")    
    return adv_scores, org_scores



def restoration_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type, is_support):
    if dev_org_data == None:
        RATIO = 0.9
        train_org_data, dev_org_data = split(train_org_data, ratio = RATIO)
        train_adv_data, dev_adv_data = split(train_adv_data, ratio = RATIO)
        train_ground_truths, dev_ground_truths = split(train_ground_truths, ratio = RATIO)
    adv_classifier, max_word = restoration_Reattack_LSTM_train_adv(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, feature_type, is_support)
    return restoration_reattack_validation(adv_classifier, max_word, test_org_data, test_adv_data, test_ground_truths, feature_type, is_support) 
    
    
def One_WORD_restoration():
    train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths = get_data()
    print_and_log(f'*************SAME attack with support**************')
    NUMBER_RUNNING = 3
    adv_result, org_result = [], []
    for _ in range(NUMBER_RUNNING):
        adv_scores, org_scores = restoration_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_SAME, is_support = True) 
        adv_result.append(adv_scores)
        org_result.append(org_scores)

    adv_final_score = overall_evaluate(adv_result)
    org_final_score = overall_evaluate(org_result)

    print_and_log("_____OVERALL RESULTS_________")
    print_and_log(f"overal adv performance = {adv_final_score}")    
    print_and_log(f"overal org performance = {org_final_score}")    

    # print_and_log(f'*************SAME attack without support**************')
    # NUMBER_RUNNING = 3
    # adv_result, org_result = [], []
    # for _ in range(NUMBER_RUNNING):
    #     adv_scores, org_scores = restoration_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_SAME, is_support = False)
    #     adv_result.append(adv_scores)
    #     org_result.append(org_scores)

    # adv_final_score = overall_evaluate(adv_result)
    # org_final_score = overall_evaluate(org_result)

    # print_and_log("_____OVERALL RESULTS_________")
    # print_and_log(f"overal adv performance = {adv_final_score}")    
    # print_and_log(f"overal org performance = {org_final_score}")    

    print_and_log(f'*************MIXED attack with support**************')
    NUMBER_RUNNING = 3
    adv_result, org_result = [], []
    for _ in range(NUMBER_RUNNING):
        adv_scores, org_scores = restoration_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_MIXED, is_support = True)
        adv_result.append(adv_scores)
        org_result.append(org_scores)

    adv_final_score = overall_evaluate(adv_result)
    org_final_score = overall_evaluate(org_result)

    print_and_log("_____OVERALL RESULTS_________")
    print_and_log(f"overal adv performance = {adv_final_score}")    
    print_and_log(f"overal org performance = {org_final_score}")    

    # print_and_log(f'*************MIXED attack without support**************')
    # NUMBER_RUNNING = 3
    # adv_result, org_result = [], []
    # for _ in range(NUMBER_RUNNING):
    #     adv_scores, org_scores = restoration_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_MIXED, is_support = False)
    #     adv_result.append(adv_scores)
    #     org_result.append(org_scores)

    # adv_final_score = overall_evaluate(adv_result)
    # org_final_score = overall_evaluate(org_result)

    # print_and_log("_____OVERALL RESULTS_________")
    # print_and_log(f"overal adv performance = {adv_final_score}")    
    # print_and_log(f"overal org performance = {org_final_score}")    


    print_and_log(f'*************DIFFERENT attack with support**************')
    NUMBER_RUNNING = 3
    adv_result, org_result = [], []
    for _ in range(NUMBER_RUNNING):
        adv_scores, org_scores = restoration_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_DIFF, is_support = True)
        adv_result.append(adv_scores)
        org_result.append(org_scores)

    adv_final_score = overall_evaluate(adv_result)
    org_final_score = overall_evaluate(org_result)

    print_and_log("_____OVERALL RESULTS_________")
    print_and_log(f"overal adv performance = {adv_final_score}")    
    print_and_log(f"overal org performance = {org_final_score}")    

    # print_and_log(f'*************DIFFERENT attack without support**************')
    # NUMBER_RUNNING = 3
    # adv_result, org_result = [], []
    # for _ in range(NUMBER_RUNNING):
    #     adv_scores, org_scores = restoration_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_DIFF, is_support = False)
    #     adv_result.append(adv_scores)
    #     org_result.append(org_scores)

    # adv_final_score = overall_evaluate(adv_result)
    # org_final_score = overall_evaluate(org_result)

    # print_and_log("_____OVERALL RESULTS_________")
    # print_and_log(f"overal adv performance = {adv_final_score}")    
    # print_and_log(f"overal org performance = {org_final_score}")    


    print_and_log('*' * 80)


def One_WORD_detection():
    train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths = get_data() 

    print_and_log(f'*************SAME attack with support**************')
    NUMBER_RUNNING = 3
    result = []
    for _ in range(NUMBER_RUNNING):
        scores = detection_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_SAME, is_support = True) 
        result.append(scores)
    final_score = overall_evaluate(result) 
    print_and_log(f"overal test performance = {final_score}")           

    # print_and_log(f'*************SAME attack without support**************')
    # NUMBER_RUNNING = 3
    # result = []
    # for _ in range(NUMBER_RUNNING):
    #     scores = detection_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_SAME, is_support = False)
    #     result.append(scores)
    # final_score = overall_evaluate(result)
    # print_and_log(f"overal test performance = {final_score}")           

    print_and_log(f'*************MIXED attack with support **************')
    # NUMBER_RUNNING = 1
    NUMBER_RUNNING = 3
    result = []
    for _ in range(NUMBER_RUNNING):
        scores = detection_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type =IS_MIXED, is_support = True)
        result.append(scores)


    final_score = overall_evaluate(result)
    print_and_log(f"overal test performance = {final_score}")     

    # print_and_log(f'*************MIXED attack without support **************')
    # NUMBER_RUNNING = 3
    # result = []
    # for _ in range(NUMBER_RUNNING):
    #     scores = detection_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_MIXED, is_support = False)
    #     result.append(scores)


    # final_score = overall_evaluate(result)
    # print_and_log(f"overal test performance = {final_score}")     


    print_and_log(f'*************DIFFERENT attack with support **************')
    NUMBER_RUNNING = 3
    result = []
    for _ in range(NUMBER_RUNNING):
        scores = detection_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_DIFF, is_support = True)
        result.append(scores)


    final_score = overall_evaluate(result)
    print_and_log(f"overal test performance = {final_score}")     

    # print_and_log(f'*************DIFFERENT attack without support **************')
    # NUMBER_RUNNING = 3
    # result = []
    # for _ in range(NUMBER_RUNNING):
    #     scores = detection_process(train_org_data, train_adv_data, train_ground_truths, dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths, feature_type = IS_DIFF, is_support = False)
    #     result.append(scores)


    # final_score = overall_evaluate(result)
    # print_and_log(f"overal test performance = {final_score}")     

    print_and_log('*' * 80)

