# import textattack

from email import header
import math
import numpy as np
from textattack.shared import  AttackedText
import inspect
from scipy.special import softmax
import time
import torch


ADV_LABEL = 1
ORG_LABEL = 0

CHANGED_LABEL = 1
UNCHANGED_LABEL = 0

MISCLASSIFIED_LABEL = 1
CORRECT_CLASSIFIED_LABEL = 0


def predict(model, text):
    """ predict a text using a model
    Args:        
        model (textattack.models.wrappers.ModelWrapper):
            a model loaded by TextAttack
        text (str):
            a text
    Returns:      
        distribution (numpy array):
            distribution prediction for the input text
    """   
    distributions = model([text])
    if not(type(distributions[0]) is np.ndarray):
        return distributions[0].cpu().numpy()
    else:
        return distributions[0]



def get_candidate_indexes(pre_transformation_constraints, original_text, transformation):
    """ checking the input text (original text) with pre-constraints
    Args:
        pre_transformation_constraints: 
            list of pre-constrainst
        original_text:
            the original text
        transformation:
            transformation is used for checking
    Returns:
        indexes(array):
            list of indexes statified the pre-constrainst
    """
    indexes = None    
    for constrainst in pre_transformation_constraints:
        if indexes == None:
            indexes = set(constrainst(original_text, transformation))        
        else:
            indexes = indexes.intersection(constrainst(original_text, transformation))        
    return set(indexes)



def get_index_priority(search_method, initial_text, selected_indexes):
    """ sort the selected_indexes by priority
    Args:
        search_method:
            the method determines the priority
        initial_text:
            the original text
        selected_indexes:
            indexes are statisfied the pre-constrainsts
        
    Returns:
        indexes(array):
            list of indexes after sorting
    """  

    if (hasattr(search_method, '_get_index_order')):
        index_order, search_over = search_method._get_index_order(initial_text)
        # print(f"index_order = {index_order}")
        indexes = []
        for index in index_order:
            if index in selected_indexes:
               indexes.append(index) 
        return indexes
    else:
        return selected_indexes  

    


def priority(attack, input_text):
    """ determine the priority for defense
    Args:
        attack (textattack.Attack):
            an attack from TextAttack
        input_text:
            the original text
    Returns:
        indexes(array):
            list of indexes after priority
    """    
    original_text = AttackedText(input_text)

    return priority_by_attack(attack, input_text)



def priority_by_attack(attack, input_text):
    """ determine the priority for defense
    Args:
        attack (textattack.Attack):
            an attack from TextAttack
        input_text:
            the original text
    Returns:
        indexes(array):
            list of indexes after priority
    """    
    original_text = AttackedText(input_text)
    indexes = get_candidate_indexes(attack.pre_transformation_constraints, original_text, attack.transformation)  
    indexes = get_index_priority(attack.search_method, original_text, list(indexes)) 
    return indexes


def transform(transformation, input_text, word_index):
    """ transform a word at the word_index in input_text
    Args:
        transformation:
            transformation of an attack from TextAttack
        input_text:
            the original text
        word_index:
            index of word to modify
    Returns:
        transform_texts(array):
            list of texts after transformation
    """ 
    original_text = AttackedText(input_text)
    transform_texts = transformation(original_text, indices_to_modify = [word_index])
    return transform_texts


def constraint(transformed_texts, attack, input_text, handle_exception = False):
    """ check constrainsts for transformed texts
    Args:        
        transformed_texts:
            transformed texts
        attack:
            An attack from TextAttack framework
        input_text:
            an input text
    Returns:                
        statisfied_texts: 
            all texts in the transformed texts statify the constrainsts
    """ 
    result = []
    if handle_exception:
        try:
            original_text = AttackedText(input_text)
            filtered_texts = attack.filter_transformations(transformed_texts, original_text, original_text)            
            for filter_text in filtered_texts:
                result.append(filter_text.text.strip())
            return result
        except:
            return result
    else:
        original_text = AttackedText(input_text)
        filtered_texts = attack.filter_transformations(transformed_texts, original_text, original_text)            
        for filter_text in filtered_texts:
            result.append(filter_text.text.strip())
        return result
        
    


def batch_model_predict(model_predict, inputs, batch_size=64):
    """Runs prediction on iterable ``inputs`` using batch size ``batch_size``.

    Aggregates all predictions into an ``np.ndarray``.
    """
    outputs = []
    i = 0
    while i < len(inputs):
        batch = inputs[i : i + batch_size]
        batch_preds = model_predict(batch)

        # Some seq-to-seq models will return a single string as a prediction
        # for a single-string list. Wrap these in a list.
        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        # Cast all predictions iterables to ``np.ndarray`` types.
        if not isinstance(batch_preds, np.ndarray):
            batch_preds = np.array(batch_preds)
        outputs.append(batch_preds)
        i += batch_size

    return np.concatenate(outputs, axis=0)



def extract_features_by_comparing_with_target_predict_speed_with_check_word(input_text, target, attack_for_defense, supporters, check_word, batch_size, handle_exception=False):
    victim_distribution = predict(target, input_text)
    predict_index = np.argmax(victim_distribution)     

    word_indexes = priority(attack_for_defense, input_text)
    word_indexes = word_indexes[:check_word]
    features = []

    plaintexts = []
    counts = []

    for word_index in word_indexes:
        transformed_texts = transform(attack_for_defense.transformation, input_text, word_index)
        filtered_texts = constraint(transformed_texts, attack_for_defense, input_text, handle_exception=handle_exception)
        if len(filtered_texts) > 0:
            counts.append(len(filtered_texts))
            plaintexts.extend(filtered_texts)

    if len(plaintexts) == 0:
        return []
    predicts = []
    with torch.no_grad():
        target_predicts = batch_model_predict(target, plaintexts, batch_size = batch_size)
    target_predicts = np.argmax(target_predicts, axis = -1)
    predicts.append(target_predicts)
    
    for supporter in supporters:
        with torch.no_grad():
            sub_predicts =  batch_model_predict(supporter, plaintexts, batch_size = batch_size)
        sub_predicts = np.argmax(sub_predicts, axis = -1)
        predicts.append(sub_predicts)
    features = []
    index = 0
    for count in counts:
        feature = []
        for sub_predicts in predicts:
            for i in range(count):
                if sub_predicts[index + i] != predict_index:
                    feature.append(CHANGED_LABEL)
                else:
                    feature.append(UNCHANGED_LABEL)
        index += count
        features.append(feature)
    return features


def abstract_extract_reattacks_features_with_check_word(input_text, target, supporters, attacks_for_defense, check_word,  batch_size, word_ratio = 1.0, detect_ratio = 0.5,  handle_exception = False, ignore_word_importance = False):    
    features = []
    for attack_for_defense in attacks_for_defense:
        feature = extract_features_by_comparing_with_target_predict_speed_with_check_word(input_text, target, attack_for_defense, supporters, check_word, batch_size, handle_exception = handle_exception) 
        features.append(feature)
    victim_distribution = predict(target, input_text)
    predict_label = np.argmax(victim_distribution)    
    defense_label =  None # dummy
    defense_features = None
    reattack_features = None
    return predict_label, defense_label, features, reattack_features, defense_features




