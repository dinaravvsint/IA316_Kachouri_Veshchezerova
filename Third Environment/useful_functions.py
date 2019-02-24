#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gc
import requests 
from time import sleep
import json
from keras.models import Model, Sequential
from keras.layers import Embedding, Flatten, Input, Dense, Dropout, Concatenate, Lambda, Dot
from keras.regularizers import l2
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

def compute_pos_rewards (rewards_history):
    """ Input: Reward History 
        Output: A dict where keys are user ids who bought the item recommended and values correpond to the item prices.
    """
    pos_rewards = {}
    for index, reward in enumerate(rewards_history):
        if reward > 0:
            # index of reward history = index of sublist in the state history
            pos_rewards[index] = reward
    return pos_rewards

def create_pos_data(pos_rewards, state_history, action_history):
    """ Input: pos_rewards, state_history, action_history
        Output : Dataframe containing positive users and items
    """
    pos_users = []
    pos_items = []
    user_features = []

    for user_index in list(pos_rewards.keys()):
        ## retieving history of a given user at position user_index in state history
        user_list = state_history[user_index]
        ## pos_user and pos_item are ids of users and positive items at a given state
        pos_user = user_list[0][0]
        pos_item = user_list[action_history[user_index]][1]
        pos_features_user =user_list[action_history[user_index]][3:5]

        pos_users.append(pos_user)
        pos_items.append(pos_item)
        user_features.append(pos_features_user)

    pos_data = pd.DataFrame()
    pos_data['user_id'] = pos_users
    pos_data['item_id'] = pos_items
    pos_data['feat_users'] =user_features
    
    
    data_grouped = pd.DataFrame(pos_data.groupby('user_id')['item_id'].unique())
    data_grouped.rename(columns={'item_id':'pos_items'},inplace=True)
    pos_data = pos_data.merge(data_grouped, right_index=True, left_on='user_id',how='left')
    return pos_data


def compute_most_similar (state_history, next_state, pos_data):
    features_users = list(pos_data.feat_users) 
    features_users = np.asarray(features_users).reshape(len(features_users),2)
    features_new_state = next_state[0][3:5] 
    features_new_state = np.asarray(features_new_state ).reshape(1,2)
    
    cs = cosine_similarity(features_new_state,features_users)
    pos_most_similar = np.argmax(cs)
    most_similar_user_id = pos_data.user_id[pos_most_similar]
    return most_similar_user_id

def identity_loss(y_true, y_pred):
    """Ignore y_true and return the mean of y_pred
    
    This is a hack to work-around the design of the Keras API that is
    not really suited to train networks with a triplet loss by default.
    """
    return tf.reduce_mean(y_pred + 0 * y_true)


def margin_comparator_loss(inputs, margin=1.):
    """Comparator loss for a pair of precomputed similarities
    
    If the inputs are cosine similarities, they each have range in
    (-1, 1), therefore their difference have range in (-2, 2). Using
    a margin of 1. can therefore make sense.

    If the input similarities are not normalized, it can be beneficial
    to use larger values for the margin of the comparator loss.
    """
    positive_pair_sim, negative_pair_sim = inputs
    return tf.maximum(negative_pair_sim - positive_pair_sim + margin, 0)


def make_interaction_mlp(input_dim, n_hidden=1, hidden_size=64,dropout=0, l2_reg=None):
    mlp = Sequential()
    if n_hidden ==0 : # no hidden layer, outputs is directly a fully connected
        mlp.add(Dense(1, input_dim=input_dim,activation='relu', W_regularizer=l2_reg))
    
    else:
        mlp.add(Dense(hidden_size, input_dim=input_dim, activation='relu', W_regularizer=l2_reg))
        mlp.add(Dropout(dropout))
        
        for i in range(n_hidden-1):
            mlp.add(Dense(hidden_size, activation='relu', W_regularizer=l2_reg))
            mlp.add(Dropout(dropout))
       
        mlp.add(Dense(1, input_dim=input_dim, activation='relu', W_regularizer=l2_reg))
    return mlp

def sample_triplets(pos_data , random_seed=0):
    """Inputs : DataFrame of positive items and their repspectiv users
       
       This function samples negative items at random from the list of items avalaible at a given state 
       without considering postiive items when sampling.
        
        Output: Triplet of [ user id, pos item id, neg item id] 
    """
    neg_items_ids=[]
    rng = np.random.RandomState(random_seed)
    user_ids = pos_data['user_id'].values
    pos_item_ids = pos_data['item_id'].values
    
    ## Sampling item ids from the list with exluding positive items for each user
    for i in range (len(user_ids)):
        l = [i for i in range(30)]
        
        l = list(set(l)-set(list(pos_data.pos_items[i])))        
        neg_items_ids.append(rng.choice(l))

    return [user_ids, pos_item_ids, np.asarray(neg_items_ids)]

def sample_quintuplets(pos_data, state_history, random_seed=0):
    """Smame as sample triplets but with adding features for users, psoitive items and negative items
    """
    neg_items_ids=[]
    features_pos_items =[]
    features_neg_items = []
    features_items = [state_history[:][:][0][i][5:] for i in range(30)]
    rng = np.random.RandomState(random_seed)
    user_ids = pos_data['user_id'].values
    pos_item_ids = pos_data['item_id'].values
    
    ## Retrieving user features 
    features_users = list(pos_data.feat_users)
    features_users = np.expand_dims(np.asarray(features_users), axis=1)
    
    ## Sampling item ids from the list with exluding positive items for each user
    for i in range (len(user_ids)):
        ## Sampling item ids from the list with exluding positive items for each user
        l = [i for i in range(30)]
        l = list(set(l)-set(list(pos_data.pos_items[i])))        
        
        neg_item_chosen = rng.choice(l)
        neg_items_ids.append(neg_item_chosen)
    
        
        ## Retierving positive items features 
        features_pos_items.append(features_items[pos_item_ids[i]])
        
        ## Retrieving negative items features
        features_neg_items.append(features_items[neg_item_chosen]) 
        
    feat_pos_items_final = np.expand_dims(np.asarray(features_pos_items) , axis=1)  
    feat_neg_items_final = np.expand_dims(np.asarray(features_neg_items), axis=1)
    
    return [user_ids, pos_item_ids, np.asarray(neg_items_ids),features_users,feat_pos_items_final, feat_neg_items_final]

def sample_quintuplets_price(pos_data, state_history, random_seed=0):
    """Same as sample triplets but with adding features for users, psoitive items and negative items
    """
    neg_items_ids=[]
    features_pos_items =[]
    features_neg_items = []
    
    ## Retrieving user features 
    features_users = list(pos_data.feat_users)
    features_users = np.expand_dims(np.asarray(features_users), axis=1)
    
    prices = [state_history[:][:][0][i][2] for i in range(30)]
    
    feat_items = [state_history[:][:][0][i][5:] for i in range(30)]
    price_mean = np.asarray(prices).mean()
    price_std = np.asarray(prices).std()
    prices_norm = [(price-price_mean)/price_std for price in prices ] 
    
    for i,feature in  enumerate(feat_items):
        feature.append(prices_norm[i])

    rng = np.random.RandomState(random_seed)
    user_ids = pos_data['user_id'].values
    pos_item_ids = pos_data['item_id'].values
    
    ## Sampling item ids from the list with exluding positive items for each user
    for i in range (len(user_ids)):
        ## Sampling item ids from the list with exluding positive items for each user
        l = [i for i in range(30)]
        l = list(set(l)-set(list(pos_data.pos_items[i])))        
        
        neg_item_chosen = rng.choice(l)
        neg_items_ids.append(neg_item_chosen)
        
        
        ## Retierving positive items features 
        features_pos_items.append(feat_items[pos_item_ids[i]])
        
        ## Retrieving negative items features
        features_neg_items.append(feat_items[neg_item_chosen]) 
        
    feat_pos_items_final = np.expand_dims(np.asarray(features_pos_items) , axis=1)  
    feat_neg_items_final = np.expand_dims(np.asarray(features_neg_items), axis=1)
    
    return [user_ids, pos_item_ids, np.asarray(neg_items_ids),features_users,feat_pos_items_final, feat_neg_items_final]

def build_models(n_users, n_items, user_dim=32, item_dim=30, n_hidden=1, hidden_size=64, dropout=0, l2_reg=0):
    
    ### Defining inputs and the shared embeddings .

    # symbolic input placeholders
    user_input = Input((1,), name='user_input')
    positive_item_input = Input((1,), name='positive_item_input')
    negative_item_input = Input((1,), name='negative_item_input')

    # embeddings
    l2_reg = None if l2_reg == 0 else l2(l2_reg)
    user_layer = Embedding(n_users, user_dim, input_length=1,
                           name='user_embedding', embeddings_regularizer=l2_reg)
    
    # The following embedding parameters will be shared to encode both the positive and negative items.
    
    item_layer = Embedding(n_items, item_dim, input_length=1,
                           name="item_embedding", embeddings_regularizer=l2_reg)

    user_embedding = Flatten()(user_layer(user_input))
    positive_item_embedding = Flatten()(item_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_layer(negative_item_input))
    
    positive_embeddings_pair = Concatenate(name="positive_embeddings_pair")([user_embedding, positive_item_embedding])
    positive_embeddings_pair = Dropout(dropout)(positive_embeddings_pair)

    negative_embeddings_pair = Concatenate( name="negative_embeddings_pair")([user_embedding, negative_item_embedding])
    negative_embeddings_pair = Dropout(dropout)(negative_embeddings_pair)
                       

    # Use a single instance of the MLP created by make_interaction_mlp()
    # and use it twice: once on the positive pair, once on the negative pair
    
    interaction_layers = make_interaction_mlp(user_dim + item_dim, n_hidden=n_hidden, hidden_size=hidden_size,
                                                dropout=dropout, l2_reg=l2_reg)
    
    positive_similarity = interaction_layers(positive_embeddings_pair)  
    negative_similarity = interaction_layers(negative_embeddings_pair)                                  


    # Building the models: 
    # one for inference, one for triplet training
    # NB:  The triplet network model, only used for training
    triplet_loss =  Lambda(margin_comparator_loss, output_shape=(1,),
                          name='comparator_loss')([positive_similarity, negative_similarity])
                         

    deep_triplet_model = Model(input=[user_input,
                                      positive_item_input,
                                      negative_item_input],
                               output=triplet_loss)

    # The match-score model, only used at inference
    deep_match_model = Model(input=[user_input, positive_item_input],
                             output=positive_similarity)

    
    return deep_match_model, deep_triplet_model

def build_models_covariates(n_users, n_items, user_dim=32, item_dim=30, n_hidden=1, hidden_size=64, dropout=0, l2_reg=0):
    
    ### Defining inputs and the shared embeddings .

    # symbolic input placeholders
    user_input = Input((1,), name='user_input')
    positive_item_input = Input((1,), name='positive_item_input')
    negative_item_input = Input((1,), name='negative_item_input')
    
    # adding features 
    features_pos_item_input = Input(shape=[1,3],name='feature_pos_item_input')
    features_neg_item_input = Input(shape=[1,3],name='feature_neg_item_input')
    features_user_input = Input(shape=[1,2],name='feature_user_input')
    

    # embeddings
    l2_reg = None if l2_reg == 0 else l2(l2_reg)
    user_layer = Embedding(n_users, user_dim, input_length=1,
                           name='user_embedding', embeddings_regularizer=l2_reg)
    
    # The following embedding parameters will be shared to encode both the positive and negative items.
    
    item_layer = Embedding(n_items, item_dim, input_length=1,
                           name="item_embedding", embeddings_regularizer=l2_reg)

    user_embedding = Flatten()(user_layer(user_input))
    positive_item_embedding = Flatten()(item_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_layer(negative_item_input))
    
    
    features_user_embedding = Flatten()(features_user_input)
    features_pos_item_embedding = Flatten()(features_pos_item_input)
    features_neg_item_embedding = Flatten()(features_neg_item_input)
    
    
    positive_embeddings_pair = Concatenate(name="positive_embeddings_pair")([user_embedding,features_user_embedding,
                                                                             features_pos_item_embedding,positive_item_embedding])
    positive_embeddings_pair = Dropout(dropout)(positive_embeddings_pair)

    negative_embeddings_pair = Concatenate( name="negative_embeddings_pair")([user_embedding,features_user_embedding, 
                                                                              features_neg_item_embedding,negative_item_embedding])
    negative_embeddings_pair = Dropout(dropout)(negative_embeddings_pair)
                       

    # Use a single instance of the MLP created by make_interaction_mlp()
    # and use it twice: once on the positive pair, once on the negative pair
    
    interaction_layers = make_interaction_mlp(user_dim + item_dim+5, n_hidden=n_hidden, hidden_size=hidden_size,
                                                dropout=dropout, l2_reg=l2_reg)
    
    positive_similarity = interaction_layers(positive_embeddings_pair)  
    negative_similarity = interaction_layers(negative_embeddings_pair)                                  


    # Building the models: 
    # one for inference, one for triplet training
    # NB:  The triplet network model, only used for training
    triplet_loss =  Lambda(margin_comparator_loss, output_shape=(1,),
                          name='comparator_loss')([positive_similarity, negative_similarity])
                         

    deep_triplet_model2 = Model(input=[user_input,
                                      positive_item_input,
                                      negative_item_input,
                                      features_user_input,
                                      features_pos_item_input,
                                      features_neg_item_input],
                               output=triplet_loss)

    # The match-score model, only used at inference
    deep_match_model2 = Model(input=[user_input, 
                                     positive_item_input,
                                     features_user_input, 
                                     features_pos_item_input],
                             output=positive_similarity)

    
    return deep_match_model2, deep_triplet_model2
def build_models_covariates_price(n_users, n_items, user_dim=32, item_dim=30, n_hidden=1, hidden_size=64, dropout=0, l2_reg=0):
    
    ### Defining inputs and the shared embeddings .

    # symbolic input placeholders
    user_input = Input((1,), name='user_input')
    positive_item_input = Input((1,), name='positive_item_input')
    negative_item_input = Input((1,), name='negative_item_input')
    
    # adding features 
    features_pos_item_input = Input(shape=[1,4],name='feature_pos_item_input')
    features_neg_item_input = Input(shape=[1,4],name='feature_neg_item_input')
    features_user_input = Input(shape=[1,2],name='feature_user_input')
    

    # embeddings
    l2_reg = None if l2_reg == 0 else l2(l2_reg)
    user_layer = Embedding(n_users, user_dim, input_length=1,
                           name='user_embedding', embeddings_regularizer=l2_reg)
    
    # The following embedding parameters will be shared to encode both the positive and negative items.
    
    item_layer = Embedding(n_items, item_dim, input_length=1,
                           name="item_embedding", embeddings_regularizer=l2_reg)

    user_embedding = Flatten()(user_layer(user_input))
    positive_item_embedding = Flatten()(item_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_layer(negative_item_input))
    
    
    features_user_embedding = Flatten()(features_user_input)
    features_pos_item_embedding = Flatten()(features_pos_item_input)
    features_neg_item_embedding = Flatten()(features_neg_item_input)
    
    
    positive_embeddings_pair = Concatenate(name="positive_embeddings_pair")([user_embedding,features_user_embedding,
                                                                             features_pos_item_embedding,positive_item_embedding])
    positive_embeddings_pair = Dropout(dropout)(positive_embeddings_pair)

    negative_embeddings_pair = Concatenate( name="negative_embeddings_pair")([user_embedding,features_user_embedding, 
                                                                              features_neg_item_embedding,negative_item_embedding])
    negative_embeddings_pair = Dropout(dropout)(negative_embeddings_pair)
                       

    # Use a single instance of the MLP created by make_interaction_mlp()
    # and use it twice: once on the positive pair, once on the negative pair
    
    interaction_layers = make_interaction_mlp(user_dim + item_dim+6, n_hidden=n_hidden, hidden_size=hidden_size,
                                                dropout=dropout, l2_reg=l2_reg)
    
    positive_similarity = interaction_layers(positive_embeddings_pair)  
    negative_similarity = interaction_layers(negative_embeddings_pair)                                  


    # Building the models: 
    # one for inference, one for triplet training
    # NB:  The triplet network model, only used for training
    triplet_loss =  Lambda(margin_comparator_loss, output_shape=(1,),
                          name='comparator_loss')([positive_similarity, negative_similarity])
                         

    deep_triplet_model4 = Model(input=[user_input,
                                      positive_item_input,
                                      negative_item_input,
                                      features_user_input,
                                      features_pos_item_input,
                                      features_neg_item_input],
                               output=triplet_loss)

    # The match-score model, only used at inference
    deep_match_model4 = Model(input=[user_input, 
                                     positive_item_input,
                                     features_user_input, 
                                     features_pos_item_input],
                             output=positive_similarity)

    
    return deep_match_model4, deep_triplet_model4
