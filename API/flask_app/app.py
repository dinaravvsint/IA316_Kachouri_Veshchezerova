# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import tensorflow as tf
from keras.layers import Input, Embedding, Flatten, Dot
from keras.models import Model
import numpy as np 

app = Flask(__name__)
app.url_map.strict_slashes = False

def build_model(nb_users, nb_items, embedding_size=30):
    user_id_input = Input(shape=[1],name='user')
    item_id_input = Input(shape=[1], name='item')

    user_embedding = Embedding(output_dim=embedding_size, input_dim=nb_users + 1,
                               input_length=1, name='user_embedding')(user_id_input)
    item_embedding = Embedding(output_dim=embedding_size, input_dim=nb_items + 1,
                               input_length=1, name='item_embedding')(item_id_input)
    user_vecs = Flatten()(user_embedding)
    item_vecs = Flatten()(item_embedding)
    y = Dot(axes=1)([user_vecs, item_vecs]) 
    model = Model(inputs=[user_id_input, item_id_input], outputs=y)
    model.compile(optimizer='adam', loss='MAE')

    return(model)

@app.route("/train", methods=['GET','POST'])
def train():
    
    user_history = [int(user) for user  in request.args.getlist('user_history')]
    item_history = [int(item) for item in request.args.getlist('item_history')]
    rating_history = [int(rating) for rating in request.args.getlist('rating_history')]
    nb_users = int(request.args.get('nb_users'))
    nb_items = int(request.args.get('nb_items'))
    

    global model,graph
    model = build_model(nb_users,nb_items,30)
    model.fit([user_history, item_history], rating_history,
                batch_size=64, 
                epochs=20, 
                validation_split=0.1,
                shuffle=True)
     
    graph = tf.get_default_graph()
 
    return 'Model trained'

@app.route("/predict", methods=['GET','POST'])
def predict():
    with graph.as_default():
        next_user = int(request.args.get('next_user'))
        next_item = int(request.args.get('next_item'))
        rating_predicted = model.predict([np.array(next_user).reshape(1,1), np.array(next_item).reshape(1,1)])[0][0]

        return jsonify({'rating':str(round(rating_predicted))})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80)