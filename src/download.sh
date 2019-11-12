# download CoQA
DATA_DIR=../data
mkdir -p $DATA_DIR

# download CoQA dataset
wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json -O $DATA_DIR/coqa-train-v1.0.json
wget https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json -O $DATA_DIR/coqa-dev-v1.0.json

# download QuAC dataset
wget https://s3.amazonaws.com/my89public/quac/train_v0.2.json -O $DATA_DIR/quac_train_v0.2.json
wget https://s3.amazonaws.com/my89public/quac/val_v0.2.json -O $DATA_DIR/quac_val_v0.2.json


# # download CoVe
# COVE_DIR=${DATA_DIR}/cove
# mkdir -p $COVE_DIR
# wget https://github.com/rgsachin/CoVe/blob/master/Keras_CoVe.h5 -O $COVE_DIR/Keras_CoVe.h5

# # download ElMO
# ELMO_DIR=${DATA_DIR}/elmo
# mkdir -p $ELMO_DIR
# wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5 -O $ELMO_DIR/lm_weights.hdf5
# wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json -O $ELMO_DIR/options.json

# # download Glove
# GLOVE_DIR=${DATA_DIR}/glove
# mkdir -p $GLOVE_DIR
# wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
# unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR
