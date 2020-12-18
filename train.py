from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

from typing import List, Dict
import os
import argparse
import random
import json
import sklearn

# external lib imports:
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, optimizers

from data import read_instances,generate_batches,get_fake_sample
from util import load_pretrained_model
from main_model import MainEncoderDecoder
from probing_model import ProbingEncoderDecoder
import Clustering
import Calculate_Cluster_Weights








def train(model: models.Model,
          optimizer: optimizers.Optimizer,
          train_instances: List[Dict[str, np.ndarray]],
          validation_instances: List[Dict[str, np.ndarray]],
          num_epochs: int,
          max_length_train: int,
          max_length_validation: int,
          embedding_dim: int,
          batch_size: int,
          number_of_clusters: int,
        hidden_units_in_autoencoder_layers: List[int],
          serialization_dir: str = None) -> tf.keras.Model:
    """
    Trains a model on the give training instances as configured and stores
    the relevant files in serialization_dir. Returns model and some important metrics.
    """
    tensorboard_logs_path = os.path.join(serialization_dir, f'tensorboard_logs')
    tensorboard_writer = tf.summary.create_file_writer(tensorboard_logs_path)
    #best_epoch_validation_accuracy = float("-inf")
    best_epoch_loss = float("inf")
    best_epoch_validation_silhouette_score = float("-inf")
    KMeans_1 = Clustering.KMeansClustering

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")
        total_training_loss = 0
        #total_abstract_train_part_hidden_value = []
        total_abstract_train_part_hidden_value = 0
        total_abstract_validation_part_hidden_value = []
        k = 0

        list_of_trainable_variables = []
        print("\nGenerating Training batches:")
        train_batches,embed_dim_train = generate_batches(train_instances, batch_size, max_length_train, embedding_dim)
        print("Generating Validation batches:")
        validation_batches,embed_dim_validation = generate_batches(validation_instances, batch_size, max_length_validation, embedding_dim)
        noise_data_train = np.random.normal(loc=0, scale=0.1,
                                            size=[len(train_instances) * 2, max_length_train, embed_dim_train])
        noise_data_validation = np.random.normal(loc=0, scale=0.1,
                                                 size=[len(validation_instances) * 2, max_length_validation,
                                                       embed_dim_validation])
        train_batch_tickers = [batch_inputs.pop("Ticker") for batch_inputs in train_batches]
        validation_batch_tickers = [batch_inputs.pop("Ticker") for batch_inputs in validation_batches]
        train_batch_dates = [batch_inputs.pop("Date") for batch_inputs in train_batches]
        validation_batch_dates = [batch_inputs.pop("Date") for batch_inputs in validation_batches]
        train_batch_asset_returns = [batch_inputs.pop('Asset_Returns') for batch_inputs in train_batches]
        validation_batch_asset_returns = [batch_inputs.pop('Asset_Returns') for batch_inputs in validation_batches]
        generator_tqdm = tqdm(list(zip(train_batches, train_batch_tickers)))
        for index, (batch_inputs,batch_tickers) in enumerate(generator_tqdm):
            with tf.GradientTape() as tape:
                # if epoch == 0:
                noise = noise_data_train[k * batch_size * 2: (k + 1) * batch_size * 2, :]

                fake_input, train_real_fake_labels =  get_fake_sample(batch_inputs['inputs'])
                real_batch_inputs = batch_inputs['inputs']
                batch_inputs['inputs'] = np.concatenate((batch_inputs['inputs'], fake_input), axis=0)
                batch_inputs['real_fake_label'] = train_real_fake_labels
                batch_inputs['noise'] = noise
                batch_inputs['Validation_Inputs'] = validation_batch_asset_returns[index]
                batch_inputs['Training_Inputs'] = train_batch_asset_returns[index]
                batch_inputs['batch_tickers'] = batch_tickers
                loss,f_new = model(batch_inputs, epoch,training=True)
                if epoch % 2 == 0:#
                    batch_inputs['F_Updated_Value'] = f_new

                """
                if epoch == 0:
                    part_hidden_val = np.array(hidden_state).reshape(-1,2*sum(hidden_units_in_autoencoder_layers) )#np.sum(config.hidden_size) * 2
                    W = part_hidden_val.T
                    U, sigma, VT = np.linalg.svd(W)
                    sorted_indices = np.argsort(sigma)
                    topk_evecs = VT[sorted_indices[:-number_of_clusters - 1:-1], :]
                    F_new = topk_evecs.T
                    batch_inputs['F_Updated_Value'] = F_new
                    #total_abstract_train_part_hidden_value.append(part_hidden_val)
                    total_abstract_train_part_hidden_value = part_hidden_val
                    km = KMeans(n_clusters=number_of_clusters)
                    cluster_labels_training = km.fit_predict(X=total_abstract_train_part_hidden_value)
                    estimated_return_vector, estimated_covariance_matrix, cluster_weights, cluster_variance, trainable_variables_loss, list_of_trainable_variables = Calculate_Cluster_Weights.calculate_train_data_return_vector_for_all_stocks(
                        cluster_labels_training, real_batch_inputs)
                    asset_weights = [value for key,value in cluster_weights.items()]
                    #asset_weights = np.perm(asset_weights,cluster_labels_training)
                    estimated_return_vector = tf.concat(estimated_return_vector, axis=0)
                    validation_period_out_of_sample_returns = validation_batches[index]['inputs']
                    mean_of_validation_period_out_of_sample_returns = tf.math.reduce_mean(
                        validation_period_out_of_sample_returns, axis=1)
                    #difference_vector = estimated_return_vector - mean_of_validation_period_out_of_sample_returns
                    difference_vector = tf.math.pow(0,2)
                    #difference_vector = tf.tensordot(difference_vector,asset_weights,axis=0)
                    #mean_squared_loss_of_in_sample_and_out_of_sample_returns = tf.keras.losses.mean_squared_error(
                        #mean_of_validation_period_out_of_sample_returns, estimated_return_vector)
                    #loss += difference_vector
                    silhouette_score_training = silhouette_score(total_abstract_train_part_hidden_value,
                                                                 cluster_labels_training)
                    loss += 1e-4  * trainable_variables_loss
                    # calculate_train_data_return_vector_for_all_stocks(cluster_labels_training,)

                else:
                    batch_inputs.pop('F_Updated_Value',None)
                """

                regularization_loss = 0
                #list_of_trainable_variables =
                for var in model.trainable_variables:
                    #print(var)
                    list_of_trainable_variables.append(var)
                    #if (var.name.find("Returns Vector") == -1) and (var.name.find("Estimated_Covariance") == -1):
                    regularization_loss += tf.math.add_n([tf.nn.l2_loss(var)])
                loss += 1e-4  *(regularization_loss)
                k+=1

                grads = tape.gradient(loss, list_of_trainable_variables)
            optimizer.apply_gradients(zip(grads, list_of_trainable_variables))
            total_training_loss += loss
            description = ("Average training loss: %.2f "
                           % (total_training_loss / len(train_instances)))
            generator_tqdm.set_description(description, refresh=False)
        average_training_loss = total_training_loss / len(train_instances)
        if average_training_loss < best_epoch_loss:
            print("Model with best training loss  so far: %.2f. Saving the model."
                  % (average_training_loss))
            best_epoch_loss = average_training_loss
            model.save_weights(os.path.join(serialization_dir, f'model.ckpt'))
        """
        if epoch % 10 == 0 and epoch != 0:
            print(total_abstract_train_part_hidden_value[0],total_abstract_train_hidden_state[0])#
            concatenated_total_abstract_train_part_hidden_value = tf.concat(total_abstract_train_part_hidden_value,axis=1)
            concatenated_total_abstract_train_part_hidden_value = concatenated_total_abstract_train_part_hidden_value.numpy()
            silhouette_score_for_k_means_clustering,cluster_labels, kmeans = KMeans.cluster_hidden_states(concatenated_total_abstract_train_part_hidden_value)
        """
        """
        total_validation_loss = 0

        generator_tqdm = tqdm(list(zip(validation_batches, validation_batch_tickers)))
        k=0

        for index, (batch_inputs,batch_tickers) in enumerate(generator_tqdm):
            #if epoch == 0:
            noise = noise_data_validation[k * batch_size * 2: (k + 1) * batch_size * 2, :]

            fake_input, train_real_fake_labels = get_fake_sample(batch_inputs['inputs'])
            batch_inputs['inputs'] = np.concatenate((batch_inputs['inputs'], fake_input), axis=0)
            batch_inputs['real_fake_label'] = train_real_fake_labels
            batch_inputs['noise'] = noise
            loss, hidden_state = model(batch_inputs, training=False)

            #if epoch % 10 == 0 and epoch != 0:
            #if epoch == 0:
            part_validation_hidden_val = np.array(hidden_state).reshape(-1, 2*sum(hidden_units_in_autoencoder_layers))
            total_abstract_validation_part_hidden_value.append(part_validation_hidden_val)
                #print(total_abstract_validation_part_hidden_value)
                #total
                #kmeans.predict()#
                #print("")
            k += 1
                #grads = tape.gradient(loss, model.trainable_variables)
            #optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_validation_loss += loss
            description = ("Average validation loss: %.2f "
                       % (total_validation_loss / (index + 1)))
            generator_tqdm.set_description(description, refresh=False)
        average_validation_loss = total_validation_loss / len(validation_batches)
        """
        #if epoch % 10 == 0 and epoch != 0:
        #print(total_abstract_validation_part_hidden_value[0])  #
        """
        concatenated_total_abstract_validation_part_hidden_value = tf.concat(total_abstract_validation_part_hidden_value,
                                                                        axis=0)
        concatenated_total_abstract_validation_part_hidden_value = concatenated_total_abstract_validation_part_hidden_value.numpy()
        km = KMeans(n_clusters=number_of_clusters)
        cluster_labels_validation = km.fit_predict(X=concatenated_total_abstract_validation_part_hidden_value)
        silhouette_score_validation = silhouette_score(concatenated_total_abstract_validation_part_hidden_value,cluster_labels_validation)

        print(silhouette_score_validation,average_validation_loss)
        if silhouette_score_validation > best_epoch_validation_silhouette_score and average_validation_loss < best_epoch_validation_loss:
            print("Model with best validation silhouette score so far: %.2f. Saving the model."
                  % (silhouette_score_validation))
            print("Model with best validation loss  so far: %.2f. Saving the model."
                  % (average_validation_loss))
            model.save_weights(os.path.join(serialization_dir, f'model.ckpt'))
            best_epoch_validation_silhouette_score = silhouette_score_validation
            best_epoch_validation_loss = average_validation_loss
        #best_epoch_validation_accuracy = validation_accuracy
        """
        with tensorboard_writer.as_default():
            tf.summary.scalar("loss/training", average_training_loss, step=epoch)
            #tf.summary.scalar("loss/validation", average_validation_loss, step=epoch)
            #tf.summary.scalar("accuracy/training", training_accuracy, step=epoch)
            #tf.summary.scalar("accuracy/validation", best_epoch_validation_silhouette_score, step=epoch)
        tensorboard_writer.flush()

    metrics = {"training_loss": float(average_training_loss),
               #"validation_loss": float(average_validation_loss),
               #"training_accuracy": float(training_accuracy),
               #"best_epoch_validation_accuracy": float(best_epoch_validation_silhouette_score),
               #"best_epoch_validation_loss": float(best_epoch_validation_loss)
               }
    #print("Best epoch validation loss: %.4f" % best_epoch_validation_loss)

    return {"model": model, "metrics": metrics}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Main')

    # Setup common parser arguments for training of either models
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('train_data_file_path', type=str, help='training data file path')
    base_parser.add_argument('validation_data_file_path', type=str, help='validation data file path')
    base_parser.add_argument('train_and_validation_data_file_path', type=str,
                             help='train and validation data file path')
    base_parser.add_argument('--load-serialization-dir', type=str,
                             help='if passed, model will be loaded from this serialization directory.')
    base_parser.add_argument('--batch-size', type=int, default=40, help='batch size')
    base_parser.add_argument('--num-epochs', type=int, default=20, help='max num epochs to train for')
    base_parser.add_argument('--suffix-name', type=str, default="",
                             help='optional model name suffix. can be used to prevent name conflict '
                                  'in experiment output serialization directory')

    subparsers = parser.add_subparsers(title='train_models', dest='model_name')

    # Setup parser arguments for main model
    main_model_subparser = subparsers.add_parser("main", description='Train Main Model',
                                                 parents=[base_parser])
    main_model_subparser.add_argument('--encoder_decoder_choice', type=str, choices=("vanilla", "gru"),
                                      help='choice of seq2vec. '
                                           'Required if load_serialization_dir not passed.')
    main_model_subparser.add_argument('--embedding-dim', type=int, help='embedding_dim '
                                                                        'Required if load_serialization_dir not passed.')
    main_model_subparser.add_argument('--num-layers', type=int, help='num layers. '
                                                                     'Required if load_serialization_dir not passed.')
    main_model_subparser.add_argument('--number-of-clusters', type=int, help='number of clusters. '
                                                                     'Required to cluster stocks.')
    main_model_subparser.add_argument('--hidden-units-layer1', type=int, help='hidden-units-layer1'
                                                                             'Required to cluster stocks.')
    main_model_subparser.add_argument('--hidden-units-layer2', type=int, help='hidden-units-layer2'
                                                                             'Required to cluster stocks.')
    main_model_subparser.add_argument('--hidden-units-layer3', type=int, help='hidden-units-layer3'
                                                                             'Required to cluster stocks.')
    """
    main_model_subparser.add_argument('--pretrained-embedding-file', type=str,
                                      help='if passed, use glove embeddings to initialize. '
                                           'the embedding matrix')
    """
    tf.random.set_seed(1337)
    np.random.seed(1337)
    random.seed(13370)
    args = parser.parse_args()

    print("Reading Training Instances")
    train_instances,max_length_train,number_of_columns_train = read_instances(args.train_data_file_path,'train')

    print("Reading validation instances.")
    validation_instances,max_length_validation,number_of_columns_validation = read_instances(args.validation_data_file_path, 'validation')

    if args.load_serialization_dir:
        print(f"Ignoring the model arguments and loading the "
              f"model from serialization_dir: {args.load_serialization_dir}")

        model = load_pretrained_model(args.load_serialization_dir)
    else:

        if args.model_name == "main":
            config = {"encoder_decoder_choice": args.encoder_decoder_choice,
                      "length_of_time_series": max(max_length_train, max_length_validation),
                      "embedding_dim": 1,
                      "num_layers": args.num_layers,
                      "number_of_clusters":args.number_of_clusters,
                      #"batch_size":args.batch_size,
                      "batch_size":len(train_instances),
                      "hidden_layer_1":args.hidden_units_layer1,
                      "hidden_layer_2": args.hidden_units_layer2,
                      "hidden_layer_3": args.hidden_units_layer3,
                      "training":True}
            model = MainEncoderDecoder(**config)
            config["type"] = "main"
        else:
            config = {"pretrained_model_path": args.base_model_dir,
                      "layer_num": args.layer_num, "classes_num": 2}
            model = ProbingEncoderDecoder(**config)
            config["type"] = "probing"

    optimizer = optimizers.Adam()
    save_serialization_dir = os.path.join("serialization_dirs", args.model_name + args.suffix_name)
    if not os.path.exists(save_serialization_dir):
        os.makedirs(save_serialization_dir)
    hidden_units_in_autoencoder_layers = [args.hidden_units_layer1,args.hidden_units_layer2,args.hidden_units_layer3]
    training_output = train(model, optimizer, train_instances,
                            validation_instances, args.num_epochs,max_length_train,max_length_validation,1,
                            len(train_instances),args.number_of_clusters,hidden_units_in_autoencoder_layers, save_serialization_dir)

    model = training_output["model"]
    metrics = training_output["metrics"]

    config_path = os.path.join(save_serialization_dir, "config.json")
    with open(config_path, "w") as file:
        json.dump(config, file)

    metrics_path = os.path.join(save_serialization_dir, "metrics.json")
    with open(metrics_path, "w") as file:
        json.dump(metrics, file)

    print(f"\nFinal model stored in serialization directory: {save_serialization_dir}")






