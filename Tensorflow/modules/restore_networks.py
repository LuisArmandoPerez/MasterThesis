import tensorflow as tf
import numpy as np
from os.path import isdir, join, split
from os import listdir
import pandas as pd
import win32api


def dataframe_graphs_in_folder(path):
    models_folder = join(path, "models")
    print(models_folder)
    experiments = [f for f in listdir(models_folder) if isdir(join(models_folder, f))]
    print(experiments)
    df_experiments = []
    df_filename = []
    for experiment in experiments:
        for file in listdir(join(models_folder, experiment, 'weights')):
            if file.endswith('.meta'):
                df_experiments.append(experiment)
                df_filename.append(file)
    df_dictionary = {'Experiment': df_experiments, 'Filenames': df_filename}
    df = pd.DataFrame(data=df_dictionary)
    print(df)
    return df


def filter_experiment_from_df(df, experiment_name):
    return df[df['Experiment'] == experiment_name]


def return_complete_path(path, df, number):
    experiment = df.iloc[number]['Experiment']
    file = df.iloc[number]['Filenames']
    complete_path = join(path, 'models', experiment, 'weights', file)
    return complete_path


def import_graph(filename):
    tf.reset_default_graph()
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(win32api.GetShortPathName(filename))
        new_saver.restore(sess, filename.replace('.meta', ''))
    print(filename + ' restored\n')
    return new_saver


def extract_variable_values(filename):
    """
    Creates a dictionary with each entry the name of a variable and its value
    :param weights_folder:
    :return:
    """
    saver = import_graph(filename)
    names = []
    values = []
    variables = tf.trainable_variables()
    with tf.Session() as sess:
        saver.restore(sess, filename.replace('.meta', ''))
        for variable in variables:
            names.append(variable.name)
            values.append(variable.eval())
    zipped_variables = zip(names, values)
    dict_variables = {key: value for (key, value) in zipped_variables}
    return dict_variables

def join_two_experiments(path, df, number_1, number_2):
    filename_1 = return_complete_path(path, df, number_1)
    filename_2 = return_complete_path(path, df, number_2)
    dict_variables_1 = extract_variable_values(filename_1)
    dict_variables_2 = extract_variable_values(filename_2)
    dict_variables_1.update(dict_variables_2)
    return dict_variables_1


