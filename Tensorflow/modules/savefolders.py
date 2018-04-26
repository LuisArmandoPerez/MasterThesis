import time
import win32api
from os.path import join, exists, isfile
from os import makedirs


class SaveFolders():
    def __init__(self, target_folder: str, experiment_name: str, extra_comments: str):
        self.target_folder = target_folder
        self.experiment_name = experiment_name
        self.extra_comments = extra_comments
        self.timestr = time.strftime("%Y-%m-%d-%H-%M_")
        self.root_folder = self._create_root_folder()
        self.subfolders = self._create_folders()

    def _create_root_folder(self):
        root_folder = join(self.target_folder, 'models', self.experiment_name)
        if not exists(root_folder):
            makedirs(root_folder)
        return root_folder

    def _create_folders(self):
        subfolders = {'weights': join(self.root_folder, 'weights'),
                      'tensorboard': join(self.root_folder, 'tensorboard'),
                      'images': join(self.root_folder, 'images')}
        root_folder = self._create_root_folder()
        for subfolder in subfolders:
            if not exists(subfolders[subfolder]):
                makedirs(subfolders[subfolder])
        return subfolders

    def get_weight_file(self):
        weight_file = join(self.subfolders['weights'],
                           self.experiment_name + self.timestr + self.extra_comments + '.cpkt')
        return weight_file

    def get_logdir_tensorboard(self):
        logdir = join(win32api.GetShortPathName(self.subfolders['tensorboard']), self.timestr, self.extra_comments)
        print("Tensorboard address to copy paste:\n")
        print(win32api.GetShortPathName(self.subfolders['tensorboard']))
        return logdir
