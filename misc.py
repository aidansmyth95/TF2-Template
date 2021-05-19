import os


def make_dir(dirname):
    """ A method to create a dir if it does not already exist """
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
