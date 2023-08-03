import os, sys, requests
import gdown
import shutil

PATH = "DATASETS"



if __name__ == "__main__":
    isExist = os.path.exists(PATH)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(PATH)

    url = 'https://drive.google.com/file/d/18bAvwpkUcefbiw8urCHSObyBt7StBIPH/view?usp=drive_link'

    output_path = PATH+'/daaset.zip'
    gdown.download(url, output_path, quiet=False,fuzzy=True)

    shutil.unpack_archive(output_path, PATH)

