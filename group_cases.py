import os
import pandas as pd
import logging
from shutil import copy

# TODO:
# import sys
# group images from preprocessing to
# the separate folders based on the diagnosis


class ManageMetadata:
    """ ManageMetadata
    Sophisticated menager for all metadata related tasks.
    Validets, groups and prepares data for future training
    of deep convolutional neural networks.
    """
    HAM_DATABASE_METADATA_DIR = "datasets\\HAM10000_metadata.csv"
    PREPROCESSED_IMAGES_DIR = "preprocessed"
    DESTINATION_TYPE_DIRECORY = "grouped_images_by_type"
    DESTINATION_SEVERITY_DIRECORY = "grouped_images_by_severity"

    def __init__(self):
        self.data = pd.read_csv(self.HAM_DATABASE_METADATA_DIR)
        self.unique_values_dx = self.data.dx.unique()
        self.mutation_types = {
            "bkl": False,   # Benign keratosis-like lesions
            "nv": False,    # Melanocytic nevi
            "df": False,    # Dermatofibroma
            "vasc": False,  # Vascular lesions
            "mel": True,    # Melanoma
            "bcc": True,    # Basal cell carcinoma
            "akiec": True   # Actinic keratoses and intraepithelial carcinoma / Bowen's diesease
        }

    def extract_diagnosis(self, image_id: str):
        image_id = image_id.strip(".jpg")
        try:
            entry = self.data.query("image_id == @image_id").dx.values[0]
            return entry
        except Exception:
            logging.error(
                "Problem occured during handling of {}".format(image_id)
            )

    def group_directory(self, directory_path: str):
        preprocessed_images = os.listdir(self.PREPROCESSED_IMAGES_DIR)
        for image in preprocessed_images:
            self.group_image(image)

    def group_image(self, image_id: str):
        diagnosis = self.extract_diagnosis(image_id)
        logging.info("Diagnosis for image {}: {}".format(image_id, diagnosis))
        severity = "malignant" if self.mutation_types[diagnosis] else "benign"
        type_destination_path = extend_path(self.DESTINATION_TYPE_DIRECORY, diagnosis)

        if not os.path.isdir(type_destination_path):
            os.mkdir(type_destination_path)

        os.replace(
            extend_path(self.PREPROCESSED_IMAGES_DIR, image_id),
            extend_path(type_destination_path, image_id)
        )

        severity_destination_path = extend_path(self.DESTINATION_SEVERITY_DIRECORY, severity)

        if not os.path.isdir(severity_destination_path):
            os.mkdir(severity_destination_path)

        copy(
            extend_path(type_destination_path, image_id),
            severity_destination_path  
        )

        logging.info("Image {} successfully transfered to type and severity folders".format(image_id))


def extend_path(path, *extensions):
    for extension in extensions:
        path = "{}//{}".format(path, extension)
    return path


if __name__ == "__main__":
    mm = ManageMetadata()
    mm.group_directory("preprocessed")
