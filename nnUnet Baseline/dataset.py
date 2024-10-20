import os
import shutil
import json
import numpy as np
import SimpleITK as sitk
from PIL import Image


def convert_to_nifti(image_path, output_path):
    """Convert PNG/JPG images to NIfTI format and save them."""
    img = Image.open(image_path).convert("L")  # Ensure grayscale
    img_array = np.array(img)

    # Convert the numpy array to SimpleITK image and save as .nii.gz
    sitk_img = sitk.GetImageFromArray(img_array)
    sitk.WriteImage(sitk_img, output_path)


def create_nnUNet_structure(base_dir, dataset_name):
    """Create nnU-Net structure for a dataset."""
    dataset_folder = os.path.join(base_dir, dataset_name)
    imagesTr_folder = os.path.join(dataset_folder, 'imagesTr')
    labelsTr_folder = os.path.join(dataset_folder, 'labelsTr')

    if not os.path.exists(imagesTr_folder):
        os.makedirs(imagesTr_folder)
    if not os.path.exists(labelsTr_folder):
        os.makedirs(labelsTr_folder)

    return imagesTr_folder, labelsTr_folder


def process_bagls_dataset(images_folder, output_imagesTr, output_labelsTr, prefix=""):
    """Process BAGLS dataset to separate images and masks."""
    image_files = sorted(os.listdir(images_folder))

    for i, img_file in enumerate(image_files):
        if img_file.endswith(('.png', '.jpg')):
            base_name = os.path.splitext(img_file)[0]

            # Split the image and mask based on the naming convention
            if "_seg" in img_file:
                # It's a mask file
                mask_output_name = f"{prefix}_patient_{i:04d}.nii.gz"
                mask_output_path = os.path.join(output_labelsTr, mask_output_name)
                convert_to_nifti(os.path.join(images_folder, img_file), mask_output_path)
            else:
                # It's an image file
                image_output_name = f"{prefix}_patient_{i:04d}_0000.nii.gz"
                image_output_path = os.path.join(output_imagesTr, image_output_name)
                convert_to_nifti(os.path.join(images_folder, img_file), image_output_path)


def process_dataset(images_folder, masks_folder, output_imagesTr, output_labelsTr, dataset_id, prefix=""):
    """Convert images and masks to nnU-Net format and move them to output folders."""
    image_files = sorted(os.listdir(images_folder))

    for i, img_file in enumerate(image_files):
        if img_file.endswith(('.png', '.jpg')):
            base_name = os.path.splitext(img_file)[0]

            # Define new nnU-Net naming: "prefix_patient_i_0000.nii.gz" for images
            image_output_name = f"{prefix}_patient_{i:04d}_0000.nii.gz"
            mask_output_name = f"{prefix}_patient_{i:04d}.nii.gz"

            image_output_path = os.path.join(output_imagesTr, image_output_name)

            # Select mask file name based on dataset_id
            if dataset_id == "001":  # BAGLS dataset (handled separately in process_bagls_dataset)
                continue
            elif dataset_id == "002":  # Brain Meningioma dataset
                mask_file = os.path.join(masks_folder, base_name + "_m.jpg")
            elif dataset_id == "003":  # Swallowing dataset
                mask_file = os.path.join(masks_folder, img_file)  # Mask name is same as image name

            mask_output_path = os.path.join(output_labelsTr, mask_output_name)

            # Convert and save images and masks
            convert_to_nifti(os.path.join(images_folder, img_file), image_output_path)
            convert_to_nifti(mask_file, mask_output_path)


def create_dataset_json(dataset_dir, dataset_name, task_id, num_cases, modalities=["Grayscale"]):
    """Generate the dataset.json file for nnU-Net."""
    dataset_json = {
        "name": dataset_name,
        "description": f"Segmentation dataset for {dataset_name}",
        "reference": "",
        "licence": "",
        "release": "0.0",
        "tensorImageSize": "4D",
        "modality": {"0": modalities[0]},  # Assuming all images are grayscale
        "labels": {"background": 0, "foreground": 1},
        "channel_names": {
            "0": "grayscale"
        },
        "numTraining": num_cases,
        "numTest": 0,
        "file_ending": ".nii.gz",
        "training": [{"image": f"./imagesTr/{dataset_name}_patient_{i:04d}_0000.nii.gz",
                      "label": f"./labelsTr/{dataset_name}_patient_{i:04d}.nii.gz"}
                     for i in range(num_cases)],
        "test": []
    }

    json_file_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(dataset_json, json_file, indent=4)


def prepare_all_datasets(base_dir, nnUNet_base):
    """Prepare all datasets into the nnUNet structure."""
    datasets_info = {
        "BAGLS": {
            "images": os.path.join(base_dir, "BAGLS", "subset"),
            "task_id": "001",
            "name": "BAGLS"
        },
        "Brain_Meningioma": {
            "images": os.path.join(base_dir, "Brain Meningioma", "images"),
            "masks": os.path.join(base_dir, "Brain Meningioma", "Masks"),
            "task_id": "002",
            "name": "BrainMeningioma"
        },
        "Swallowing": {
            "images": os.path.join(base_dir, "Swallowing", "images"),
            "masks": os.path.join(base_dir, "Swallowing", "masks"),
            "task_id": "003",
            "name": "Swallowing"
        }
    }

    for dataset_key, dataset_info in datasets_info.items():
        # Create nnUNet folders
        dataset_name = dataset_info["name"]
        images_folder = dataset_info["images"]
        task_id = dataset_info["task_id"]

        if task_id == "001":  # Special handling for BAGLS
            output_imagesTr, output_labelsTr = create_nnUNet_structure(nnUNet_base, f"Dataset{task_id}_{dataset_name}")
            process_bagls_dataset(images_folder, output_imagesTr, output_labelsTr, prefix=dataset_name)
        else:
            masks_folder = dataset_info["masks"]
            output_imagesTr, output_labelsTr = create_nnUNet_structure(nnUNet_base, f"Dataset{task_id}_{dataset_name}")
            process_dataset(images_folder, masks_folder, output_imagesTr, output_labelsTr, task_id, prefix=dataset_name)

        # Count number of cases
        num_cases = len(os.listdir(output_imagesTr))

        # Create the dataset.json
        create_dataset_json(os.path.join(nnUNet_base, f"Dataset{task_id}_{dataset_name}"), dataset_name, task_id, num_cases)


if __name__ == "__main__":
    base_data_dir = r"D:\qy44lyfe\LLM segmentation\Data sets"
    nnUNet_base_dir = os.path.join(base_data_dir, "nnUnet_raw")

    # Prepare all datasets
    prepare_all_datasets(base_data_dir, nnUNet_base_dir)
