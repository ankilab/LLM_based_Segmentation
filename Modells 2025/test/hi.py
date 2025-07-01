import os

images_root = r"D:\qy44lyfe\LLM segmentation\Data sets\Uterine myoma MRI\Dataset003_UMD\imagesTr"
for case_folder in os.listdir(images_root):
    folder_path = os.path.join(images_root, case_folder)
    if not os.path.isdir(folder_path):
        continue
    print("Folder:", case_folder)
    inner = os.listdir(folder_path)
    print("  Contains:", inner)