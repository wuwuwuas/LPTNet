import re
from pathlib import Path

def generate_specified_image_paths(base_folder_path, target_subfolders, output_folder_path, target_suffix):
    dir_base = Path(base_folder_path).resolve()
    dir_output = Path(output_folder_path).resolve()

    dir_output.mkdir(parents=True, exist_ok=True)

    if not target_suffix.startswith('.'):
        target_suffix = f".{target_suffix}"

    if not dir_base.is_dir():
        print("The specified base folder path does not exist or is not a directory.")
        return

    for folder_name in target_subfolders:
        sub_dir = dir_base / folder_name
        
        if not sub_dir.is_dir():
            print(f"Warning: The folder {sub_dir} does not exist or is not a directory and has been skipped.")
            continue

        match = re.search(r'\d+', folder_name)
        cam_index = match.group() if match else "0"

        output_txt_path = dir_output / f"cam{cam_index}ImageNames.txt"
        
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            for img_file in sorted(sub_dir.glob(f"*{target_suffix}")):
                txt_file.write(f"{img_file.absolute()}\n")
        
        print(f"{output_txt_path.name} has been generated.")

if __name__ == "__main__":
    base_dir = "test/ABC_TR_0.05ppp"
    specified_folders = ["Img_Cam0", "Img_Cam1", "Img_Cam2", "Img_Cam3"] 
    output_dir = base_dir
    extension = ".tif"

    generate_specified_image_paths(base_dir, specified_folders, output_dir, extension)