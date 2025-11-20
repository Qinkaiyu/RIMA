import zipfile
import os
import shutil
import numpy as np
import os
from PIL import Image

# Harvard-30k Data Process1
work_dir = "G:\Medical_data\oct_30k\AMD"
output_dir = "G:\merged_folder"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


merged_training = os.path.join(output_dir, "merged_training")
merged_test = os.path.join(output_dir, "merged_test")
merged_validation = os.path.join(output_dir, "merged_validation")

os.makedirs(merged_training, exist_ok=True)
os.makedirs(merged_test, exist_ok=True)
os.makedirs(merged_validation, exist_ok=True)


for filename in os.listdir(work_dir):
    print('file',filename)
    if filename.endswith('.zip') and not filename.startswith('.'):
        zip_path = os.path.join(work_dir, filename)
        print(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:

            temp_dir = os.path.join(work_dir, "temp")
            zip_ref.extractall(temp_dir)

            for subdir in ['Training', 'test', 'validation']:
                subdir_path = os.path.join(temp_dir, subdir)
                if os.path.exists(subdir_path):
                    for root, dirs, files in os.walk(subdir_path):
                        for file in files:
                            if file.endswith('.jpg'):
                                os.remove(os.path.join(root, file))

                    target_dir = {
                        'Training': merged_training,
                        'test': merged_test,
                        'validation': merged_validation
                    }

                    for item in os.listdir(subdir_path):
                        s_path = os.path.join(subdir_path, item)
                        d_path = os.path.join(target_dir[subdir], item)
                        if os.path.isdir(s_path):
                            shutil.copytree(s_path, d_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(s_path, d_path)

# Harvard-30k Data Process2
source_folder = "/path/Test"
fundus_folder = "/path/fundus"

labels_fundus_file = '/path/fundus.txt'

condition_disease_mapping = {'not.in.icd.table': 0.,
                    'no.dr.diagnosis': 0.,
                    'mild.npdr': 0.,
                    'moderate.npdr': 0.,
                    'severe.npdr': 1.,
                    'pdr': 1.}

fundus_labels = open(labels_fundus_file, 'w')



for file in os.listdir(source_folder):
    if file.endswith('.npz'):
        file_path = os.path.join(source_folder, file)
        data = np.load(file_path)
        slo_fundus = data['slo_fundus']
        fundus_image = Image.fromarray(slo_fundus)
        resized_fundus_image = fundus_image.resize((448, 448), Image.Resampling.LANCZOS)
        fundus_image_file = f'{file[:-4]}_fundus.png'
        resized_fundus_image.save(os.path.join(fundus_folder, fundus_image_file))
        condition = data['dr_subtype'].item()
        label = int(condition_disease_mapping[condition])
        fundus_labels.write(f'{fundus_image_file} {label}\n')

# Harvard-30k Data Process3
def npz_to_nii_zip(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.npz'):
            npz_path = os.path.join(input_folder, file_name)

            data = np.load(npz_path)

            if 'oct_bscans' in data:
                oct_bscans = data['oct_bscans']

                oct_bscans_nifti = np.transpose(oct_bscans, (0, 1, 2))

                nifti_img = nib.Nifti1Image(oct_bscans_nifti, affine=np.eye(4))

                nii_file_name = file_name.replace('.npz', '.nii')
                nii_path = os.path.join(output_folder, nii_file_name)
                nib.save(nifti_img, nii_path)

                zip_file_name = file_name.replace('.npz', '.zip')
                zip_path = os.path.join(output_folder, zip_file_name)
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(nii_path, arcname=nii_file_name)
                os.remove(nii_path)

input_folder = "/path/Test"
output_folder = "/path/oct"
npz_to_nii_zip(input_folder, output_folder)
