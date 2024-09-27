import os
import re
import sys
import shutil
import pandas as pd
import itk
import SimpleITK as sitk

def natural_sort_key(string):
    """Key function for natural sorting"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def convert_dicom_series_to_nrrd(dicom_folder, voxel_spacing):
    print(f'Reading the DICOM files of {dicom_folder}')
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()
    
    # Set the voxel spacing
    image.SetSpacing(voxel_spacing)
    
    # Save the NRRD file
    nrrd_file = os.path.join(dicom_folder, os.path.splitext(os.path.basename(dicom_folder))[0] + '.nrrd')
    print(f'Writing the NRRD file {nrrd_file}')
    sitk.WriteImage(image, nrrd_file)

def dicom_to_nrrd_pipeline(main_folder, csv_file):
    voxel_spacing_df = pd.read_csv(csv_file)
    subfolders = sorted([subfolder for subfolder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, subfolder))], key=natural_sort_key)
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        
        # Extract study number and find voxel spacing
        study_number = int(re.sub(r"\D", "", subfolder))
        voxel_spacing_row = voxel_spacing_df[voxel_spacing_df['Breast n.'] == study_number]

        if not voxel_spacing_row.empty:
            coronal_pixel_pitch = voxel_spacing_row['Coronal pixel pitch (mm)'].values[0]
            slice_thickness = voxel_spacing_row['slicethickness (mm)'].values[0]
            voxel_spacing = (coronal_pixel_pitch, coronal_pixel_pitch, slice_thickness)
            convert_dicom_series_to_nrrd(subfolder_path, voxel_spacing)
        else:
            print(f"No voxel spacing information found for study {study_number}. Skipping.")

def move_nrrd_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subdir)
        if os.path.isdir(subfolder_path):
            nrrd_file = os.path.join(subfolder_path, f'{subdir}.nrrd')
            if os.path.exists(nrrd_file):
                output_subfolder = os.path.join(output_folder, subdir)
                os.makedirs(output_subfolder, exist_ok=True)
                shutil.move(nrrd_file, os.path.join(output_subfolder, f'{subdir}.nrrd'))
    print("NRRD file moving complete.")

def resample_to_isotropic(main_folder, isotropic_spacing):
    subfolders = sorted([subfolder for subfolder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, subfolder))], key=natural_sort_key)

    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        PixelType = itk.US
        Dimension = 3
        ImageType = itk.Image[PixelType, Dimension]
        input_image_path = os.path.join(subfolder_path, f'{subfolder}.nrrd')
        input_image = itk.imread(input_image_path, PixelType)
        
        desired_spacing = (isotropic_spacing, isotropic_spacing, isotropic_spacing)
        origin = [0.0, 0.0, 0.0]
        identity_matrix = itk.Matrix[itk.D, 3, 3]()
        identity_matrix.SetIdentity()
        original_spacing = input_image.GetSpacing()
        original_size = input_image.GetLargestPossibleRegion().GetSize()

        new_size = [int(round(original_size[i] * (original_spacing[i] / desired_spacing[i]))) for i in range(3)]
        ResampleImageFilterType = itk.ResampleImageFilter[ImageType, ImageType]
        resample_filter = ResampleImageFilterType.New()
        resample_filter.SetInput(input_image)
        resample_filter.SetOutputSpacing(desired_spacing)
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputOrigin(origin)
        resample_filter.SetOutputDirection(identity_matrix)
        interpolator = itk.NearestNeighborInterpolateImageFunction[ImageType, itk.D].New()
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetDefaultPixelValue(0)
        resample_filter.Update()

        resampled_file = os.path.join(subfolder_path, f'{subfolder}_resampled.nrrd')
        print(f'Writing the resampled NRRD {resampled_file}')
        itk.imwrite(resample_filter.GetOutput(), resampled_file, compression=True)
        
        try:
            os.remove(input_image_path)
            print(f'Removed input image {input_image_path}')
        except OSError as e:
            print(f"Error: {e.strerror} - {input_image_path}")

def full_pipeline(dicom_folder, csv_file, output_folder, isotropic_spacing):
    dicom_to_nrrd_pipeline(dicom_folder, csv_file)
    move_nrrd_files(dicom_folder, output_folder)
    resample_to_isotropic(output_folder, isotropic_spacing)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python combined_pipeline_uncompressed.py <dicom_folder> <csv_file> <output_folder> <isotropic_spacing>")
        sys.exit(1)

    dicom_folder = sys.argv[1]
    csv_file = sys.argv[2]
    output_folder = sys.argv[3]
    isotropic_spacing = float(sys.argv[4])

    full_pipeline(dicom_folder, csv_file, output_folder, isotropic_spacing)
