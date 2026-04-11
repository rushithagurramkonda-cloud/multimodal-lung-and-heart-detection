import h5py

file_path = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\heart_image\raw_data\patient101_frame01.h5"

with h5py.File(file_path, "r") as hf:
    print("Keys inside the file:")
    print(list(hf.keys()))