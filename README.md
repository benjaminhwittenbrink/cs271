## CS 271 Final Project

Basic steps to run classification model: 
1. Data: create a directory with all image files (saved as .npy files), labeled with their DHSID, and save a CSV file containing the mapping of DHSID to any outcome variables of interest 
2. Model: Run `main.py` passing in the the root directory name for the images as `data_dir`, the CSV file name as `csv_file`, the model name as `model_name` as well as any model parameters (`epoch_n`, `batch_size`, etc.).


An example bash command would be: 
`python main.py --data_dir image_folder --csv_file dhsid_df --model_name ViT --outcome MeanBMI_bin --epoch_n 10 --batch_size 64`, 
which would train a Vision Transformer classification model on the set of images in `image_folder` with outcome as the column `MeanBMI_bin` in the file `dhsids_df.csv` for 10 epochs with a batch size of 64. 

Notes: 
- To use a different file format than .npy for the images, change the `loader` passed into `SatelliteImageDataset` class (see `create_dataset` in `main.py`)
- Custom model architectures can be found in `models.py`. 
- To run a regression model, use `main_regression.py` instead. The interface should be the same

