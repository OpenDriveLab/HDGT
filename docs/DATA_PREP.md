# Prepare Dataset

## Download the Dataset
You should prepare ~2TB disk space in total to run HDGT in Waymo. First, download the Waymo Open Motion Dataset from their official website. The folder structure should be:

    HDGT/
    ├── dataset
        ├── waymo
            ├── training
            ├── validation
            ├── ...
    |-preprocess
        ├── preprocess_waymo.py
        ├── ...       
Note that you could learn about the linux command *ln -s* to avoid copy the huge Waymo dataset around.


## Preprocess the Dataset
Then, we preprocess these tfrecords so that each scene (91 steps) is saved as one pickle file. To parallelly preprocess files, we split all train tfrecords into 12 parts by index (the validation is just 1 part) and thus we could run 13 process to conduct preprocessing:
```shell
## In the HDGT/preprocess directory
preprocess_data_folder_name=hdgt_waymo
for i in {0..12}
do
    nohup python preprocess_waymo.py $i $preprocess_data_folder_name 2>&1 > preprocess_$i.log &
done
```
It could take ~12 hours and ~700 GB disk space.