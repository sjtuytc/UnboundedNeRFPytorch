# How to generate pytorch_block_nerf_dataset

1. download the Waymo Block dataset via the following command:

	```bash
	pip install gdown # download google drive download.
	cd data
	gdown --id 1iRqO4-GMqZAYFNvHLlBfjTcXY-l3qMN5 --no-cache 
	unzip v1.0.zip
	cd ../
	```
   The Google cloud may [limit the download speed in this operation](https://stackoverflow.com/questions/16856102/google-drive-limit-number-of-download). You can instead:
   (1) Downloading in your browser by clicking [this link](https://drive.google.com/file/d/1iRqO4-GMqZAYFNvHLlBfjTcXY-l3qMN5/view). (2) Alternatively, you can directly download from the official [Waymo](https://waymo.com/research/block-nerf/licensing/) website. However, this download may needs the sudo access to install the [gsutil tool](https://cloud.google.com/storage/docs/gsutil_install#deb) (if you don't have sudo access, you can download from your local laptop and then transport it to your server). The reference script is as follows:

	```bash
	# install gsutil tool
	sudo apt-get install apt-transport-https ca-certificates gnupg # needs sudo access
	echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
	curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
	sudo apt-get update && sudo apt-get install google-cloud-cli # needs sudo access
	gcloud init # login your google account then
	cd data
	gsutil -m cp -r \
	  "gs://waymo-block-nerf/v1.0" \
	  .
	unzip v1.0.zip
	cd ..
	```
   You may otherwise symbol link the downloaded dataset ("v1.0") under the "data" folder. The Waymo official files (e.g., v1.0/v1.0_waymo_block_nerf_mission_bay_train.tfrecord-00000-of-01063) would be put under the data folder. 

2. Transfer the original data in TF to the pytorch format via the following command:

   ```bash
   python data_preprocess/fetch_data_from_tf_record.py
   ```

3. Split the waymo dataset into blocks and extract corresponding information.

	```bash
	python data_preprocess/split_block.py
	```

Now you have finished the waymo data preprocess procedure and you can start training.