SCRATCH_DISK_MOUNTPOINT = /media/scratch
ZIP_DOWNLOAD_PATH = $(SCRATCH_DISK_MOUNTPOINT)/zip
DATASET_PATH = $(SCRATCH_DISK_MOUNTPOINT)/dataset
OUTPUT_PATH = $(SCRATCH_DISK_MOUNTPOINT)/output
DATA_PLATFORM_HOST ?= api.foxglove.dev

INPUT_BUCKET = gs://nuscenes-all/full/s3.amazonaws.com/data.nuscenes.org/public/v1.0

AUX_INPUTS = can_bus.zip nuScenes-map-expansion-v1.3.zip

MINI_DATASET_INPUTS = v1.0-mini.tgz

FULL_DATASET_INPUTS = \
	v1.0-trainval01_blobs.tgz \
	v1.0-trainval02_blobs.tgz \
	v1.0-trainval03_blobs.tgz \
	v1.0-trainval04_blobs.tgz \
	v1.0-trainval05_blobs.tgz \
	v1.0-trainval06_blobs.tgz \
	v1.0-trainval07_blobs.tgz \
	v1.0-trainval08_blobs.tgz \
	v1.0-trainval09_blobs.tgz \
	v1.0-trainval10_blobs.tgz \
	v1.0-trainval_meta.tgz

CONVERTER_IMAGE_NAME = "mcap_converter"

.apt-install.stamp:
	sudo apt-get install -y --no-install-recommends \
		unzip build-essential docker.io e2fsprogs tmux pv
	touch $@

.PHONY: apt-install
apt-install: .apt-install.stamp

.PHONY: scratch-disk
scratch-disk: apt-install
	sudo python3 ready_scratch_disk.py --commit

.download-aux-inputs.stamp: scratch-disk
	@echo downloading CAN and map data
	mkdir -p $(ZIP_DOWNLOAD_PATH) $(DATASET_PATH)
	gsutil cp "$(INPUT_BUCKET)/can_bus.zip" "$(ZIP_DOWNLOAD_PATH)/can_bus.zip"
	unzip "$(ZIP_DOWNLOAD_PATH)/can_bus.zip" -d $(DATASET_PATH)
	gsutil cp "$(INPUT_BUCKET)/nuScenes-map-expansion-v1.3.zip" "$(ZIP_DOWNLOAD_PATH)/nuScenes-map-expansion-v1.3.zip"
	unzip "$(ZIP_DOWNLOAD_PATH)/nuScenes-map-expansion-v1.3.zip" -d "$(DATASET_PATH)/maps"
	touch $@

.download-mini-dataset.stamp: .download-aux-inputs.stamp
	@echo downloading mini dataset
	mkdir -p $(DATASET_PATH)
	for tgz in $(MINI_DATASET_INPUTS); do \
		gsutil cp "$(INPUT_BUCKET)/$${tgz}" - | pv | tar -xzC $(DATASET_PATH); \
	done
	touch $@

.PHONY: download-mini-dataset
download-mini-dataset: .download-mini-dataset.stamp

.download-full-dataset.stamp: scratch-disk
	@echo "downloading full dataset"
	mkdir -p $(DATASET_PATH)
	for tgz in $(FULL_DATASET_INPUTS); do \
		gsutil cp "$(INPUT_BUCKET)/$${tgz}" - | pv | tar -xzC $(DATASET_PATH); \
	done
	touch $@

.PHONY: download-full-dataset
download-full-dataset: .download-full-dataset.stamp .download-aux-inputs.stamp

.PHONY: convert-mini-dataset
convert-mini-dataset: converter-image download-mini-dataset
	docker run -t --rm \
		--user $(id -u):$(id -g) \
		-v $(DATASET_PATH):/data \
		-v $(OUTPUT_PATH):/output \
		$(CONVERTER_IMAGE_NAME) \
		python3 convert_to_mcap.py \
		--dataset-name "v1.0-mini" \
		--data-dir /data \
		--output-dir /output

.PHONY: convert-full-dataset
convert-full-dataset: converter-image download-full-dataset
	docker run -t --rm \
		--user $(id -u):$(id -g) \
		-v $(DATASET_PATH):/data \
		-v $(OUTPUT_PATH):/output \
		$(CONVERTER_IMAGE_NAME) \
		python3 convert_to_mcap.py \
		--dataset-name "v1.0-trainval" \
		--data-dir /data \
		--output-dir /output

.PHONY: upload-mcaps
upload-mcaps: converter-image
	docker run -t --rm \
		-v $(OUTPUT_PATH):/output \
		-e FOXGLOVE_DATA_PLATOFRM_TOKEN \
		$(CONVERTER_IMAGE_NAME) \
		python3 upload_mcap.py /output --host $(DATA_PLATFORM_HOST)


.PHONY: upload-events
upload-events: converter-image
	docker run -t --rm \
		-v $(OUTPUT_PATH):/output \
		-e FOXGLOVE_DATA_PLATFORM_TOKEN \
		$(CONVERTER_IMAGE_NAME) \
		python3 upload_events.py /output --host $(DATA_PLATFORM_HOST)


.PHONY: delete-all-events
delete-all-events: converter-image
	docker run -t --rm \
		-e FOXGLOVE_DATA_PLATFORM_TOKEN \
		$(CONVERTER_IMAGE_NAME) \
		python3 delete_all_events.py --host $(DATA_PLATFORM_HOST)


.PHONY: converter-image
converter-image:
	docker build -t $(CONVERTER_IMAGE_NAME) ..

.PHONY: clean
clean:
	rm -rf $(ZIP_DOWNLOAD_PATH) $(DATASET_PATH) $(OUTPUT_PATH)
	rm -f .*.stamp
