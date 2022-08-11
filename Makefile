

SCRATCH_DISK_DEVICE = /dev/nvme0n1
SCRATCH_DISK_MOUNTPOINT = /media/scratch
ZIP_DOWNLOAD_PATH = $(SCRATCH_DISK_MOUNTPOINT)/zip
DATASET_PATH = $(SCRATCH_DISK_MOUNTPOINT)/dataset
OUTPUT_PATH = $(SCRATCH_DISK_MOUNTPOINT)/output

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

.scratch-disk-format.stamp:
	@echo Formatting the scratch disk
	sudo mkfs.ext4 -F $(SCRATCH_DISK_DEVICE)
	touch $@

.mount-scratch-disk.stamp: .scratch-disk-format.stamp
	@echo mounting the scratch disk
	sudo mkdir -p $(SCRATCH_DISK_MOUNTPOINT)
	sudo mount $(SCRATCH_DISK_DEVICE) $(SCRATCH_DISK_MOUNTPOINT)
	sudo chmod a+rwc $(SCRATCH_DISK_MOUNTPOINT)
	touch $@

.download-aux-inputs.stamp: .mount-scratch-disk.stamp
	@echo downloading CAN and map data
	mkdir -p $(ZIP_DOWNLOAD_PATH) $(DATASET_PATH)
	for zip in $(AUX_INPUTS); do \
		gsutil cp "$(INPUT_BUCKET)/$${zip}" "$(ZIP_DOWNLOAD_PATH)/$${zip}"; \
		unzip "$(ZIP_DOWNLOAD_PATH)/$${zip}" -d $(DATASET_PATH); \
	done
	touch $@

.download-mini-dataset.stamp: .download-aux-inputs.stamp
	@echo "downloading mini dataset"
	mkdir -p $(DATASET_PATH)
	for tgz in $(MINI_DATASET_INPUTS); do \
		gsutil cp "$(INPUT_BUCKET)/$${tgz}" - | tar -xC $(DATASET_PATH); \
	done
	touch $@

.PHONY: download-mini-dataset
download-mini-dataset: .download-mini-dataset.stamp

.download-full-dataset: .mount-scratch-disk.stamp
	@echo "downloading full dataset"
	mkdir -p $(DATASET_PATH)
	for tgz in $(FULL_DATASET_INPUTS); do \
		gsutil cp "$(INPUT_BUCKET)/$${tgz}" - | tar -xC $(DATASET_PATH); \
	done
	touch $@

.PHONY: download-full-dataset
download-full-dataset: .download-full-dataset.stamp .download-aux-inputs.stamp

.PHONY: convert-mini-dataset
convert-mini-dataset: converter-image download-mini-dataset
	docker run -t --rm \
		-v $(DATASET_PATH):/data \
		-v $(OUTPUT_PATH):/output \
		$(CONVERTER_IMAGE_NAME) \
		python3 to_mcap.py \
		--dataset-name "v1.0-mini" \
		--data-dir /data \
		--output-dir /output

.PHONY: convert-full-dataset
convert-full-dataset: converter-image download-full-dataset
	docker run -t --rm \
		-v $(DATASET_PATH):/data \
		-v $(OUTPUT_PATH):/output \
		$(CONVERTER_IMAGE_NAME) \
		python3 to_mcap.py \
		--dataset-name "v1.0-trainval" \
		--data-dir /data \
		--output-dir /output

.PHONY: upload-mcaps
upload-mcaps: converter-image
	docker run -t --rm \
		-v $(OUTPUT_PATH):/output \
		-e FOXGLOVE_CONSOLE_TOKEN \
		$(CONVERTER_IMAGE_NAME) \
		python3 upload_mcap.py --skip-existing /output

.PHONY: upload-events
upload-events: converter-image
	docker run -t --rm \
		-v $(OUTPUT_PATH):/output \
		-e FOXGLOVE_CONSOLE_TOKEN \
		$(CONVERTER_IMAGE_NAME) \
		python3 upload_events.py /output
	
.PHONY: converter-image
converter-image:
	docker build -t $(CONVERTER_IMAGE_NAME) .

.PHONY: clean
clean:
	rm -rf $(ZIP_DOWNLOAD_PATH) $(DATASET_PATH) $(OUTPUT_PATH)
	rm -f .*.stamp