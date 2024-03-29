import argparse
import sys
import os
from pathlib import Path

from foxglove_data_platform.client import Client
from mcap.mcap0.reader import make_reader
from device_name import make_device_name

from tqdm import tqdm


class UploadProgressBar(tqdm):
    def update_to(self, size, progress):
        self.total = size
        self.update(progress - self.n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="MCAP files to annotate")
    parser.add_argument(
        "--token",
        "-t",
        help="data platform secret token (if not provided, FOXGLOVE_DATA_PLATFORM_TOKEN from environment is used)",
    )
    parser.add_argument("--host", default="api.foxglove.dev", help="custom host to send data to")
    args = parser.parse_args()
    if args.token is None:
        token = os.environ.get("FOXGLOVE_DATA_PLATFORM_TOKEN")
        if token is None:
            print("FOXGLOVE_DATA_PLATFORM_TOKEN not in environment", file=sys.stderr)
            return 1
        args.token = token

    client = Client(token=args.token, host=args.host)
    device_ids = {resp["name"]: resp["id"] for resp in client.get_devices()}

    filepaths = []
    for name in args.files:
        path = Path(name)
        if path.is_dir():
            filepaths.extend(path.glob("*.mcap"))
        elif path.is_file():
            filepaths.append(path)
        else:
            raise RuntimeError(f"path does not exist: {name}")

    for filepath in filepaths:
        filename = filepath.name
        print(f"checking for previous imports of {filename} ...")
        previous_uploads = client.get_imports(filename=filename)
        with open(filepath, "rb") as f:
            reader = make_reader(f)
            scene_info = next(metadata for metadata in reader.iter_metadata() if metadata.name == "scene-info")
            device_name = make_device_name(scene_info.metadata)
            device_id = device_ids.get(device_name)
            if device_id is None:
                client.create_device(name=device_name)
                device_id = device_ids.get(device_name)
                device_ids[device_name] = device_id

            f.seek(0)
            print(f"uploading {filename} with device name {device_name} ...")

            with UploadProgressBar(unit="B", unit_scale=True) as progress_bar:
                client.upload_data(
                    device_id=device_id,
                    filename=filename,
                    data=f,
                    callback=progress_bar.update_to,
                )

        if previous_uploads:
            print(f"removing {len(previous_uploads)} previously-uploaded instance(s) of {filename}")
        for upload in previous_uploads:
            client.delete_import(
                device_id=upload["device_id"],
                import_id=upload["import_id"],
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
