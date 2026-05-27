import argparse
import sys
import os
from pathlib import Path

from foxglove.client import Client
from mcap.reader import make_reader

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
        help="data platform secret token (if not provided, FOXGLOVE_API_KEY from environment is used)",
    )
    parser.add_argument("--host", default="api.foxglove.dev", help="custom host to send data to")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="actually perform the upload (runs as a dry-run dry run by default)",
    )
    parser.add_argument(
        "--project",
        "-p",
        help="Foxglove Data Platform project ID (required if committing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite any pre-existing recordings on Foxglove with matching filename (deletes the old recordings before starting the new upload)",
    )
    args = parser.parse_args()
    if args.commit and args.project is None:
        parser.error("--project/-p is required when committing uploads")
    if args.token is None:
        token = os.environ.get("FOXGLOVE_API_KEY")
        if token is None:
            print("FOXGLOVE_API_KEY not in environment", file=sys.stderr)
            return 1
        args.token = token

    client = Client(token=args.token, host=args.host)

    filepaths = []
    for name in args.files:
        path = Path(name)
        if path.is_dir():
            filepaths.extend(path.glob("*.mcap"))
        elif path.is_file():
            filepaths.append(path)
        else:
            raise RuntimeError(f"path does not exist: {name}")

    # Sort filepaths by scene number (e.g. nuscenes-scene-0032.mcap) so we upload in scene order
    filepaths.sort(key=lambda x: x.name)

    for filepath in filepaths:
        filename = filepath.name
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        
        with open(filepath, "rb") as f:
            reader = make_reader(f)
            scene_info = next(metadata for metadata in reader.iter_metadata() if metadata.name == "scene-info")
            device_name = scene_info.metadata["vehicle"]

        if not args.commit:
            print(f"[DRY-RUN] Would upload {filename} ({file_size_mb:.2f} MB) to device: {device_name} (project ID: {args.project})")
            continue

        print(f"checking for previous imports of {filename} ...")
        previous_uploads = client.get_recordings(path=filename, project_id=args.project)
        if previous_uploads:
            if not args.overwrite:
                print(f"Skipping upload: {filename} already exists on Foxglove. Use --overwrite to overwrite.")
                continue
            
            print(f"removing {len(previous_uploads)} previously-uploaded instance(s) of {filename} before starting new upload...")
            for upload in previous_uploads:
                client.delete_recording(
                    recording_id=upload["id"],
                )

        with open(filepath, "rb") as f:
            print(f"uploading {filename} with device name {device_name} into project {args.project} ...")

            with UploadProgressBar(unit="B", unit_scale=True) as progress_bar:
                client.upload_data(
                    device_name=device_name,
                    project_id=args.project,
                    filename=filename,
                    data=f,
                    callback=progress_bar.update_to,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
