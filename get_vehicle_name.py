import sys

from mcap.mcap0.reader import make_reader

if __name__ == "__main__":
    for filename in sys.argv[1:]:
        with open(filename, "rb") as f:
            reader = make_reader(f)
            for metadata in reader.iter_metadata():
                if metadata.name != "scene-info":
                    continue
                print(f"{filename}: {metadata.metadata['vehicle']}")
