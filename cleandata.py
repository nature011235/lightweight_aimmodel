import glob
import os

DATASET_ROOT = "dataset/fps.v2i.yolov11"
SUB_FOLDERS = ["train", "valid", "test"]


def clean_dataset():
    total_deleted = 0
    total_kept = 0

    for sub in SUB_FOLDERS:
        label_dir = os.path.join(DATASET_ROOT, sub, "labels")
        image_dir = os.path.join(DATASET_ROOT, sub, "images")

        label_files = glob.glob(os.path.join(label_dir, "*.txt"))

        for label_path in label_files:
            file_name = os.path.basename(label_path)
            image_name = file_name.replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, image_name)

            if not os.path.exists(image_path):
                image_name = file_name.replace(".txt", ".png")
                image_path = os.path.join(image_dir, image_name)
                if not os.path.exists(image_path):
                    print(f"can't find: {label_path}，delete label")
                    os.remove(label_path)
                    continue

            has_head = False

            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id is not None:  # has class label
                        has_head = True
                        break

            if has_head:
                total_kept += 1
            else:
                # dont't have label delete it
                os.remove(label_path)
                os.remove(image_path)
                total_deleted += 1
                print(f"delete: {file_name}")

    print(f"keep: {total_kept}")
    print(f"delete: {total_deleted}")


if __name__ == "__main__":
    clean_dataset()
