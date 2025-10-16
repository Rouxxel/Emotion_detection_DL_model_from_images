import os
import glob
import logging
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

logging.basicConfig(level=logging.INFO)

def augment_minority_class(class_path, target_count, img_size=(48, 48)):
    """Augment images in a class directory to reach target count."""
    try:
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"Class directory not found: {class_path}")
            
        datagen = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        image_paths = glob.glob(os.path.join(class_path, '*.png')) + \
                        glob.glob(os.path.join(class_path, '*.jpg')) + \
                        glob.glob(os.path.join(class_path, '*.jpeg'))
        current_count = len(image_paths)

        if current_count >= target_count:
            logging.info(f"Class '{os.path.basename(class_path)}' already has {current_count} images. Skipping augmentation.")
            return

        needed = target_count - current_count
        logging.info(f"Augmenting {needed} images for class '{os.path.basename(class_path)}'")

        if current_count == 0:
            logging.warning(f"No images found in {class_path} to augment.")
            return

        i = 0
        while i < needed:
            for img_path in image_paths:
                if i >= needed:
                    break

                try:
                    img = load_img(img_path, color_mode='grayscale', target_size=img_size)
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)  # This shape should be (1, 48, 48, 1)

                    aug_iter = datagen.flow(x, batch_size=1, save_to_dir=class_path, save_prefix='aug', save_format='png')
                    next(aug_iter)
                    i += 1
                except Exception as e:
                    logging.warning(f"Failed to augment image {img_path}: {str(e)}")
                    continue

                if i >= needed:
                    break

        logging.info(f"Finished augmenting class '{os.path.basename(class_path)}'. Total images now: {current_count + needed}")
        
    except Exception as e:
        logging.error(f"Error augmenting class {class_path}: {str(e)}")
        raise

def augment_minority_classes_in_train(train_dir, target_count=7000, img_size=(48, 48)):
    """Augment all minority classes in the training directory."""
    try:
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Train directory {train_dir} does not exist.")

        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        
        if not class_dirs:
            logging.warning(f"No class directories found in {train_dir}")
            return

        logging.info(f"Found {len(class_dirs)} classes: {class_dirs}")

        for cls in class_dirs:
            class_path = os.path.join(train_dir, cls)
            try:
                augment_minority_class(class_path, target_count, img_size)
            except Exception as e:
                logging.error(f"Failed to augment class {cls}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Error in augment_minority_classes_in_train: {str(e)}")
        raise

def main(train_dir="Emotion_detection_DL_model_from_images/dataset/train", target_count=7000):
    augment_minority_classes_in_train(train_dir, target_count)

if __name__ == "__main__":
    main()

######################################################################################
#HANDLE CLASS IMBALANCE
# train_dir = os.path.join(full_dataset_dir, "train")  #Adjust path as needed

# target_samples = 6000  #Desired number of images per class
# minority_classes = ["Anger", "Disgust", "Fear", "Happy", 
#                     "Neutral", "Sadness", "Surprise"]

# for classes in minority_classes:
#     class_folder = os.path.join(train_dir, classes)
#     if os.path.exists(class_folder):
#         augment_minority_class(class_folder, target_count=target_samples)
######################################################################################
