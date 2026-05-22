import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import shutil
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# =========================
# Merge Training + Testing
# =========================

train_dir = "Training"
test_dir = "Testing"
merged_dir = "BraTS_dataset"

CATEGORIES = ["glioma", "meningioma", "pituitary", "notumor"]

# Create merged dataset folders
os.makedirs(merged_dir, exist_ok=True)

for category in CATEGORIES:
    os.makedirs(os.path.join(merged_dir, category), exist_ok=True)


def copy_images(source_folder):

    for category in CATEGORIES:

        src_folder = os.path.join(source_folder, category)
        dest_folder = os.path.join(merged_dir, category)

        if os.path.exists(src_folder):

            for file in os.listdir(src_folder):

                src_path = os.path.join(src_folder, file)
                dest_path = os.path.join(dest_folder, file)

                try:
                    shutil.copy2(src_path, dest_path)

                except Exception:
                    pass


print("📥 Merging dataset folders...")

copy_images(train_dir)
copy_images(test_dir)

print("✅ Dataset merged successfully!")

# =========================
# Load Images
# =========================

IMG_SIZE = 100

print("🧠 Loading images...")

data = []
labels = []

for category in CATEGORIES:

    path = os.path.join(merged_dir, category)
    class_num = CATEGORIES.index(category)

    for img_name in os.listdir(path):

        try:
            img_path = os.path.join(path, img_name)

            # Convert to grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:

                # Resize image
                resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                data.append(resized)
                labels.append(class_num)

        except Exception as e:
            print(f"Error loading image {img_name}: {e}")

# =========================
# Prepare Dataset
# =========================

X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

y = to_categorical(labels, num_classes=len(CATEGORIES))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Build CNN Model
# =========================

print("🛠 Building CNN model...")

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(IMG_SIZE, IMG_SIZE, 1)
    ),

    MaxPooling2D(2, 2),

    Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(len(CATEGORIES), activation='softmax')

])

# =========================
# Compile Model
# =========================

model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train Model
# =========================

print("🚀 Training model...")

model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# =========================
# Evaluate Model
# =========================

loss, accuracy = model.evaluate(X_test, y_test)

print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# =========================
# Save Model
# =========================

model.save("brain_tumor_model.h5")

print("💾 Model saved as brain_tumor_model.h5")