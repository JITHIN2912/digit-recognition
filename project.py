import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from PIL import ImageOps, Image
import cv2

# Title
st.title("🖍️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) and click **Predict**")

# Model path
MODEL_PATH = "digit_model.keras"

# Try to load model or train if failed
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.warning("⚠️ Model not found or corrupted. Retraining the model...")

    # Load data
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # Set seed
    tf.random.set_seed(42)
    np.random.seed(42)

    # Define model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=0)
    model.save(MODEL_PATH)
    st.success("✅ Model trained and saved successfully!")

# Drawing canvas
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict on drawing
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    if st.button("Predict"):
        # Preprocess image
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = ImageOps.invert(Image.fromarray(img))
        img = np.array(img).astype("float32") / 255.0

        # Optional: Center and pad digit
        try:
            inverted_img = cv2.bitwise_not((img * 255).astype(np.uint8))
            coords = cv2.findNonZero(inverted_img)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                img = img[y:y+h, x:x+w]
                img = cv2.resize(img, (20, 20))
                img = np.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=0)
            else:
                st.warning("Empty drawing detected — please draw a digit before predicting.")
        except Exception as e:
            st.warning(f"Couldn't center the digit properly: {e}")

        img = img.reshape(1, 28, 28, 1)

        # Predict
        pred = model.predict(img)
        digit = np.argmax(pred)

        # Show results
        st.image(canvas_result.image_data, caption="Your Drawing", width=150)
        st.markdown(
            f"<h2 style='text-align: center; color: green;'>✏️ Predicted Digit: {digit}</h2>", 
            unsafe_allow_html=True
        )
