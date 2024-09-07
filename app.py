# Import necessary libraries
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pyttsx3  # Text-to-speech library
import torch
import cv2  # OpenCV library for camera access

# Load the pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to capture an image using the laptop camera
def capture_image():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    print("Press 'Space' to capture the image or 'Esc' to exit.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture an image.")
            break
        
        # Display the frame
        cv2.imshow('Capture Image', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == 32:  # Space key to capture
            # Save the captured frame as an image file
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)
            print("Image captured successfully.")
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    return image_path

# Function to preprocess the input image and generate a description
def generate_image_description(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Preprocess the image
    inputs = processor(images=img, return_tensors="pt")

    # Generate output from the model
    output = model.generate(**inputs)

    # Decode the generated output into a readable format
    description = processor.decode(output[0], skip_special_tokens=True)
    return description

# Function to convert text to speech
def text_to_speech(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    # Set properties for the voice (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

    # Speak the text
    engine.say(text)
    engine.runAndWait()

# Main execution
if __name__ == "__main__":
    # Capture image from the laptop camera
    image_path = capture_image()
    
    if image_path:
        # Generate the image description
        description = generate_image_description(image_path)
        print("Generated Description:", description)

        # Convert the description to speech
        text_to_speech(description)
