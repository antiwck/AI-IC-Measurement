from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import skimage.transform as st

my_model = YOLO("yoloModel.pt")

# Source is the source file to the video for AI model to work on
source = ('E:\OneDrive - Asia Pacific University\\University\Degree Year 3\Semester 2\Machine Vision and Intelligence\Assessment\Code\Video\MV-CA013-21UC (02E24880956)\Video_20231229114120518_v2.mp4')
cap = cv2.VideoCapture(source)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# List to store all detected dimensions
detected_dimensions = []

coin_mask_data = None
ic_mask_data = None

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
    H, W = frame.shape[:2]

    results = list(my_model.predict(frame, show=True, save=False, save_txt=False, verbose=False))
    names = my_model.names

    for r in results:
        for index, c in enumerate(r.boxes.cls):
            if names[int(c)] == 'Coin':
                # Extract the mask data
                coin_mask_data = results[0].masks.data[index].cpu().numpy()
                coin_mask_data = st.resize(coin_mask_data, (H, W), order=0, preserve_range=True, anti_aliasing=False)
            elif names[int(c)] == 'BigIC' or names[int(c)] == 'NormalIC' or names[int(c)] == 'SmallIC':
                ic_mask_data = results[0].masks.data[index].cpu().numpy()
                ic_mask_data = st.resize(ic_mask_data, (H, W), order=0, preserve_range=True, anti_aliasing=False)

    if coin_mask_data is not None and ic_mask_data is not None:
        # Find contours in the mask
        coin_contours, _ = cv2.findContours(coin_mask_data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ic_contours, _ = cv2.findContours(ic_mask_data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Coin contouring
        # Create a list to store the areas of all contours
        coin_contour = [cv2.contourArea(coin_contour) for coin_contour in coin_contours]

        # Find the index of the largest contour
        largest_coin_contour_index = np.argmax(coin_contour)

        # Get the largest contour using the index
        coin_contour = coin_contours[largest_coin_contour_index]

        # Fit a circle to the contours
        (x, y), radius = cv2.minEnclosingCircle(coin_contour)
        diameter = round(radius * 2)

        # Draw the contour on a black image (for visualization purposes)
        coin_contour_image = np.zeros_like(coin_mask_data)
        cv2.drawContours(coin_contour_image, [coin_contour], 0, 255, thickness=cv2.FILLED)


        # IC contouring
        # Create a list to store the areas of all contours
        ic_contour = [cv2.contourArea(ic_contour) for ic_contour in ic_contours]

        # Find the index of the largest contour
        largest_IC_contour_index = np.argmax(ic_contour)

        # Get the largest contour using the index
        ic_contour = ic_contours[largest_IC_contour_index]

        # Get the minimum area rectangle that bounds the contour
        rect = cv2.minAreaRect(ic_contour)

        # Draw the rectangle on the frame
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        ic_contour_image = np.zeros_like(ic_mask_data)
        cv2.drawContours(ic_contour_image, [ic_contour], 0, 255, thickness=cv2.FILLED)

        contour_image = ic_contour_image + coin_contour_image
        # cv2.imshow("Coin mask Contour", coin_contour_image)
        # cv2.imshow("IC mask Contour", ic_contour_image)

        # Calculate the width and length of the rectangle in millimeters
        width_in_pixels = np.linalg.norm(box[0] - box[1])
        length_in_pixels = np.linalg.norm(box[1] - box[2])

        # Diameter of 10 cents is 18.8mm
        width_in_mm = round(min(width_in_pixels, length_in_pixels) * 18.8 / diameter, 2)
        length_in_mm = round(max(width_in_pixels, length_in_pixels) * 18.8 / diameter, 2)

        # Display dimensions as overlay text
        cv2.putText(frame, f"Width: {width_in_mm:.2f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 0), 3)
        cv2.putText(frame, f"Length: {length_in_mm:.2f} mm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 0), 3)

        # Append the dimensions to the list
        detected_dimensions.append((width_in_mm, length_in_mm))

        # Draw the circle and rectangle on the original image
        cv2.circle(frame, (int(x), int(y)), int(radius), (85, 102, 0), 3)
        cv2.polylines(frame, [box], isClosed=True, color=(128, 64, 0), thickness=3)  # Draw rectangle

        # Show the result
        # cv2.imshow('Detected dimension', frame)

        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)

        # Make sure all images have the same height
        min_height = min(contour_image.shape[0], frame.shape[0])
        contour_image = contour_image[:min_height, :]
        frame = frame[:min_height, :]

        # Create a window with the desired size
        window_height = max(contour_image.shape[0], contour_image.shape[0])
        window_width = contour_image.shape[1] + frame.shape[1]
        window = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        # Copy images to the window
        window[:contour_image.shape[0], :contour_image.shape[1]] = contour_image
        window[:frame.shape[0], contour_image.shape[1]:] = frame

        cv2.imshow('Final Output', window)
    else:
        # Display dimensions as overlay text
        cv2.putText(frame, "Width: Width is not available", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 0), 3)
        cv2.putText(frame, "Length: Length is not available", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 0), 3)
        cv2.imshow('Final Output', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

if detected_dimensions:
    # Use Counter to find the most common dimensions
    counter_dimensions = Counter(detected_dimensions)
    most_common_dimensions = counter_dimensions.most_common(1)

    # Get the most common dimensions directly
    most_common_dimensions = most_common_dimensions[0][0]

    # Get dimensions from the most common set
    width, length = most_common_dimensions

    # Print or use the average dimensions
    print("\n\nFinal dimension:")
    print(f"Width: {width:.2f} mm")
    print(f"Length: {length:.2f} mm")
else:
    print("Dimension cannot be obtained due to lack of reference object.")

# Close all windows
cv2.destroyAllWindows()
