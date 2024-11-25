import cv2

def get_mser_image(image, regions):
    # Convert the image to RGB (if it is in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a copy of the image to draw on
    image_copy = image_rgb.copy()

    # Iterate through the regions and draw the bounding boxes or contours
    for region in regions:
        # Convert the region to a contour (polygonal representation)
        region = region.reshape((-1, 1, 2))

        # Draw the contour (polygon) on the image
        cv2.polylines(
            image_copy, [region], isClosed=True, color=(255, 0, 0), thickness=2
        )

    return image_copy

def detect_mser(image):
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(image)
    return regions


def detect_kaze(image):
    kaze = cv2.KAZE_create()
    keypoints, descriptors = kaze.detectAndCompute(image, None)
    return keypoints


def detect_akaze(image):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    return keypoints
