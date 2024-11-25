import cv2


def sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def surf_features(image, hessian_threshold=400):
    # does not work due to patent!!
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    keypoints, descriptors = surf.detectAndCompute(image, None)
    return keypoints, descriptors


def orb_features(image, max_features=500):
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def fast_brief_features(image, max_features=500):
    fast = cv2.FastFeatureDetector_create()  # Create FAST keypoint detector
    brief = (
        cv2.xfeatures2d.BriefDescriptorExtractor_create()
    )  # Create BRIEF descriptor extractor

    # Detect keypoints using FAST
    keypoints = fast.detect(image, None)

    # Compute BRIEF descriptors for the keypoints
    keypoints, descriptors = brief.compute(image, keypoints)

    return keypoints, descriptors
