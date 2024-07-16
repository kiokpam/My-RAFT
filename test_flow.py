import cv2
import numpy as np

# Load two consecutive frames
frame1 = cv2.imread('demo-frames/V_1_frame0.jpg')
frame2 = cv2.imread('demo-frames/V_1_frame1.jpg')
# Assuming frame1 and frame2 are your consecutive frames from a video source
# Convert images to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Points to track
p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

# Calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

# Select good points
good_new = p1[st == 1]
good_old = p0[st == 1]

# Create a mask image for drawing purposes
mask = np.zeros_like(frame1)

# Draw the tracks
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
    frame2 = cv2.circle(frame2, (a, b), 5, (0, 0, 255), -1)

img = cv2.add(frame2, mask)

# Display the optical flow image
cv2.imshow('Optical Flow', img)
cv2.waitKey(0)
cv2.destroyAllWindows()