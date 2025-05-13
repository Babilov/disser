import cv2

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
cv2.imshow("Test", cv2.imread("/home/griwa/Рабочий стол/Dissertation/code/client-server-app/fastapi_server/temp_resized_image.jpg"))
cv2.waitKey(0)
cv2.destroyAllWindows()