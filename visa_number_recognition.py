from time import sleep
import cv2
import easyocr
import pytesseract
from ultralytics import YOLOWorld, YOLO
from PIL import Image
from paddleocr import PaddleOCR
import re

# Load a pretrained YOLOv8s-worldv2 model
model = YOLO("yolov8s-worldv2.engine")
ocr = PaddleOCR(
    det_model_dir='path_to_det_model',
    rec_model_dir='path_to_rec_model',
    use_tensorrt=True, use_angle_cls=False, lang='en')  # Initialize OCR
# model.set_classes(["card"])

def ocr_tesseract():
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(card_number_crop, config=custom_config)
    if len(text) >= 16:
        print(text)
        cv2.putText(frame, text, (int(x1), int(y1+0.25*(y2-y1)) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        pass

def ocr_easyocr(card_number_crop):
    if not hasattr(ocr_easyocr, "reader"):
        reader = easyocr.Reader(['en'])
    results = reader.readtext(card_number_crop, allowlist='0123456789')

    card_number = ""
    for result in results:
        card_number += result[1] + " "
    return card_number

def ocr_paddle(card_number_crop):
    # if not hasattr(ocr_easyocr, "ocr"):
    #     ocr = PaddleOCR(
    #         det_model_dir='path_to_det_model',
    #         rec_model_dir='path_to_rec_model',
    #         use_tensorrt=True, use_angle_cls=True, lang='en')  # Initialize OCR
    results = ocr.ocr(card_number_crop, det=False, cls=False)

    # Extract detected numbers
    card_number = ""
    print("="*10)
    for result in results:
        if result is not None:
            for line in results:
                for word in line:
                    # text, conf = word[1]  # Extract text
                    text, conf = word  # Extract text
                    print(f"word is {text} with conf {conf}")
                    # card_number += text

                    if conf > 0.925:
                        card_number += text

    print("="*10)
    card_number = re.sub(r'\D', '', card_number)

    if len(card_number) != 16:
        card_number = ""

    return card_number

cap = cv2.VideoCapture(0)

freeze_frame = False
while cap.isOpened():
    if not freeze_frame:
        ret, frame = cap.read()
        results = model.predict(frame, conf=0.05, verbose=False, max_det=1)
        for result in results:
            for i in range(len(result.boxes.xyxy)):
                x1, y1, x2, y2 = result.boxes.xyxy[i].to(int).tolist()
                conf = result.boxes.conf[i].item()
                cls = result.boxes.cls[i].item()
                name = result.names[cls]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                card_num_coord = x1, int(y1+0.55*(y2-y1)), x2, int(y1+(0.71)*(y2-y1))
                x1, y1, x2, y2 = card_num_coord
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                card_number_crop = frame[y1:y2+1, x1:x2+1]

                card_number = ocr_paddle(card_number_crop)

                
                if len(card_number) > 0:
                    cv2.putText(frame, f"{card_number}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
                    freeze_frame = True
        
    cv2.imshow("Mahmoud Hashem", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u'): # Press 'u' to unfreeze 
        freeze_frame = False

cap.release()
cv2.destroyAllWindows()
    # reader = easyocr.Reader(['en'])  # Load English model
    # result = reader.readtext("IMG_20250103_135902.jpg", allowlist='0123456789')
    # print(result)
