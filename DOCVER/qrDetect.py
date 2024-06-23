from qreader import QReader
import cv2
qreader = QReader()

# Get the image that contains the QR code
image = cv2.cvtColor(cv2.imread("ad.jpg"), cv2.COLOR_BGR2RGB)

# Use the detect_and_decode function to get the decoded QR data
decoded_text = qreader.detect_and_decode(image=image)

def convertTuple(tup):
        # initialize an empty string
    str = ''
    for item in tup:
        str = str + item
    return str

decodedS= convertTuple(decoded_text)

print(decoded_text)
