import matplotlib.pyplot as plt
import classification
from detection import LaysDetector
from PIL import Image
import torchvision.transforms as T
import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, Response
import numpy as np
from starlette.responses import StreamingResponse
from fastapi.responses import FileResponse,HTMLResponse
from PIL import Image, ImageDraw
from torchvision.utils import save_image
import cv2
import base64

app = FastAPI()

transform = T.Compose([
    T.ToTensor(),
])

transform_toPil = T.Compose([
    T.ToPILImage(),
])

detector = LaysDetector()
classification_model = classification.load_model()

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png","jfif")
    if not extension:
        return "Image must be of proper format!"
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    #image = read_imagefile(await file.read())

    img = transform(image)

    #detector = LaysDetector()
    boxes = detector.get_boxes(img)
    # detector.draw_detection()
    # print(boxes)

    detected_images = []

    count = 0
    for box in boxes:
        count += 1
        x0, y0, x1, y1 = [int(val) for val in box]
        imgx = img.permute(1, 2, 0)
        cropx = imgx[y0:y1, x0:x1]

        cropy = transform_toPil(cropx.permute(2, 0, 1))

        detected_flv, detected_score = classification.predict_image(cropy, classification_model)

        detected_images.append((detected_flv, detected_score))

    predict_image,new_boxes=detector.draw_detection(detected_images,img)

    #predict_image=transform_toPil(predict_image)

    new_boxes_dict = dict()
    for key in range(len(new_boxes)):
        new_boxes_dict['box '+str(new_boxes[key][0])+':'+str(detected_images[key][0])] = np.array(new_boxes[key][1]).tolist()
    #return new_boxes_dict
    #return FileResponse(predict_image)
    #return FileResponse(predict_image),new_boxes_dict

    # with open("sample.png", "rb") as imageFile:
    #     p = base64.b64encode(imageFile.read())
    #encoded_img = base64.b64encode(p)

    predict_image = np.asarray(predict_image)
    _,encoded_image = cv2.imencode('.PNG',predict_image)

    encoded_image  = base64.b64encode(encoded_image)

    with open('new_image3.png','wb') as new_file:
        new_file.write(base64.decodebytes(encoded_image ))

    return {
        'dimensions': new_boxes_dict,
        'encoded_img': encoded_image ,
    }


if __name__ == "__main__":
    uvicorn.run(app, debug=True)