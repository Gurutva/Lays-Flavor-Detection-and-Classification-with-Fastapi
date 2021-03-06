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
from time import time


app = FastAPI()

transform = T.Compose([
    T.ToTensor(),
])

transform_toPil = T.Compose([
    T.ToPILImage(),
])

start_t1=time()

detector = LaysDetector()

end_t1=time()

start_t2=time()

classification_model = classification.load_model()

end_t2=time()

print("detector loading time:",end_t1-start_t1)
print("classifier loading time:",end_t2-start_t2)
print("both models loading time:",end_t2-start_t1)

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    start_t3 = time()
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
        #new_boxes_dict['box '+str(new_boxes[key][0])+':'+str(detected_images[key][0])] = np.array(new_boxes[key][1]).tolist()
        #new_boxes_dict['box ' + str(new_boxes[key][0])] = {'falvor':detected_images[key][0],'bounded-box':np.array(new_boxes[key][1]).tolist()}
        boxes_list=np.array(new_boxes[key][1]).tolist()
        x0=boxes_list[0]
        y0=boxes_list[1]
        x1=boxes_list[2]
        y1=boxes_list[3]
        new_boxes_dict['box ' + str(new_boxes[key][0])] = {'falvor': detected_images[key][0],
                                                           'bounded-box': {'x0':x0,'y0':y0,'x1':x1,'y1':y1}}

    #return new_boxes_dict
    #return FileResponse(predict_image)
    #return FileResponse(predict_image),new_boxes_dict

    # with open("sample.png", "rb") as imageFile:
    #     p = base64.b64encode(imageFile.read())
    #encoded_img = base64.b64encode(p)

    predict_image = np.asarray(predict_image)
    print(type(image))
    #print(image.size)

    _,encoded_image = cv2.imencode('.PNG',predict_image)
    print(type(encoded_image))
    #print(image.size)


    #encoded_image = base64.b64encode(encoded_image)

    with open('new_image.png','wb') as new_file:
        new_file.write(base64.decodebytes(encoded_image ))

    end_t3=time()
    print("after image passed:",end_t3-start_t3)
    print('total time:',end_t3-start_t1)

    return {
        'dimensions': new_boxes_dict,
        'encoded_img': encoded_image ,
    }

if __name__ == "__main__":
    uvicorn.run(app, debug=True)