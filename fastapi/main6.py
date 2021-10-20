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

    img = transform(image)

    start_t4 = time()
    boxes = detector.get_boxes(img)
    end_t4 = time()
    print('get bounded boxes from detector:',end_t4-start_t4)

    detected_images = []

    count = 0
    start_t5 = time()

    for box in boxes:
        x0, y0, x1, y1 = [int(val) for val in box]
        imgx = img.permute(1, 2, 0)
        cropx = imgx[y0:y1, x0:x1]

        cropy = transform_toPil(cropx.permute(2, 0, 1))

        start_t6=time()
        detected_flv, detected_score = classification.predict_image(cropy, classification_model)
        end_t6 = time()
        print('get box {0} flavor in for classifier:'.format(count), end_t6 - start_t6)

        detected_images.append((detected_flv, detected_score))
        count += 1

    end_t5 = time()
    print('get all flavors from classifier:', end_t5 - start_t5)

    start_t8=time()
    predict_image,new_boxes=detector.draw_detection(detected_images,img)
    end_t8=time()
    print('draw detection part:',end_t8-start_t8)

    new_boxes_dict = dict()
    for key in range(len(new_boxes)):
        boxes_list=np.array(new_boxes[key][1]).tolist()
        x0=boxes_list[0]
        y0=boxes_list[1]
        x1=boxes_list[2]
        y1=boxes_list[3]
        new_boxes_dict['box ' + str(key)] = {'falvor': detected_images[key][0],
                                                           'bounded-box': {'x0':x0,'y0':y0,'x1':x1,'y1':y1}}

    # predict_image = np.asarray(predict_image)
    # print('2',type(image))
    # print(image.size)
    #
    # _,encoded_image = cv2.imencode('.PNG',predict_image)
    # print('3',type(encoded_image))
    # print(encoded_image.shape)
    #
    # encoded_image=np.reshape(encoded_image,image.size)
    #
    #
    # encoded_image = base64.b64encode(encoded_image)
    #
    # with open('new_image.png','wb') as new_file:
    #     new_file.write(base64.decodebytes(encoded_image ))

    #########################################
    b=io.BytesIO()
    predict_image.save(b,format='jpeg')
    encoded_image=base64.b64encode(b.getvalue())

    with open('new_image.png', 'wb') as new_file:
        new_file.write(base64.decodebytes(encoded_image ))
    #########################################

    end_t3=time()
    print("after image passed till return:",end_t3-start_t3)
    print('total time from loading to return:',end_t3-start_t1)

    return {
        'dimensions': new_boxes_dict,
        'encoded_img': encoded_image ,
    }

if __name__ == "__main__":
    uvicorn.run(app, debug=True)