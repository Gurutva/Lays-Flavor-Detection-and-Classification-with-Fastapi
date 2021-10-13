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
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw
from torchvision.utils import save_image
import cv2

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

    new_boxes_dict = dict()
    for key in range(len(new_boxes)):
        new_boxes_dict['box '+str(new_boxes[key][0])+':'+str(detected_images[key][0])] = np.array(new_boxes[key][1]).tolist()
    #return new_boxes_dict
    return FileResponse(predict_image)
    #return FileResponse(predict_image),new_boxes_dict


    # image=transform(image)
    # save_image(image, 'sample.png')
    # return new_boxes
    #return FileResponse('sample.png')
    #return {FileResponse('sample.png') , new_boxes}

    #return Response(io.BytesIO(image.tobytes()), media_type="image/png")
    # byte_io=io.BytesIO()
    # return image.save(byte_io, 'PNG')

    #return FileResponse(image)

    #return plt.imshow(image)

    #return FileResponse(file.filename)

# @app.post("/vector_image")
# def image_endpoint(*, vector):
#     # Returns a cv2 image array from the document vector
#     cv2img = my_function(vector)
#     res, im_png = cv2.imencode(".png", cv2img)
#     return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, debug=True)