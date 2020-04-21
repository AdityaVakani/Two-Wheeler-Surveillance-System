import requests
import io
import cv2
def plate_reader(img):
    _, buffer = cv2.imencode(".jpg", img)
    io_buf = io.BytesIO(buffer)
    token="1d3d7517ff820f00f7a4c9d0404aef2eacce850f"
    regions = ['in', 'it'] # Change to your country
    response = requests.post(
        'https://api.platerecognizer.com/v1/plate-reader/',
        data=dict(regions=regions),  # Optional
        files=dict(upload=io_buf),
        headers={'Authorization': 'Token 1d3d7517ff820f00f7a4c9d0404aef2eacce850f '})
    out=response.json()
    if out['results']==[]:
         return "Can't read"
    num_plate = out['results'][0]['plate']
    score=out['results'][0]['score']
    return num_plate,score

# img = cv2.imread('D:\\Programs\\Final Project\\code_output\\wb01jc0137\\num_plate.jpg')
# print(plate_reader(img))

