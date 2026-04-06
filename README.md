# Fruits Object Detection Project

### Labelling Objects with Label Studio

[Label Studio](https://labelstud.io/?utm_source=youtube&utm_medium=video&utm_campaign=edjeelectronics)
```bash
# install label studio
uv add label-studio

# launch label studio server and create account with any email and password
label-studio start
```

* On ```label-studio``` server, annotate each image with respect to the classes/labels. After annotation of all images, click project name, click export ```(Yolo with Images)``` to download the following:

1. ***images***
2. ***labels***
3. ***classes***
4. ***notes***

* Upload downloaded annotated data into the working directory

### Training on Google Colab
[fruit-yolo notebook](Yolo_fruit_model_from_scratch.ipynb)
[yolo-lecture notebook](Train_YOLO_Models.ipynb)
[Streamlit_app](my_model\yolo_streamlit_app.py)

### Run Inference Locally
```bash
cd my_model
uv add ultralytics

# inference on PC Live Stream Camera
uv run python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720

# inference on test image
uv run python yolo_detect.py --model my_model.pt --source "fruits/test" #--resolution 1280x720

# inference on video
uv run python yolo_detect.py --model my_model.pt --source <video path>

# Streamlit UI
uv run streamlit run yolo_streamlit_app.py
```