from django.shortcuts import render
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import io
import sys
# Create your views here.


def index(request):
    return render(request,'myapp/index.html')

def login(request):
    if request.method=="POST":
        username = request.POST['uname']
        password = request.POST['pwd']
        print(username,password)
        if username == 'admin' and password == 'admin':
            return render(request, 'myapp/homepage.html')

    return render(request,'myapp/login.html')



def homepage(request):
    return render(request,'myapp/homepage.html')

def dataupload(request):
    # Define paths for the dataset
    data_dir = 'D:/cardamom//cardamom/cardamom_project/dataset/'
    train_dir = os.path.join(data_dir, 'Train')
    new_train_dir = os.path.join(data_dir, 'New_Train')
    new_test_dir = os.path.join(data_dir, 'New_Test')

    # Create new directories for training and testing data
    os.makedirs(new_train_dir, exist_ok=True)
    os.makedirs(new_test_dir, exist_ok=True)
    # Image dimensions
    img_height, img_width = 224, 224
    batch_size = 32

    # Data preprocessing and augmentation
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2)

    # Load and preprocess all data
    full_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)


    # # Split the data into training and testing sets
    # train_images, test_images,train_labels , test_labels = train_test_split(full_generator.x, full_generator.y, test_size=0.2, random_state=42)
    #
    content={
        'data1':"Number of original training images: 1725",
        'data2':"Number of augmented training images: 14216",
        'data3':"Number of augmented testing images: 32",

    }
    return render(request,'myapp/dataupload.html',content)

def createmodel(request):
    # Load saved model
    model = load_model('D:/cardamom/cardamom/model/Cardamom_plant.h5')
    print(model.summary())
    # Model summary
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Print the model summary to the redirected stdout
    # model_lstm.summary()
    model.summary()
    # Get the model summary as a string
    summary_string = sys.stdout.getvalue()

    # Reset stdout to its original value
    sys.stdout = original_stdout

    # Now, `summary_string` contains the model summary
    print(summary_string)
    content1 = {
        'data': summary_string
    }
    return render(request,'myapp/createmodel.html',content1)

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.  # Rescale to [0,1]


def predict_image(model, img_path, class_labels):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = predictions[0][predicted_class] * 100
    return predicted_label, confidence


def predictdata(request):
    if request.method=='POST':
        imgname = request.POST['myFile']
        imgpath='D:/Test Images/'
        print(imgname)
        img_path=imgpath+imgname

        model = load_model('D:/cardamom/cardamom/model/Cardamom_plant.h5')

        # Define class labels
        class_labels = ['Blight', 'Healthy', 'Phylosticta']

        # Get user input
        # img_path = 'D:/2023-24/finalprojects/cardamom/cardamom_project/dataset/Test Images/C_Blight35.jpg'

        # Check if the file exists
        if not os.path.exists(img_path):
            print("File not found!")
            return
        # res = predict(img_path)
        # Make predictions
        predicted_label, confidence = predict_image(model, img_path, class_labels)
        print("Predicted Class:", predicted_label)
        print("Confidence:", confidence)

        # Display input image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add text for predicted class and probability to the image
        text = f"Predicted Class: {predicted_label}\nConfidence: {confidence:.2f}%"
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        # Print associated explanation
        print("\nExplanation for", predicted_label + ":")
        print(class_explanations[predicted_label])
        res=class_explanations[predicted_label]
        content={
            'data':"Predicted Class:",
            'data1':predicted_label,
            'data2':'Confidence: ',
            'data3':confidence,
            'data4':res ,

        }
        return render(request, 'myapp/predictdata.html',content)
    return render(request, 'myapp/predictdata.html')

class_explanations = {
    'Blight': """
        Leaf blight, also known as Chenthal disease,
        is a fungal disease that affects cardamom leaves.
        It causes water-soaked lesions to appear on the leaf's upper surface,
        and the lesions may become brown or dark brown with a pale yellow halo.
        The leaves may wither and the pseudostems may wilt.
        The disease can also cause new shoots to develop that are smaller,
        and flowers may fail to develop. The affected garden may have a burnt appearance
        """,
    'Healthy': """
        Good cardamom is an herb known for its aromatic flavor and potential health benefits.
        Chewing cardamom seeds can promote oral health by balancing mouth pH levels and increasing saliva production,
        which may help prevent cavities.
        Additionally, this cardamom has been associated with other potential health benefits such as balancing blood sugar levels,
        treating respiratory conditions like bronchitis, supporting heart and liver health,
        preventing ulcers, and possibly even exhibiting anti-cancer properties.
        """,
    'Phylosticta': """
        Phyllosticta elettariae, also known as nursery leaf spot, is a fungus that causes leaf spots in cardamom plants.
        The spots are small, round, or oval, and are dull white in color. They usually appear between February and April.
        In severe cases, the leaves may rot and tillering may decrease.
        """
    }



def viewgraph(request):
    return render(request,'myapp/viewgraph.html')
