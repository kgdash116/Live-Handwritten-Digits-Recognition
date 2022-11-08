import pandas as pd
import numpy as np
import tensorflow as tf
import cv2


def image_analysis():
    model = tf.keras.models.load_model('Model.h5')
    img=input('Enter file name ')
    image = cv2.imread(img)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preprocessed_digit = []
    for c in contours:
        x, y, width, height = cv2.boundingRect(c) #get dimensions

        if width * height >= 200:
            cv2.rectangle(image, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
            digit = thresh[y:y + height, x:x + width]  # Cropping out the digit from the image
            # new_ratio = cv2.resize(digit, (12, 12))
            # augment = np.pad(resized_digit, ((3, 3), (3 3)),
            resized_digit = cv2.resize(digit, (18, 18))  # Padding the digit with 5 pixels of black color
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
            preprocessed_digit.append(padded_digit)

            for digit in preprocessed_digit:
                data = tf.keras.utils.normalize(digit)
                prediction = model.predict(data.reshape(1, 28, 28, 1))
                label = str(np.argmax(prediction))
                #(prediction[0][np.argmax(prediction)])*100)

            cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Detection', image)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


def webcam_detetction():
    model = tf.keras.models.load_model('Model.h5')

    cap = cv2.VideoCapture(0)
    #ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    #contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam error")
            break

        image = frame
        #convert to grayscale
        grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        #Retrieving threshold
        ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        preprocessed_digit = []
        for c in contours:
            x, y, width, height = cv2.boundingRect(c)

            if width * height >= 200:
                cv2.rectangle(image, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
                one_digit = thresh[y:y + height, x:x + width]  # Cropping out the digit from the image

                #new_ratio = cv2.resize(digit, (12, 12))
                #augment = np.pad(resized_digit, ((3, 3), (3 3)),

                resized_digit = cv2.resize(one_digit, (18, 18))  # Padding the digit with 5 pixels of black color
                padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
                preprocessed_digit.append(padded_digit)

                for digit in preprocessed_digit:
                    data = tf.keras.utils.normalize(digit)
                    prediction = model.predict(data.reshape(1, 28, 28, 1))
                    label = str(np.argmax(prediction))
                    #(prediction[0][np.argmax(prediction)])*100)

                cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    x = input('Enter option 1 for recognition on images or 2 for webcam detection ')
    if(x=='1'):
        image_analysis()
    elif(x=='2'):
        webcam_detetction()
    else:
        print("WRONG OPTION SELECTED!")
        exit(225)