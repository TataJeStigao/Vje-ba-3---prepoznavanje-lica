from deepface import DeepFace
import cv2

def PrepoznavanjeSlike():
    backend = "opencv"
    imeslike = #bilo koja od 4 slike#
    slika = cv2.imread(imeslike)
    slika = cv2.resize(slika, (720, 640))
    sivaslika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    lice = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    analiza = DeepFace.analyze(img_path=imeslike, actions=["age", "gender"])
    print(analiza)

    for (x,y,w,h) in lice:
        cv2.rectangle(slika, (x, y), (x+w, y+h), (0, 255, 0), 2)
        godine = f"Dob: {rezultat[0]['age']}; Spol: {rezultat[0]['dominant_gender']}"
        cv2.putText(slika, godine, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 1)

    slika_rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)

    cv2.imshow("lice", slika_rgb)
    cv2.waitKey(0)

def PrepoznavanjeKamere():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    kamera = cv2.VideoCapture(0)
    KrajAnalize = False

    while True:
        _, frame = kamera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lice = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)

        if not KrajAnalize:
            result = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)
            KrajAnalize = True

        for (x,y, w, h) in lice:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            godine = f"Age:{result[0]['age']}/Gender:{result[0]['dominant_gender']}"
            cv2.putText(frame, godine, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) == ord("q"):
            break


    kamera.release()
    cv2.destroyAllWindows()
