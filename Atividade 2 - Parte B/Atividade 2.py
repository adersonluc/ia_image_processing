# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:38:44 2019

@author: AdersonLucas
"""

# Atividade 2 - IESB IA
# Aluno: Aderson Lucas Guimarães Mendonça Medeiros
# Matrícula: 1831143043

# importar os pacotes necessários
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
import cv2

cap = cv2.VideoCapture(0)
known_image = face_recognition.load_image_file('eu.jpg')
class_name = 'Aderson'
known_encoding = face_recognition.face_encodings(known_image)[0]

print(known_encoding)

while True:
    ret, frame_rgb = cap.read()
#    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    

    try:

        face_locations = face_recognition.face_locations(frame_rgb, model='hog')
        matches = []
        
        unknown_encoding_list = face_recognition.face_encodings(frame_rgb, face_locations)
        
        for unknown_encoding in unknown_encoding_list:
            results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.65)
            print(results)
            matches.append(results[0])
    
    except IndexError:
        print('Face não reconhecida!!!')
        continue

# Draw rectangle
# Draw box around the face
    for face_tuple, match in zip(face_locations, matches):
        cv2.rectangle(frame_rgb, (face_tuple[3], face_tuple[0]), (face_tuple[1], face_tuple[2]), (0, 255, 0), 2)
        cv2.rectangle(frame_rgb, (face_tuple[3], face_tuple[2] - 35), (face_tuple[1], face_tuple[2]), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        
        if match:
            cv2.putText(frame_rgb, class_name, (face_tuple[3] + 6, face_tuple[2] - 6), font, 1.0, (255, 255, 255), 1)
        else:
            cv2.putText(frame_rgb, 'Desconhecido', (face_tuple[3] + 6, face_tuple[2] - 6), font, 1.0, (255, 255, 255), 1)
        
    # Display the resulting frame
    cv2.imshow('frame', frame_rgb)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        cap.release()
        break