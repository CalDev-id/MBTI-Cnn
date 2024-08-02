from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=16))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def predict_mbti():
    # Mendapatkan input dari permintaan POST
    request_data = [
        "Y",
        "Y",
        "Y",
        "N",
        "N",
        "N",
        "N",
        "Y",
        "Y",
        "Y",
        "Y",
        "Y",
        "Y",
        "N",
        "N",
        "N"
    ]

    # Pertanyaan untuk setiap dimensi
    questions = [
        "Apakah Anda lebih suka belajar melalui interaksi dengan orang lain? (Y/T)",
        "Apakah Anda lebih suka belajar melalui pengamatan dan pengalaman langsung? (Y/T)",
        "Apakah Anda lebih suka belajar melalui membaca dan riset mandiri? (Y/T)",
        "Apakah Anda lebih suka belajar dengan mendengarkan penjelasan dari orang lain? (Y/T)",
        "Apakah Anda lebih suka belajar dengan mencoba dan berlatih secara langsung? (Y/T)",
        "Apakah Anda lebih suka belajar dengan berdiskusi dan berbagi ide dengan orang lain? (Y/T)",
        "Apakah Anda lebih suka belajar dengan membuat catatan dan merencanakan langkah-langkah? (Y/T)",
        "Apakah Anda lebih suka belajar dengan mengikuti instruksi dan panduan yang jelas? (Y/T)",
        "Apakah Anda lebih suka menghabiskan waktu luang dengan aktivitas luar ruangan? (Y/T)",
        "Apakah Anda lebih suka menghabiskan waktu luang dengan membaca buku atau menonton film? (Y/T)",
        "Apakah Anda lebih suka menghabiskan waktu luang dengan bermain olahraga atau aktif fisik? (Y/T)",
        "Apakah Anda lebih suka menghabiskan waktu luang dengan berkreasi atau menggambar? (Y/T)",
        "Apakah Anda lebih suka menghabiskan waktu luang dengan berinteraksi sosial bersama teman? (Y/T)",
        "Apakah Anda lebih suka menghabiskan waktu luang dengan merenung atau berpikir secara pribadi? (Y/T)",
        "Apakah Anda lebih suka menghabiskan waktu luang dengan mengeksplorasi tempat baru? (Y/T)",
        "Apakah Anda lebih suka menghabiskan waktu luang dengan menulis atau menyusun ide? (Y/T)"
    ]
    
    print("Request array: ")
    print(request_data)

    # Inisialisasi poin untuk setiap dimensi
    scores = {
        'E': 0,
        'I': 0,
        'S': 0,
        'N': 0,
        'T': 0,
        'F': 0,
        'J': 0,
        'P': 0,
        'E2': 0,
        'I2': 0,
        'S2': 0,
        'N2': 0,
        'T2': 0,
        'F2': 0,
        'J2': 0,
        'P2': 0
    }
    
    print("Score sebelum if answer == 'Y' / 'N' : ")
    print(scores)

    # Ajukan pertanyaan dan update skor
    for i in range(len(questions)):
        answer = request_data[i].upper()
        if answer == 'Y':
            scores['E'] += 1
            scores['S'] += 1
            scores['T'] += 1
            scores['J'] += 1
        elif answer == 'N':
            scores['I'] += 1
            scores['N'] += 1
            scores['F'] += 1
            scores['P'] += 1
    
    print("Score setelah if answer == 'Y' / 'N' : ")
    print(scores)

    # Tentukan preferensi berdasarkan skor
    mbti_type = ""
    if scores['E'] > scores['I']:
        mbti_type += 'E'
    else:
        mbti_type += 'I'

    if scores['S'] > scores['N']:
        mbti_type += 'S'
    else:
        mbti_type += 'N'

    if scores['T'] > scores['F']:
        mbti_type += 'T'
    else:
        mbti_type += 'F'

    if scores['J'] > scores['P']:
        mbti_type += 'J'
    else:
        mbti_type += 'P'

    # Konversi skor menjadi input untuk model
    input_data = []
    for key in scores.keys():
        input_data.append(scores[key])

    # Normalisasi input
    total = sum(input_data)
    input_data = [score / total for score in input_data]

    # Load model
    model = create_model()
    model.load_weights('content/model_mbti.h5')

    # Prediksi tipe MBTI
    prediction = model.predict([input_data])
    mbti_types = ['ESTJ', 'ISTJ', 'ENTJ', 'INTJ', 'ESTP', 'ISTP', 'ENTP', 'INTP', 'ESFJ', 'ISFJ', 'ENFJ', 'INFJ', 'ESFP', 'ISFP', 'ENFP', 'INFP']
    mbti_index = prediction.argmax()
    mbti_type_nn = mbti_types[mbti_index]

    response = {
        'predicted_mbti': mbti_type,
        'matched_mbti': mbti_type_nn
    }

    print(response)
    
predict_mbti()
