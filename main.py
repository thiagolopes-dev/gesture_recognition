import cv2
import math
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# Inicializa o módulo Hands do Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Variáveis para armazenar as coordenadas anteriores dos landmarks
prev_landmark_coords = []

# Carrega uma fonte TrueType com suporte a caracteres UTF-8
font_path = "/Users/thiagolopes/Downloads/Rajdhani/Rajdhani-Bold.ttf"  # Substitua pelo caminho para sua fonte TrueType
font_size = 50
font = ImageFont.truetype(font_path, font_size)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        # Lê o quadro da câmera
        success, frame = cap.read()
        if not success:
            print("Não foi possível ler o quadro da câmera.")
            break

        # Converte a imagem de BGR para RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processa a imagem com o Mediapipe
        results = hands.process(image)

        # Desenha landmarks e conectores nas mãos detectadas
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

                # Verifica se houve mudança significativa nas coordenadas dos landmarks
                current_landmark_coords = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
                if prev_landmark_coords:
                    if any(math.dist(curr_coord, prev_coord) > 0.05 for curr_coord, prev_coord in zip(current_landmark_coords, prev_landmark_coords)):
                        # Imprime as coordenadas dos landmarks
                        for idx, (curr_coord, prev_coord) in enumerate(zip(current_landmark_coords, prev_landmark_coords)):
                            x_diff = curr_coord[0] - prev_coord[0]
                            y_diff = curr_coord[1] - prev_coord[1]
                            z_diff = curr_coord[2] - prev_coord[2]
                            # print(f"Landmark {idx}: ({x_diff}, {y_diff}, {z_diff})")

                # Atualiza as coordenadas anteriores dos landmarks
                prev_landmark_coords = current_landmark_coords

                # Verifica se a mão está fechada
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distance = abs(thumb_tip.x - index_finger_tip.x)

                if distance < 0.02:
                    # Cria uma imagem vazia usando o formato PIL
                    pil_image = Image.fromarray(image)

                    # Cria um objeto ImageDraw para desenhar no PIL image
                    draw = ImageDraw.Draw(pil_image)

                    # Obtém a posição atual da mão fechada
                    hand_pos = (int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0]))

                    # Define a posição e o texto a ser exibido
                    text = 'Mão Fechada'
                    text_position = (hand_pos[0] - 100, hand_pos[1] - 100)

                    # Desenha o texto na imagem PIL
                    draw.text(text_position, text, font=font, fill=(0, 0, 255))

                    # Converte a imagem PIL de volta para o formato OpenCV
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Mostra o resultado
        cv2.imshow('Hand Tracking', image)

        # Verifica se a tecla 'q' foi pressionada para sair do loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
