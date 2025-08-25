import cv2
import mediapipe as mp
import os

IMAGE_DIR = '/data2/masino_lab/abamini/Face2pheno/images_human'
ANNOTATED_IMAGE_DIR = '/data2/masino_lab/abamini/Face2pheno/images_human_annotated'  #  save annotated images
LANDMARK_SAVE_DIR = '/home/abamini/Face2pheno/images'  # save landmarks

# Make sure save dir exists
os.makedirs(LANDMARK_SAVE_DIR, exist_ok=True)
os.makedirs(ANNOTATED_IMAGE_DIR, exist_ok=True) 

IMAGE_FILES = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        if image is None:
            print(f"Cannot read {file}")
            continue

        image_height, image_width, _ = image.shape

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Print number of faces detected in this image
        num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        print(f"{os.path.basename(file)}: {num_faces} face(s) detected")
        
        if not results.multi_face_landmarks:
            continue
            
        annotated_image = image.copy()
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # Save landmarks to a txt file
            landmark_filename = os.path.join(
                LANDMARK_SAVE_DIR,
                f'{os.path.splitext(os.path.basename(file))[0]}_landmarks_pixels.txt' # Changed filename slightly
            )
            with open(landmark_filename, 'w') as lm_file:
                for i, lm in enumerate(face_landmarks.landmark):
                    pixel_x = int(lm.x * image_width)
                    pixel_y = int(lm.y * image_height)
                    
                    # The z coordinate's scale is approximately the same as x's. 
                    # We scale it by the width to keep it relative.
                    pixel_z = lm.z * image_width
                    
                    visibility = lm.visibility if hasattr(lm, "visibility") else ""
                    
                    # Write the new pixel coordinates to the file 
                    #lm_file.write(f'{i},{pixel_x},{pixel_y},{pixel_z},{visibility}\n')
                    lm_file.write(f'{i},{pixel_x},{pixel_y}\n')
            
            print(f"Saved pixel landmarks to {landmark_filename}")

            # Draw as before
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                
        # Save annotated image
        output_path = os.path.join(
            ANNOTATED_IMAGE_DIR, f'annotated_{os.path.basename(file)}')
        cv2.imwrite(output_path, annotated_image)
        print(f'Saved annotated image to {output_path}')