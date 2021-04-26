import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

class HumanRemover:
    def __init__(self, img_path):
        self.img_path = img_path

    # HaarCascad 함수
    # 매개변수 : 이미지경로('C:/Study/project/data/test5.jpg')
    # 반환값 : 검출된 몸의 사각형 좌표 리스트
    def haarcascade(self, img):

        # 이미지 로드, BGR 색공간 그레이 스케일
        image = cv2.imread(self.img_path)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 얼굴검출 haarcascades 경로
        # 이미 얼굴, 눈 등에 대해 미리 훈련된 데이터를 XML 파일 형식으로 제공
        face_path_dir = 'C:/Study/project/haarcascades/face'
        face_file_list = os.listdir(face_path_dir)
        # 몸검출 haarcascades 경로
        body_path_dir = 'C:/Study/project/haarcascades/body'
        body_file_list = os.listdir(body_path_dir)

        full_body = [] # 검출된 사각형 저장 할 리스트

        # 얼굴검출
        for cascade in face_file_list:
            # 사전 훈련된 모델생성, haarcascades XML파일 로드 
            face_cascade = cv2.CascadeClassifier(f'{face_path_dir}/{cascade}')

            # 위에서 지정한 얼굴 인식을 시도
            face = face_cascade.detectMultiScale(grayImage, # image
                                                    1.03,    # ScaleFactor       
                                                    5)       # minNeighbor
            if len(face) == 0 : #검출된 것이 없으면 아래코드 실행 안함
                continue

            if face.shape[0] > 1:   #검출된 것이 한개 이상이면 1개 배열씩 결합하기위해
                for i in range(face.shape[0]):
                    full_body.append(face[i:i+1,:4])
                continue

            full_body.append(face)


        # 몸검출
        for cascade in body_file_list:
            # 사전 훈련된 모델생성, haarcascades XML파일 로드 
            body_cascade = cv2.CascadeClassifier(f'{body_path_dir}/{cascade}')

            # 위에서 지정한 얼굴 인식을 시도
            body = body_cascade.detectMultiScale(grayImage, # image
                                                    1.03,    # ScaleFactor       
                                                    5)       # minNeighbor

            if len(body) == 0 : #검출된 것이 없으면 아래코드 실행 안함
                continue

            if body.shape[0] > 1:   #검출된 것이 한개 이상이면 1개 배열씩 결합하기위해
                for i in range(body.shape[0]):
                    full_body.append(body[i:i+1,:4])
                continue

            full_body.append(body)

        # 배열변환 
        full_body = np.array(full_body)
        
        return full_body


    # GrabCut 함수
    # 매개변수 : 이미지, 검출된 몸의 사각형 리스트
    # 반환값 : 사람이 제거된 이미지 
    def grabcut(self, img, rect):

        image = cv2.imread(img)
        mask = np.zeros(image.shape[:2],np.uint8) #원본과 같은 크기 0으로 채워진 마스크
                                    # np.uint8: 양수만 표현

        # # rect.shape(0,)아닐때만
        # rect = rect.reshape(-1,4)
        if len(rect) != 0:
            # (1,65)모양 0으로 채워진 배열 반환
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)

            # rect = haarcascade로 나온 결과 3차원
            # grabcut 범위 넓히기
            plus = [-20,-15,35,80]
            rect[:,:,0] = rect[:,:,0] + plus[0]
            rect[:,:,1] = rect[:,:,1] + plus[1]
            rect[:,:,2] = rect[:,:,2] + plus[2]
            rect[:,:,3] = rect[:,:,3] + plus[3]
            
            for i in rect :

                cv2.grabCut(image,      # 원본이미지
                            mask,       # 마스크
                            i,          # 사각형
                            bgdModel,   # 배경을 위한 임시 배열 
                            fgdModel,   # 전경을 위한 임시 배열
                            5,          # 반복횟수
                            cv2.GC_INIT_WITH_RECT)   # 사각형을 위한 초기화

                mask2 = np.where((mask==2)|(mask==0),1,0).astype('uint8') # 사람이면 *0, 배경이면 *1
                image = image*mask2[:,:,np.newaxis] # 사람에 *0을해서 검정색으로 나오는 상태

            # plt.imshow(image),plt.colorbar(),plt.show()

        return image


    def remove(self, img_folder_path, grabcut_folder_path, search_name_list):
        for search in search_name_list:     # 이미지종류선택
            img_file_list = os.listdir(f'{img_folder_path}/{search}')   # 해당이미지 폴더 대입
            
            for img in img_file_list:           # 해당이미지폴더의 전체에서 이미지 한개씩
                origin_image = f'{img_folder_path}/{search}/{img}'  # 이미지 대입
                print(origin_image)
                # 사람검출 함수 호출
                rect = self.haarcascade(origin_image)
                # 사람제거 함수 호출
                grabcut_img = self.grabcut(origin_image, rect)
                # 사람제거완료한 이미지 저장
                if not os.path.isdir(f'{grabcut_folder_path}/{search}'): 
                    os.mkdir(f'{grabcut_folder_path}/{search}')

                cv2.imwrite(f'{grabcut_folder_path}/{search}/grabcut_{img}', grabcut_img)
                cv2.waitKey(0)

                print(search, '저장')
            

            
        

