from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import urllib.request
import os
import glob
import numpy as np


class Data:
    def __init__(self, img_folder_path, search_name_list):
        self.img_folder_path = img_folder_path
        self.search_name_list = search_name_list

    # crawling 함수
    # 매개변수: 검색할 키워드 리스트, 이미지를 저장할 경로(예시'./project/data/img4'), 
    def crawling(self):
        img_folder_path = self.img_folder_path
        search_name = self.search_name_list

        for search_name in search_name_list :
            img_folder_name = search_name.replace(' ', '')
            
            # select webbrowser (chrome)
            driver = webdriver.Chrome('./project/chromedriver_win32/chromedriver.exe')

            # link to address
            driver.get('https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl')

            # fine specified elements
            elem=driver.find_element_by_name('q')

            # input keys & enter
            elem.send_keys(search_name)
            elem.send_keys(Keys.RETURN)

            # scroll web page
            SCROLL_PAUSE_TIME=1
            last_height=driver.execute_script('return document.body.scrollHeight')
            # 스크롤 높이를 java Script 로 찾아서 last_height 란 변수에 저장 시킴

            while True: # 무한 반복
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                # 스크롤을 끝까지 내린다

                time.sleep(SCROLL_PAUSE_TIME) # 스크롤이 끝나면 1초동안 기다림

                new_height=driver.execute_script('return document.body.scrollHeight')
                if new_height==last_height:
                    try:
                        driver.find_element_by_css_selector('.mye4qd').click()
                        # 결과 더보기 버튼 클릭
                    except:
                        break
                last_height=new_height

            # select & click image in webbrowser 
            images=driver.find_elements_by_css_selector('.rg_i.Q4LuWd')
            count=1

            if not os.path.isdir(f'{img_folder_path}/{img_folder_name}'): 
                    os.mkdir(f'{img_folder_path}/{img_folder_name}/')

            for image in images:
                try:
                    image.click() # 인터넷 상의 이미지 클릭
                    time.sleep(3) # 이미지 로드 시간을 위해 지연시간 추가
                    imgUrl=driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img').get_attribute('src') # 저장할 이미지 경로
                    # if imgUrl==driver.find_element_by_link_text('https://images.costco-static.com/ImageDelivery/imageService?profileId=12026540&itemId=1462223-847&recipeName=680'):
                    #     print('tq')
                    opener=urllib.request.build_opener()
                    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
                    urllib.request.install_opener(opener)
                    urllib.request.urlretrieve(imgUrl, f'{img_folder_path}/{img_folder_name}/{img_folder_name}{count}.jpg') # 이미지 저장
                    count=count+1 # 이미지 파일 이름을 순서대로 맞추기 위해 증가시킴
                    
                    time.sleep(5) # 저장 후 페이지 로드 할 시간을 위해 지연시간 추가
                    '''
                    Forbidden 이 뜨면 위의 코드를 추가한다
                    python 으로 제어 되는 브라우저를 봇으로 인식하는 경우,
                    위의 header 를 추가해주면 해당 문제를 벗어날 수 있다.
                    '''

                except:
                    pass

            driver.close() # 웹페이지 종료

        print('--->crawling 완료')
   
    # DataToNumpy 함수
    # 매개변수 : 이미지 너비, 높이, 이미지데이터제너레이터 여부, validation_split
    # 반환값 : x, y 
    def data_to_numpy(self, image_w, image_h, IDG=False, validation_split=0.25):

        # ======================================분류 대상 카테고리 선택하기 
        img_folder_path = self.img_folder_path
        categories = self.search_name_list
        nb_classes = len(categories)

        # =====================================이미지 크기 지정 
        image_w = image_w 
        image_h = image_h
        pixels = image_w * image_h * 3

        # ======================================이미지 데이터 읽어 들이기 
        X = []
        Y = []
        for idx, value in enumerate(categories):
            # 레이블 지정 
            label = [0 for i in range(nb_classes)]
            label[idx] = 1
            # 이미지 
            image_dir = img_folder_path + "/" + value
            files = glob.glob(image_dir +"/*.jpg")
            for i, f in enumerate(files):      
                img = Image.open(f) 
                img = img.convert("RGB")
                img = img.resize((image_w, image_h))
                data = np.asarray(img)      # 데이터 형태가 다를 경우에만 numpy 배열로 변환 
                X.append(data)
                Y.append(label)
        # 이미지를 RGB로 변환 후, 64x64 크기로 resize

        # ======================================ImageDataGenerator
        if IDG == True:
            train_datagen = ImageDataGenerator(
                rescale= 1./255,            # 0-1사이값으로 정규화
                horizontal_flip=True,       # 수평뒤집음
                # vertical_flip=True,       # 수직뒤집음
                width_shift_range=0.1,      # 수평이동
                height_shift_range=0.1,     # 수직이동
                rotation_range=5,           # 무작위회전 
                zoom_range=1.2,             # 임의확대, 축소
                # shear_range=0.7,          # 층밀리기의 강도
                fill_mode='nearest',        # 빈자리 주변 유사수치로 채움(=0이면 0으로 채움)
                validation_split=validation_split
        
            )
            test_dategen = ImageDataGenerator(rescale=1./255) # test 이미지는 증폭XXX, 0-1사이값으로 정규화

            xy_train = train_datagen.flow_from_directory(
                img_folder_path,
                target_size = (image_w,image_h),        
                batch_size = 100000,
                class_mode = 'categorical'   ,        
                subset='training'
            )

            xy_val = train_datagen.flow_from_directory(
                img_folder_path,
                target_size = (image_w,image_h),       
                batch_size = 100000,
                class_mode = 'categorical' ,          
                subset='validation'
            )
            # npy 저장
            np.save('./project/data/npy/class_data_train_x2.npy', arr=xy_train[0][0])
            np.save('./project/data/npy/class_data_train_y2.npy', arr=xy_train[0][1])
            np.save('./project/data/npy/class_data_val_x2.npy', arr=xy_val[0][0])
            np.save('./project/data/npy/class_data_val_y2.npy', arr=xy_val[0][1])

            return xy_train, xy_val

        X = np.array(X)
        Y = np.array(Y)

        # npy 저장
        np.save('./project/data/npy/class_data_X.npy', arr=X)
        np.save('./project/data/npy/class_data_Y.npy', arr=Y)

        return X, Y

    

    