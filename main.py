import numpy as np
import os
import cv2
import imutils
import matplotlib.pyplot as plt
import class_data as Data
import class_human_remover as HR
import class_modeling as Modeling
import collections
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, RMSprop, Nadam
from PIL import Image
from sklearn.model_selection import train_test_split

# 크롤링한 이미지 폴더
img_folder_path = 'C:/Study/project/data/img3'
# 사람제거한 이미지 폴더
grabcut_folder_path = 'C:/Study/project/data/grabcut'
search_name_list = [ "dumbbell","gymball","ladderbarrel","runningmachine","yogamat"]

# 사람제거 인스턴스 생성 및 제거함수 호출
human_remover = HR.HumanRemover(img_folder_path)
human_remover.remove(img_folder_path,grabcut_folder_path,search_name_list)


#파라미터 튜닝
IMAGE_W = 100
IMAGE_H = 100
EPOCHS = 100
BS = 20
OPTI = Nadam
INIT_LR = 1e-2
FAC = 0.1
#############


# data to numpy
# Data 인스턴스 생성
data = Data.Data(grabcut_folder_path, search_name_list)
xy_train, xy_val = data.data_to_numpy(IMAGE_W, IMAGE_H, IDG=True)

x_train = np.load('C:/Study/project/data/npy/class_data_train_x2.npy')
y_train = np.load('C:/Study/project/data/npy/class_data_train_y2.npy')
x_val = np.load('C:/Study/project/data/npy/class_data_val_x2.npy')
y_val = np.load('C:/Study/project/data/npy/class_data_val_y2.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 104, shuffle = True)


# 모델, 컴파일, 훈련
# Modeling 인스턴스 생성
modeling = Modeling.Modeling(x_train, search_name_list, OPTI, INIT_LR, FAC)
model= modeling.model()
er,mo,lr = modeling.callbacks()
# model.summary()

# history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BS, callbacks=[er, lr], validation_data=(x_val, y_val))

# 학습 완료된 모델 저장
hdf5_file = "C:/Study/project/data/h5/0220_main2_modeling3.hdf5"
if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    history =  model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BS, callbacks=[er, lr], validation_data=(x_val, y_val))
    model.save_weights(hdf5_file)


# 모델 평가하기 
score = model.evaluate(x_test, y_test, batch_size = BS)
print('loss : ', score[0])
print('acc  : ', score[1])
print("{:.2f}%".format(score[1]*100))

# loss :  1.0878324508666992
# acc  :  0.5301204919815063
# 53.01%



####################################### 사진과 결과 시각화 

test_path_dir = 'C:/Study/project/data/ppt/input2' 
test_file_list = os.listdir(test_path_dir)
for test in test_file_list:
    # 적용해볼 이미지 
    test_image = f'{test_path_dir}/{test}'
    # 이미지 resize
    img = Image.open(test_image)
    img = img.convert("RGB")
    img = img.resize((IMAGE_W,IMAGE_H))
    data = np.asarray(img)
    X = np.array(data)
    X = X.astype("float") / 255
    X = X.reshape(-1, IMAGE_W,IMAGE_H, 3)

    # 예측
    pred = model.predict(X)  
    result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
    print('New data category : ',search_name_list[result[0]])


    ####################################### 사진과 결과 시각화 
    output = cv2.imread(test_image)
    output = imutils.resize(output, width=400)
    label = "{}: {:.2f}%".format(search_name_list[result[0]], (pred[0][result[0]]) * 100)
    cv2.putText(output, label, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
    cv2.imshow("Output", output)
    cv2.waitKey(0)  # 키 입력 대기 시간 (무한대기)


'''
#######################################시각화
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,6))
plt.title(OPTI)      

plt.subplot(2,1,1)  #2행 1열중 첫번째
plt.plot(loss, marker='.', c='red', label='loss')
plt.plot(val_loss, marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2)  #2행 1열중 두번째
plt.plot(acc, marker='.', c='red')
plt.plot(val_acc, marker='.', c='blue')
plt.grid()

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['accuracy','val_accuracy'])

plt.show()
#######################################
'''
  
        





            








