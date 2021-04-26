from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

class Modeling:
    def __init__(self, x_train, categories, OPTI=Adam, INIT_LR = 1e-3 , FAC = 0.2):
        self.x_train = x_train
        self.categories = len(categories)
        self.OPTI = OPTI
        self.INIT_LR = INIT_LR
        self.FAC = FAC

    # modeling함수
    # 반환값: 모델
    # 모델구성과 컴파일
    def model(self):
        model = Sequential()
        model.add(Conv2D(16,(3,3),activation='relu', input_shape=self.x_train.shape[1:], padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
        model.add(BatchNormalization())
        model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3,3)))
        model.add(Dropout(0.3))
        
        model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3,3)))
        model.add(Dropout(0.3))
        
        model.add(Flatten())

        model.add(Dense(128,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(self.categories))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=self.OPTI(self.INIT_LR), metrics=['acc'])
        return model


    # callbacks함수
    # 반환값 : callbacks 요소 반환
    def callbacks(self):
        modelpath = 'C:/Study/project/data/modelcheckpoint/class_modeling_{epoch:2d}_{val_loss:.4f}.hdf5'
        er = EarlyStopping(monitor = 'val_loss',patience=10)
        mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
        lr = ReduceLROnPlateau(monitor = 'val_loss', patience=5,factor=self.FAC ,verbose=1)
        return er,mo,lr