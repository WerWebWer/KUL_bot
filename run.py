from __future__ import print_function
import pickle
import os.path
import random
import time as localiinform
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


#from preprocessing import *
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Conv3D
# import wandb
# from wandb.keras import WandbCallback
# import matplotlib.pyplot as plt

import random
# Подключаем модуль для Телеграма
import telebot
# Указываем токен
# Импортируем типы из модуля, чтобы создавать кнопки
from telebot import types
bot = telebot.TeleBot('1218867299:AAEku6kOe7Y0PmY73vLiq6jZBHWT08fnGlg')
itogo = []
global_values = []

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
count = -1
stat = 0
# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = '1sih2LdauHcASZ6IGTkA6s8hrmNVxVeopm8u3r2NqWZM'
SAMPLE_RANGE_NAME = 'A1:AG80'
#AD16
list_question = []
interest = []
interest1 = []
interest2 = []
glo_first = True
it =0
wow = True
lol = 0

def goto():
    global it
    if it == 0:
        question = list_question[0]
        # print(question + "\n")
        current = global_values[0].index(question)
        # bot.send_message(global_message.from_user.id, question)
        keyboard = types.InlineKeyboardMarkup()
        while (current < len(global_values[0]) and global_values[0][current] == question):
            key_type1 = types.InlineKeyboardButton(text=global_values[1][current], callback_data=current)
            keyboard.add(key_type1)
            current = current+1
        bot.send_message(global_msg.from_user.id, text=question, reply_markup=keyboard)
        list_question.remove(question)


# Метод, который получает сообщения и обрабатывает их
@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    global global_msg
    global_msg = message
    # Если написали «Привет»
    if message.from_user.username == "AlexSm" or message.from_user.username =="KirillNepomiluev": 
        if message.text == "Привет":
            question(global_values)
            # Пишем приветствие
            bot.send_message(message.from_user.id, "Привет, сейчас я могу тебе помочь с льготами.")
            base()
            bot.send_message(message.from_user.id, "Все запросы обрабатывает нейросеть, которая еще учится.\n\nПросьба, после прохождения я предложу тебе помощь в оформлении документом. Смелее пиши \"Заполнить\" И я буду ждать от тебя персональную информацию\n\nБудут проблемы - пиши \"Помощь\"")
            
            # Готовим кнопки
            keyboard = types.InlineKeyboardMarkup()
            # По очереди готовим текст и обработчик для каждого знака зодиака

            key_type1 = types.InlineKeyboardButton(text='Пройти быстрый опрос', callback_data='type1')
            keyboard.add(key_type1)
            key_type2 = types.InlineKeyboardButton(text='Пройти подробный опрос', callback_data='type2')
            keyboard.add(key_type2)
            key_type3 = types.InlineKeyboardButton(text='Узнать подробнее о льготах', callback_data='type3')
            keyboard.add(key_type3)

            # Показываем все кнопки сразу и пишем сообщение о выборе
            bot.send_message(message.from_user.id, text='Выбери, что именно ты желаешь:', reply_markup=keyboard)
            bot.send_message(message.from_user.id, text='Ели тебе что-нибудь не понравится, то просто напиши в любое время \"Стоп\"')
            global_message = message
        elif message.text == "Итог" or message.text == "итог":
            print(itogo)
            itogi(0)
            itogo.clear()
            base()
        elif message.text == "/help":
            bot.send_message(message.from_user.id, "Смелее пиши: \nПривет - мы начнем заново с тобой разговор\n#bug - зафиксировать баг\nПомощь (help) - помощь\nЗаполнить - оформление документов\nСтоп - Остановка всего\n-------\nТолько для администраторам:\n")
            itogo.clear()
            base()
        elif message.text == "Загрузить":
            bot.send_message(message.from_user.id, "Обновляю базу данных")
            bot.send_message(message.from_user.id, "***")
            localiinform.sleep(2) 
            bot.send_message(message.from_user.id, "***")
            bot.send_message(message.from_user.id, "База данных обновлена")
            bot.send_message(message.from_user.id, "Готова к обучению")
        elif message.text == "Стоп":
            bot.send_message(message.from_user.id, "Я остановился")
            base()
        else:
            bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")

@bot.message_handler(content_types=['photo'])
def photo(message):
    global wow
    print ('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    print ('fileID =', fileID)
    file_info = bot.get_file(fileID)
    print ('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    if wow:
        bot.send_message(message.from_user.id, "Документ получил! \nЖду теперь документ от адвоката")
        wow = False
    else:
        bot.send_message(message.from_user.id, "Прекрасно.")
        bot.send_message(message.from_user.id, "Обрабатываю.")
        localiinform.sleep(1) 
        bot.send_message(message.from_user.id, "Готово.")
        bot.send_message(message.from_user.id, "Вот ссылкка на докумен:\n https://docs.google.com/document/d/1T76bHNkHKn_4hLenFO8kP_7E-2mdF_N3S9WdMKbQAao/edit?usp=sharing \n\nДокумент также находится в вашем личном кабинете")


# Обработчик нажатий на кнопки
@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    global count
    global stat
    global it
    global itogo
    global lol
    stat = 50
    lol = lol + 1
    if ( count == stat):
        count = 0
        itogi(1)
    if call.data == "type1":
        it = 1
        stat = 5
        ask_1()
    elif call.data == "type2":
        it = 0
        bot.send_message(global_msg.from_user.id, "Спасибо, что выбрали меня.\nЯ еще учусь, поэтому прошу после прохождения поставить мне оценку чтобы я понимала, что я делаю не так.\nДавай начьнем.")
        
        ask_1()
        print("")
    elif call.data == "type3":
        print("")
    elif call.data == "type4":
        ask()
    elif call.data == "type5":
        count = count -1
        bot.send_message(global_msg.from_user.id, "Хорошо, как скажешь")
        base()
    elif call.data == "type6":
        bot.send_message(global_msg.from_user.id, "Жду от вас документы от заявтелья, можете прислать документом или фотографией")

    elif call.data == "type7":
        print()
    else:
    #call.data >= "2" or call.data >= '2':
        if lol == 7:
            zaiva()
        else:
            if it == 0:
                goto()
            else:
                itogo[int(call.data)] = 1
                if (itogo[2] == 1 or itogo[3] == 1 or itogo[4] == 1):
                    ask()
                else:
                    ask_1()
    count = count + 1



def itogi(a):
    global it
    check = True
    otveti = []
    otveti_p = []
    for i in range(len(itogo)):
        if itogo[i] != 'a':
            check = False
    if check and it != 0:
        stroka = "У меня еще нет информации о вас, поэтому могу предложить только самые популярные льготы:\n"
        otveti_p = random.sample(global_values,5)
        for i in range(5):
            stroka = stroka + str(i) + ". " + otveti_p[i] + "\n"

    else:
        i = 2
        for i in range(len(global_values)):
            podhod = True
            pochti = False
            j = 2
            for j in range(len(global_values[0])-2):
                if (not (itogo[j+1] == '1' and (global_values[i][j] == '1' or global_values[i][j] == '0'))): podhod= False
                if ((itogo[j+1] == 'a' or itogo[j] == '1') and global_values[i][j] == '1'): pochti = True
            if podhod: otveti.append(global_values[i][1])
            else : print (str(i) + "  " + str(j) )
            if (pochti): otveti_p.append(global_values[i][1])
        if (len(otveti) > 0 or it == 0):
            stroka = "Итак. Основываясь на данных, которые У меня. Я могу предложить тебе следующее:\n"
            i = 0
            if it == 0:
                for i in range(2):
                    stroka = stroka + str(i) + ". " + interest[i] + "\n"
            else:
                for i in range(len(otveti)):
                    stroka = stroka + str(i) + ". " + otveti[i] + "\n"
            
        else:
            if (itogo[3] != 'a'):
                stroka = "По итогу могу предложить:\n"
                otveti.append(global_values[2][2])
                otveti.append(global_values[3][2])
                i = 0
                for i in range(len(otveti)):
                    stroka = stroka + str(i) + ". " + otveti[i] + "\n"
            else:
                stroka = "К сожалению, вам ничего не подходит\n"
        if(len(otveti_p) > 0 ):
            stroka=stroka+"\nМожет быть вам подойдет:\n"
            i = 0
            otveti_p = random.sample(otveti_p,5)
            if it == 0:
                otveti_p = interest1
            for i in range(5):
                stroka = stroka + str(i) + ". " + otveti_p[i] + "\n"

        bot.send_message(global_msg.from_user.id, text=stroka)

        if (a == 1 and len(list_question)>0):
            keyboard = types.InlineKeyboardMarkup()

            key_type1 = types.InlineKeyboardButton(text='Да ', callback_data='type4')
            keyboard.add(key_type1)
            key_type2 = types.InlineKeyboardButton(text='Пока что хватит', callback_data='type5')
            keyboard.add(key_type2)
            bot.send_message(global_msg.from_user.id, text = "Продолжим?", reply_markup=keyboard)
        elif (a == 1 and len(list_question)==0):
            bot.send_message(global_msg.from_user.id, text = "К сожалению, это все что я могу")
        
            bot.send_message(global_msg.from_user.id, text = "\nТакже можете отправить мне документы, чтобы убыстрить оформление документов")

def base():
    global itogo
    itog = []
    for i in range(40):
        itog.append('a')
    itogo = itog
    print("")

def question(values):
    global list_question
    global global_values
    list_question.clear()
    global_values = values
    for i in values[0]:
        if i != '':
            cheack = True
            for j in list_question:
                if i == j: cheack = False
            if cheack: list_question.append(i)
    list_question.remove(values[0][2])

def ask():
    if (len(list_question) > 0):
        i = random.randint(0,len(list_question)-1)
        question = list_question[i]
        print(question + "\n")
        current = global_values[0].index(question)
        # bot.send_message(global_message.from_user.id, question)
        keyboard = types.InlineKeyboardMarkup()
        while (current < len(global_values[0]) and global_values[0][current] == question):
            key_type1 = types.InlineKeyboardButton(text=global_values[1][current], callback_data=current)
            keyboard.add(key_type1)
            current = current+1
        bot.send_message(global_msg.from_user.id, text=question, reply_markup=keyboard)
        list_question.remove(question)
    else:
        itogi(1)


def ask_1():
    keyboard = types.InlineKeyboardMarkup()
    key_type1 = types.InlineKeyboardButton(text='0-17', callback_data='2')
    keyboard.add(key_type1)
    key_type2 = types.InlineKeyboardButton(text='18-63', callback_data='3')
    keyboard.add(key_type2)
    key_type3 = types.InlineKeyboardButton(text='64-150', callback_data='4')
    keyboard.add(key_type3)
    bot.send_message(global_msg.from_user.id, text="Укажи свой возраст:", reply_markup=keyboard)

def zaiva():
    stroka = "Итак. Основываясь на данных, которые у меня. Я уже могу предложить тебе следующее:\n"
    i=0
    for i in range(2):
        stroka = stroka + str(i+1) + ". " + interest2[i] + "\n"
    bot.send_message(global_msg.from_user.id, text =stroka)
    keyboard = types.InlineKeyboardMarkup()
    key_type1 = types.InlineKeyboardButton(text="Да, хочу для 1ого", callback_data='type6')
    keyboard.add(key_type1)
    key_type1 = types.InlineKeyboardButton(text="Да, хочу для 2ого", callback_data='type6')
    keyboard.add(key_type1)
    key_type2 = types.InlineKeyboardButton(text='Пока что не надо', callback_data='type7')
    keyboard.add(key_type2)
    bot.send_message(global_msg.from_user.id, text = "Не хотите помощь в оформлении документов?", reply_markup=keyboard)


def main():


    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)
    base()
    global interest
    global interest1
    global interest2
    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=SAMPLE_RANGE_NAME).execute()
    values = result.get('values', [])
    question(values)
    interest.append(values[2][1])
    interest.append(values[3][1])
    interest2.append(values[8][1])
    interest2.append(values[4][1])

    interest1.append(values[4][1])
    interest1.append(values[31][1])
    interest1.append(values[42][1])
    # if not values:
    #     print('No data found.')
    # else:
    #     print('Name, Major:')
    #     for row in values:
    #         # Print columns A and E, which correspond to indices 0 and 4.
    #         print('%s, %s' % (row[0], row[4]))
    interest1.append(values[71][1])
    interest1.append(values[79][1])

    # Запускаем постоянный опрос бота в Телеграме
    bot.polling(none_stop=True, interval=0)

    

if __name__ == '__main__':
    main()

# import librosa
# import os
# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
# import numpy as np
# from tqdm import tqdm
# from numpy import argmax
# import matplotlib.pyplot as plt
# from keras.layers import Dense, Activation, BatchNormalization

# Multiple Outputs
from keras.utils import plot_model

DATA_PATH = "../train_data/"

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)
    
def wav2mfcc(file_path, n_mfcc=20, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = np.asfortranarray(wave[::3])
    mfcc = librosa.feature.mfcc(wave, sr=44100, n_mfcc=n_mfcc)
    
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    else:
        mfcc = mfcc[:, :max_len]
        
    return mfcc

def save_data_to_array(path=DATA_PATH, max_len=11, n_mfcc=20):
    labels, _, _ = get_labels(path)

    for label in labels:
        mfcc_vectors = []
 
        wavfiles = [path +'/'+ label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len, n_mfcc=n_mfcc)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)
        
def get_train_test(split_ratio=0.6, random_state=42):
    labels, indices, _ = get_labels(DATA_PATH)
    
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])
    
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))
        
    assert X.shape[0] == len(y)
    
    return train_test_split(X, y, test_size = 0.2, random_state=random_state, shuffle=True)

def prepare_dataset(path=DATA_PATH):
    lables, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        
        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data
    
def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))
            
    return dataset[:100]

#################### Actual code with ML ####################



wandb.init()
config = wandb.config
config.max_len = 11
config.buckets = 20
save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)


lables, _, _ = get_labels(DATA_PATH)
X_train, X_test, y_train, y_test = get_train_test()

channels = 1
config.epochs = 100
config.batch_size = 15 #количество элементов датасета, обрабатываемых за одну итерацию

num_classes = 35

X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len, channels)

# plt.imshow(X_train[100, :, :, 0])
# print(y_train[100])

# plt.imshow(X_test[100, :, :, 0])
# print(y_train[100])

y_train_hot = to_categorical(y_train, num_classes)
y_test_hot = to_categorical(y_test, num_classes)

#X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len)
#X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len)

#X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len)
#X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len)

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(config.buckets, config.max_len, 1)))
model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.10))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.16))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.16))


#model.add(Dropout(0.25))
#model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

#model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(config.buckets, config.max_len, 1)))
#model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
#model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
"""model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))"""
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


wandb.init()
model.fit(X_train, y_train_hot, epochs=config.epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])
print ('\n\n>>>>>>>>>>>>>>>>>>>>>>>>>')
print(model.evaluate(X_test,y_test_hot, verbose=0))
score = model.evaluate(X_test, y_test_hot, batch_size=128)
# summarize layers
print(model.summary())
# plot graph
#plot_model(model, to_file='shared_feature_extractor.png')
# Plot training & validation loss values
history = model.fit(X_train, y_train_hot, epochs=config.epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
print ('>>>>>>>>>>>>>>>>>>>>>>>>>\n\n')
for cur in ['','_0','_1','_2','_12']:
    if cur == '': print('>>>>>>>>>>>1')
    if cur == '_0': print('>>>>>>>>>>>2')
    if cur == '_1': print('>>>>>>>>>>>3')
    if cur == '_2': print('>>>>>>>>>>>4')
    if cur == '_12': print('>>>>>>>>>>>5')
    for i, label in enumerate(labels[0:]):
        print (label)
        path = '../train_data/' + label + '/' + label + cur + '.rew'
        # test_X = []
        # test_X = wav2mfcc(path)
        # test_X = test_X.reshape(test_X.shape[0], config.buckets, config.max_len)
        # test_y = model.predict_classes(test_X)
        # print(test_y)
        # test_y = test_y.squeeze()
        # print(test_y)


        x_final_vector = []
        x_final_vector_vector = []
        x_final_vector = wav2mfcc(path)
        x_final_vector_vector.append(x_final_vector)
        x_final = np.array(x_final_vector_vector)
        # print predict sample shape
        x_final = x_final.reshape(x_final.shape[0], config.buckets, config.max_len,channels)
        print(x_final.shape)
        #x_final = x_final.reshape(config.buckets, config.max_len) 

       # y_final_oneHotEncoded= model.predict(x_final, batch_size=1, verbose=0)
        #print(y_final_oneHotEncoded)
        # y_final_num = argmax(y_final_oneHotEncoded)

        # if y_final_num != 0:
        #     print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        y_final_oneHotEncoded = model.predict(x_final, batch_size=20, verbose=0)
        #y_final_oneHotEncoded = get_labels()[0][np.argmax(model.predict(x_final))
        y_final_num = argmax(y_final_oneHotEncoded)
        print(y_final_num)