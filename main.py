import tkinter as tk
import random
from string import punctuation
from tkinter import filedialog
from PyQt5 import QtCore, QtGui, QtWidgets
import tensorflow as tf
import numpy as np
import os
import pickle
import tqdm
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import threading
import time

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


class Ui_MainWindow(object):
    def __init__(self):
        self.funktion = None
        self.sequenzlangesave = None
        self.unikechars = 0
        self.char2int = {}
        self.int2char = {}
        self.model = None
        self.vocablenint = 85
        self.funktion = 3
        self.pathtohp = "data/hp.txt"

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 724)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 10, 801, 661))
        self.tabWidget.setObjectName("tabWidget")
        self.Maintab = QtWidgets.QWidget()
        self.Maintab.setObjectName("Maintab")
        self.frame = QtWidgets.QFrame(self.Maintab)
        self.frame.setGeometry(QtCore.QRect(0, 0, 801, 631))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.textEditgenerated = QtWidgets.QTextEdit(self.frame)
        self.textEditgenerated.setGeometry(QtCore.QRect(410, 30, 381, 421))
        self.textEditgenerated.setObjectName("textEditgenerated")
        self.inputtext = QtWidgets.QTextEdit(self.frame)
        self.inputtext.setGeometry(QtCore.QRect(10, 30, 381, 421))
        self.inputtext.setObjectName("inputtext")
        self.inputtext.textChanged.connect(self.updateinputtextinfos)
        self.labelinput = QtWidgets.QLabel(self.frame)
        self.labelinput.setGeometry(QtCore.QRect(20, 10, 55, 16))
        self.labelinput.setObjectName("labelinput")
        self.progressBar_7 = QtWidgets.QProgressBar(self.frame)
        self.progressBar_7.setGeometry(QtCore.QRect(10, 460, 118, 23))
        self.progressBar_7.setProperty("value", 24)
        self.progressBar_7.setObjectName("progressBar_7")
        self.progressBar_8 = QtWidgets.QProgressBar(self.frame)
        self.progressBar_8.setGeometry(QtCore.QRect(10, 490, 118, 23))
        self.progressBar_8.setProperty("value", 24)
        self.progressBar_8.setObjectName("progressBar_8")
        self.progressBar_9 = QtWidgets.QProgressBar(self.frame)
        self.progressBar_9.setGeometry(QtCore.QRect(10, 520, 118, 23))
        self.progressBar_9.setProperty("value", 24)
        self.progressBar_9.setObjectName("progressBar_9")
        self.progressBar = QtWidgets.QProgressBar(self.frame)
        self.progressBar.setGeometry(QtCore.QRect(10, 550, 118, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.labelanalyzetext = QtWidgets.QLabel(self.frame)
        self.labelanalyzetext.setGeometry(QtCore.QRect(140, 460, 121, 21))
        self.labelanalyzetext.setObjectName("labelanalyzetext")
        self.labeldurchlaufepoche = QtWidgets.QLabel(self.frame)
        self.labeldurchlaufepoche.setGeometry(QtCore.QRect(140, 520, 111, 21))
        self.labeldurchlaufepoche.setObjectName("labeldurchlaufepoche")
        self.labeltrain = QtWidgets.QLabel(self.frame)
        self.labeltrain.setGeometry(QtCore.QRect(140, 490, 101, 16))
        self.labeltrain.setObjectName("labeltrain")
        self.lineepochen = QtWidgets.QLineEdit(self.frame)
        self.lineepochen.setGeometry(QtCore.QRect(380, 460, 113, 22))
        self.lineepochen.setObjectName("lineepochen")
        self.buffersize = QtWidgets.QLineEdit(self.frame)
        self.buffersize.setGeometry(QtCore.QRect(380, 550, 113, 22))
        self.buffersize.setObjectName("buffersize")
        self.buffersize.setText(str(10000))
        self.linesequenzlange = QtWidgets.QLineEdit(self.frame)
        self.linesequenzlange.setGeometry(QtCore.QRect(380, 490, 113, 22))
        self.linesequenzlange.setObjectName("linesequenzlange")
        self.tempregler = QtWidgets.QSlider(self.frame)
        self.tempregler.setGeometry(QtCore.QRect(375, 600, 360, 22))
        self.tempregler.setOrientation(QtCore.Qt.Horizontal)
        self.tempregler.setObjectName("tempregler")
        self.tempregler.valueChanged.connect(self.tempreglerfunk)
        self.templabel = QtWidgets.QLabel(self.frame)
        self.templabel.setGeometry(QtCore.QRect(560, 560, 91, 21))
        self.templabel.setObjectName("templabel")
        self.temperaturfeld = QtWidgets.QLineEdit(self.frame)
        self.temperaturfeld.setGeometry(QtCore.QRect(650, 560, 113, 22))
        self.temperaturfeld.setObjectName("temperaturfeld")
        self.temperaturfeld.textChanged.connect(self.tempfeldfunk)
        self.checkbox = QtWidgets.QCheckBox(self.frame)
        self.checkbox.setGeometry(QtCore.QRect(600, 460, 131, 20))
        self.checkbox.setObjectName("checkbox")
        self.checkboxlower = QtWidgets.QCheckBox(self.frame)
        self.checkboxlower.setGeometry(QtCore.QRect(600, 490, 131, 20))
        self.checkboxlower.setObjectName("checkbox")
        self.checkboxlower.setText("Lowercase")
        self.labeloutput = QtWidgets.QLabel(self.frame)
        self.labeloutput.setGeometry(QtCore.QRect(410, 10, 55, 16))
        self.labeloutput.setObjectName("labeloutput")
        self.buttonlearn = QtWidgets.QPushButton(self.frame)
        self.buttonlearn.setGeometry(QtCore.QRect(520, 520, 93, 28))
        self.buttonlearn.setObjectName("buttonlearn")
        self.buttonlearn.clicked.connect(self.learn)
        self.buttongenerate = QtWidgets.QPushButton(self.frame)
        self.buttongenerate.setGeometry(QtCore.QRect(650, 520, 93, 28))
        self.buttongenerate.setObjectName("buttongenerate")
        self.buttongenerate.clicked.connect(self.generate)
        self.labelgeneratetext = QtWidgets.QLabel(self.frame)
        self.labelgeneratetext.setGeometry(QtCore.QRect(140, 550, 101, 16))
        self.labelgeneratetext.setObjectName("labelgeneratetext")
        self.labelepochen = QtWidgets.QLabel(self.frame)
        self.labelepochen.setGeometry(QtCore.QRect(280, 460, 91, 21))
        self.labelepochen.setObjectName("labelepochen")
        self.labelsequenzlang = QtWidgets.QLabel(self.frame)
        self.labelsequenzlang.setGeometry(QtCore.QRect(280, 490, 91, 20))
        self.labelsequenzlang.setObjectName("labelsequenzlang")
        self.linebatchsize = QtWidgets.QLineEdit(self.frame)
        self.linebatchsize.setGeometry(QtCore.QRect(380, 520, 113, 22))
        self.linebatchsize.setObjectName("linebatchsize")
        self.labelbatchsize = QtWidgets.QLabel(self.frame)
        self.labelbatchsize.setGeometry(QtCore.QRect(280, 520, 91, 20))
        self.labelbatchsize.setObjectName("labelbatchsize")
        self.buffsizelabel = QtWidgets.QLabel(self.frame)
        self.buffsizelabel.setGeometry(QtCore.QRect(280, 550, 91, 20))
        self.buffsizelabel.setObjectName("buffsizelabel")
        self.buffsizelabel.setText("buffsizelabel")
        self.buttonselecttext = QtWidgets.QPushButton(self.frame)
        self.buttonselecttext.setGeometry(QtCore.QRect(160, 0, 93, 28))
        self.buttonselecttext.setObjectName("buttonselecttext")
        self.buttonselecttext.clicked.connect(self.selecttextfunk)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(20, 580, 81, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(20, 606, 91, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(180, 580, 55, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(180, 605, 81, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setGeometry(QtCore.QRect(90, 580, 55, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setGeometry(QtCore.QRect(100, 605, 55, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.frame)
        self.label_7.setGeometry(QtCore.QRect(230, 580, 71, 21))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.frame)
        self.label_8.setGeometry(QtCore.QRect(270, 606, 71, 20))
        self.label_8.setObjectName("label_8")
        self.tabWidget.addTab(self.Maintab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.pushButton = QtWidgets.QPushButton(self.tab_2)
        self.pushButton.setGeometry(QtCore.QRect(30, 30, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.updatefunktion)
        self.settingsinteration = QtWidgets.QLineEdit(self.tab_2)
        self.settingsinteration.setGeometry(QtCore.QRect(150, 30, 113, 31))
        self.settingsinteration.setObjectName("lineEdit")
        self.outputlenth = QtWidgets.QLineEdit(self.tab_2)
        self.outputlenth.setGeometry(QtCore.QRect(150, 80, 113, 31))
        self.outputlenth.setObjectName("outlenthline")
        self.outputlenth.setText("1000")
        self.labeloutputlen = QtWidgets.QLabel(self.tab_2)
        self.labeloutputlen.setGeometry(QtCore.QRect(50, 80, 150, 30))
        self.labeloutputlen.setObjectName("labeafdsl_8")
        self.labeloutputlen.setText("Outputlenght")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionTEst = QtWidgets.QAction(MainWindow)
        self.actionTEst.setObjectName("actionTEst")
        self.actionSettings = QtWidgets.QAction(MainWindow)
        self.actionSettings.setObjectName("actionSettings")
        self.actionMain_windows = QtWidgets.QAction(MainWindow)
        self.actionMain_windows.setObjectName("actionMain_windows")
        self.progressBar.setMinimum(0)
        self.progressBar.setValue(0)
        self.progressBar_7.setMinimum(0)
        self.progressBar_7.setValue(0)
        self.progressBar_8.setMinimum(0)
        self.progressBar_8.setValue(0)
        self.progressBar_9.setMinimum(0)
        self.progressBar_9.setValue(0)
        self.lineepochen.setText(str(1))
        self.linesequenzlange.setText(str(100))
        self.linebatchsize.setText(str(128))
        self.checkbox.setChecked(True)
        self.tempregler.setMaximum(1000)
        self.temperaturfeld.setText("0")
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def updatefunktion(self):
        #print("updatefuntion")
        try:
            a = self.settingsinteration.text()
            b = int(a)
            if (b == 2 or b == 3 or b == 4 or b == 5):
                self.funktion = b
            else:
                self.funktion = 0
                self.settingsinteration.setText("3")
        except:
            self.funktion = 0
            print("An error occured")
            self.settingsinteration.setText("3")
        print(self.funktion, "ist die ausgewählte funktion")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.labelinput.setText(_translate("MainWindow", "Input"))
        self.labelanalyzetext.setText(_translate("MainWindow", "Analysiere Text"))
        self.labeldurchlaufepoche.setText(_translate("MainWindow", "Durchlaufe Epoche"))
        self.labeltrain.setText(_translate("MainWindow", "Trainiere AI"))
        self.templabel.setText(_translate("MainWindow", "Temperature :"))
        self.checkbox.setText(_translate("MainWindow", "Satzzeichen"))
        self.labeloutput.setText(_translate("MainWindow", "Output"))
        self.buttonlearn.setText(_translate("MainWindow", "Learn "))
        self.buttongenerate.setText(_translate("MainWindow", "Generate"))
        self.labelgeneratetext.setText(_translate("MainWindow", "Generiere Text"))
        self.labelepochen.setText(_translate("MainWindow", "Epochen"))
        self.labelsequenzlang.setText(_translate("MainWindow", "Sequenzlänge"))
        self.labelbatchsize.setText(_translate("MainWindow", "Batch-Size"))
        self.buttonselecttext.setText(_translate("MainWindow", "Select text"))
        self.label.setText(_translate("MainWindow", "Charakter: "))
        self.label_2.setText(_translate("MainWindow", "Unique Char:"))
        self.label_3.setText(_translate("MainWindow", "Words :"))
        self.label_4.setText(_translate("MainWindow", "Random num"))
        self.label_5.setText(_translate("MainWindow", "TextLabel"))
        self.label_6.setText(_translate("MainWindow", "TextLabel"))
        self.label_7.setText(_translate("MainWindow", "TextLabel"))
        self.label_8.setText(_translate("MainWindow", "TextLabel"))
        self.settingsinteration.setText("3")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Maintab), _translate("MainWindow", "MainTab"))
        self.pushButton.setText(_translate("MainWindow", "Apply"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Settings"))
        self.actionTEst.setText(_translate("MainWindow", "TEst"))
        self.actionSettings.setText(_translate("MainWindow", "Settings"))
        self.actionMain_windows.setText(_translate("MainWindow", "Main windows"))

    def updateinputtextinfos(self):

        print("updated")
        text = self.inputtext.toPlainText()
        if (len(text) > 0):
            n_chars = len(text)
            unique_chars = sorted(list(set(text)))
            a = text.split(" ")
            self.label_5.setText(str(n_chars))
            self.label_6.setText(str(len(unique_chars)))
            self.label_7.setText(str(len(a)))

            self.label_8.setText(str(random.randint(0, 123456789)))
        else:
            self.label_5.setText("0")
            self.label_6.setText("0")
            self.label_7.setText("0")
            self.label_8.setText("0")

    def learn(self):
        f = open("data/data.txt",mode="w",encoding="utf-8")
        f.write(self.inputtext.toPlainText(),)
        f.close()
        print("file copy finisched")
        if (self.funktion == 2):
            t = threading.Thread(target=self.learn2, daemon=True)
            t.start()
        elif (self.funktion == 3):
            t = threading.Thread(target=self.learn3, daemon=True)
            t.start()
        elif (self.funktion == 4):
            t = threading.Thread(target=self.learn4, daemon=True)
            t.start()
        elif (self.funktion == 5):
            t = threading.Thread(target=self.learn5, daemon=True)
            t.start()
        else:
            print("No funktion selected")


    def learn4(self):
        try:
            sequence_length = int(self.linesequenzlange.text())
            batchsize = int(self.linebatchsize.text())
            epochen = int(self.lineepochen.text())
            temperatur = float(self.tempregler.value() / 1000)
            bufsize = int(self.buffersize.text())
            learningrate = temperatur
        except:
            self.lineepochen.setText(str(50))
            self.linesequenzlange.setText(str(100))
            self.linebatchsize.setText(str(128))
            self.textEditgenerated.append("Bitte gültige Werte eingeben")
            return
        self.sequenzlangesave = sequence_length
        sequence_length = sequence_length
        BATCH_SIZE = batchsize
        EPOCHS = epochen
        # dataset file path
        FILE_PATH = "data/hp.txt"
        # FILE_PATH = "data/python_code.py"
        BASENAME = os.path.basename(FILE_PATH)

        text = self.inputtext.toPlainText()
        if (len(text) <= sequence_length * batchsize):
            #print("fehlern fehler fehler")
            self.inputtext.append("bitte text eingeben")
            #print("ffffffffffffffff")
            return
        if (self.checkboxlower.isChecked()):
            text = text.lower()
        if (self.checkbox.isChecked()):
            text = text.translate(str.maketrans("", "", punctuation))
        n_chars = len(text)
        vocab = ''.join(sorted(set(text)))
        print("unique_chars:", vocab)
        n_unique_chars = len(vocab)
        print("Number of characters:", n_chars)
        print("Number of unique characters:", n_unique_chars)

        char2int = {c: i for i, c in enumerate(vocab)}
        int2char = {i: c for i, c in enumerate(vocab)}

        pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
        pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))

        encoded_text = np.array([char2int[c] for c in text])
        char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

        for char in char_dataset.take(8):
            print(char.numpy(), int2char[char.numpy()])

        # build sequences by batching
        sequences = char_dataset.batch(2 * sequence_length + 1, drop_remainder=True)

        # print sequences
        for sequence in sequences.take(2):
            print(''.join([int2char[i] for i in sequence.numpy()]))

        def split_sample(sample):
            ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
            for i in range(1, (len(sample) - 1) // 2):
                input_ = sample[i: i + sequence_length]
                target = sample[i + sequence_length]
                other_ds = tf.data.Dataset.from_tensors((input_, target))
                ds = ds.concatenate(other_ds)
            return ds

        dataset = sequences.flat_map(split_sample)

        def one_hot_samples(input_, target):
            return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

        dataset = dataset.map(one_hot_samples)
        for element in dataset.take(2):
            print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
            print("Target:", int2char[np.argmax(element[1].numpy())])
            print("Input shape:", element[0].shape)
            print("Target shape:", element[1].shape)
            print("=" * 50, "\n")

        ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)

        model = Sequential([
            LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
            Dropout(0.1),
            LSTM(256, return_sequences=True),
            Dropout(0.1),
            LSTM(256),
            Dense(n_unique_chars, activation="softmax"),
        ])


        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        if not os.path.isdir("results"):
            os.mkdir("results")
        model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS)
        model.save(f"results/{BASENAME}-{sequence_length}.h5")
        print("finischend")

    def learn3(self):
        input_example_batch = 0
        try:
            sequence_length = int(self.linesequenzlange.text())
            batchsize = int(self.linebatchsize.text())
            epochen = int(self.lineepochen.text())
            temperatur = float(self.tempregler.value() / 1000)
            bufsize = int(self.buffersize.text())
            learningrate = temperatur
        except:
            self.lineepochen.setText(str(50))
            self.linesequenzlange.setText(str(100))
            self.linebatchsize.setText(str(128))
            self.textEditgenerated.append("Bitte gültige Werte eingeben")
            return
        self.progressBar.setValue(0)

        # self.progressBar_9.setMaximum((len(encoded_text) - sequence_length) // BATCH_SIZE)
        if (learningrate == 0):
            learningrate = 0.001
        self.progressBar_7.setMaximum(10)
        text = self.inputtext.toPlainText()
        if (len(text) <= sequence_length * batchsize):
            self.inputtext.append("bitte text eingeben")
            return
        if (self.checkboxlower.isChecked()):
            text = text.lower()
        if (self.checkbox.isChecked()):
            text = text.translate(str.maketrans("", "", punctuation))
        print("anzahl Zeichen:", len(text))
        vocab = sorted(set(text))
        print(f'{len(vocab)} unique characters', vocab)

        example_texts = ['abcdefg', 'xyz']
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
        print(chars, "chars form ids codes thing somethingg ")

        ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(vocab))
        print(len(ids_from_chars.get_vocabulary()), "länge vocab sure")

        self.char2int = ids_from_chars
        ids = ids_from_chars(chars)
        print(ids)
        chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=ids_from_chars.get_vocabulary(),
            invert=True)
        print(chars_from_ids, "chars form id ")
        chars = chars_from_ids(ids)
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        tf.strings.reduce_join(chars, axis=-1).numpy()


        def text_from_ids(ids):
            return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        print(all_ids, "alle ids hopefully good ")
        ids_dataset = tf.data.Dataset.from_tensor_slices(
            all_ids)  # in ids_dataset ist der gesamte text codiert in integer dargelegt

        seq_length = sequence_length  # wie viele zeichen man beachten muss
        examples_per_epoch = len(text) // (
                seq_length + 1)  # doppelted dividieren heist division ohne komma ,rest wird verworfen
        print(examples_per_epoch, "examples per epoch ")
        self.progressBar_9.setMaximum(examples_per_epoch // batchsize)
        sequences = ids_dataset.batch(seq_length + 1,
                                      drop_remainder=True)  # man kann viel mehr sequenzen erstelln wenn man will
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        def split_input_target(
                sequence):  # teilt ein segment auf in input vo der letzte fehlt und output( target) wo das erste fehlt dafür der letzt steht
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)


        BATCH_SIZE = batchsize  # wie viele dinger getestet werden bevor sachen geupdated werden r
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        BUFFER_SIZE = bufsize

        dataset = (
            dataset
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))
        vocab_size = len(vocab)
        embedding_dim = 256
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        # Number of RNN units
        rnn_units = 1024
        model = MyModel(
            vocab_size=len(ids_from_chars.get_vocabulary()),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units)
        print(len(ids_from_chars.get_vocabulary()), "len idform char ")
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        for input_example_batch, target_example_batch in dataset.take(1):
            print("in der fore schleife")
            example_batch_predictions = model(input_example_batch)
            print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        print("bitte bitte funktioniere")
        model.summary()
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        print("huso")
        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

        print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
        print()
        print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

        example_batch_loss = loss(target_example_batch, example_batch_predictions)
        mean_loss = example_batch_loss.numpy().mean()
        print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
        print("Mean loss:        ", mean_loss)
        tf.exp(mean_loss).numpy()
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        print("leaarning rage : ", learningrate)
        model.compile(optimizer="adam", loss=loss)
        c=CustomCallback(self.progressBar_8,self.progressBar_9,epochen,len(text)//sequence_length//batchsize-1)
        model.fit(dataset,epochs=epochen,callbacks=c)
        EPOCHS = epochen
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        self.model = model
        self.int2char = chars_from_ids
        print("ende")

    def learn2(self):
        try:
            sequence_length = int(self.linesequenzlange.text())
            BATCH_SIZE = int(self.linebatchsize.text())
            EPOCHS = int(self.lineepochen.text())
            temp = float(self.tempregler.value() / 100)
        except:
            self.lineepochen.setText(str(50))
            self.linesequenzlange.setText(str(100))
            self.linebatchsize.setText(str(128))
            self.textEditgenerated.append("Bitte gültige Werte eingeben ")
            return
        self.progressBar.setValue(0)
        self.progressBar_7.setValue(0)
        self.progressBar_8.setValue(0)
        self.progressBar_9.setValue(0)
        self.progressBar_8.setMaximum(EPOCHS)
        self.progressBar_7.setMaximum(10)
        text = self.inputtext.toPlainText()
        print("sucsessuflly read", self.progressBar.value())
        print("vor inputtext")
        if (len(text) <= sequence_length * BATCH_SIZE):
            print("fehlern fehler fehler")
            self.inputtext.append("bitte text eingeben")
            print("ffffffffffffffff")
            return
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        BASENAME = os.path.basename(self.pathtohp)
        print("lost")

        if (self.checkboxlower.isChecked()):
            text = text.lower()
        if (self.checkbox.isChecked()):
            text = text.translate(str.maketrans("", "", punctuation))
        print(text, "testadtadfa", len(text))

        print("fehler")
        # print some stats
        n_chars = len(text)
        unique_chars = sorted(list(set(text)))
        vocab = ''.join(sorted(set(text)))
        print("unique_chars:", vocab)
        n_unique_chars = len(vocab)
        print("Number of characters:", n_chars)
        print("Number of unique characters:", n_unique_chars)
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        # dictionary that converts characters to integers
        char2int = {c: i for i, c in enumerate(unique_chars)}
        # dictionary that converts integers to characters
        int2char = {i: c for i, c in enumerate(unique_chars)}
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        print("char to int", char2int)
        print("int 2char ", int2char)
        # save these dictionaries for later generation
        pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
        pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))
        print("after pickle")
        encoded_text = np.array([char2int[c] for c in text])
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        # print first 5 characters
        print("after dingends hoffentlich")
        for char in char_dataset.take(8):
            print(char.numpy(), int2char[char.numpy()])
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        # build sequences by batching
        sequences = char_dataset.batch(2 * sequence_length + 1, drop_remainder=True)

        # print sequences
        for sequence in sequences.take(2):
            print(''.join([int2char[i] for i in sequence.numpy()]))

        def split_sample(sample):
            ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
            for i in range(1, (len(sample) - 1) // 2):
                input_ = sample[i: i + sequence_length]
                target = sample[i + sequence_length]
                # extend the dataset with these samples by concatenate() method
                other_ds = tf.data.Dataset.from_tensors((input_, target))
                ds = ds.concatenate(other_ds)
            return ds

        print("Flat map", sequences, split_sample)
        dataset = sequences.flat_map(split_sample)
        print("value errror be flat map")
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        print("progress bar update")

        def one_hot_samples(input_, target):
            # onehot encode the inputs and the targets
            return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

        print("one hot function")
        dataset = dataset.map(one_hot_samples)
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        # print first 2 samples
        for element in dataset.take(2):
            print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
            print("Target:", int2char[np.argmax(element[1].numpy())])
            print("Input shape:", element[0].shape)
            print("Target shape:", element[1].shape)
            print("=" * 50, "\n")

        ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)
        self.progressBar_9.setMaximum((len(encoded_text) - sequence_length) // BATCH_SIZE)
        model = Sequential()
        model.add(LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True))
        model.add(Dropout(temp))
        model.add(LSTM(256))
        model.add(Dense(n_unique_chars, activation="softmax"))
        model.summary()
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        # make results folder if does not exist yet
        print("model created")
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        print("results")
        c = [CustomCallback(self.progressBar_9, self.progressBar_8)]
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS,
                  use_multiprocessing=True, workers=5)

        model.save_weights(f"modelweight.h5")
        self.unikechars = unique_chars
        self.char2int = char2int
        self.int2char = int2char
        self.model = model

    def learn5(self):
        print("learn5")
        self.progressBar_7.setMaximum(10)
        try:
            sl = int(self.linesequenzlange.text())
            bs = int(self.linebatchsize.text())
            e = int(self.lineepochen.text())
            temperatur = float(self.tempregler.value() / 1000)
            buffs = int(self.buffersize.text())
            l = int(self.outputlenth.text())
        except:
            self.lineepochen.setText(str(50))
            self.linesequenzlange.setText(str(100))
            self.linebatchsize.setText(str(128))
            self.textEditgenerated.append("Bitte gültige Werte eingeben")
            return
        sequence_length = sl
        BATCH_SIZE = bs
        EPOCHS = e
        # dataset file path
        FILE_PATH = "data/data.txt"
        BASENAME = os.path.basename(FILE_PATH)
        # read the data
        text = open(FILE_PATH, encoding="utf-8").read()
        if (len(text) <= sequence_length * BATCH_SIZE):
            self.inputtext.append("bitte text eingeben")
            return
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        if (self.checkboxlower.isChecked()):
            text = text.lower()
        if (self.checkbox.isChecked()):
            text = text.translate(str.maketrans("", "", punctuation))
        n_chars = len(text)
        vocab = ''.join(sorted(set(text)))
        unique_chars = vocab
        print("unique_chars:", vocab)
        n_unique_chars = len(vocab)
        print("Number of characters:", n_chars)
        print("Number of unique characters:", n_unique_chars)
        # dictionary that converts characters to integers
        char2int = {c: i for i, c in enumerate(unique_chars)}
        # dictionary that converts integers to characters
        int2char = {i: c for i, c in enumerate(unique_chars)}
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
        pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))
        encoded_text = np.array([char2int[c] for c in text])
        char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        for char in char_dataset.take(8):  # test output
            print(char.numpy(), int2char[char.numpy()])
        sequences = char_dataset.batch(2 * sequence_length + 1, drop_remainder=True)
        for sequence in sequences.take(2):  # test output sequenzen
            print(''.join([int2char[i] for i in sequence.numpy()]))
            print("sequenzene")
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)

        def split_sample(sample):
            ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
            for i in range(1, (len(sample) - 1) // 2):
                input_ = sample[i: i + sequence_length]
                target = sample[i + sequence_length]
                # extend the dataset with these samples by concatenate() method
                other_ds = tf.data.Dataset.from_tensors((input_, target))
                ds = ds.concatenate(other_ds)
            return ds

        dataset = sequences.flat_map(split_sample)
        print("after flatmap")
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)

        def one_hot_samples(input_, target):
            # onehot encode the inputs and the targets
            return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        dataset = dataset.map(one_hot_samples)
        for element in dataset.take(2):
            print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
            print("Target:", int2char[np.argmax(element[1].numpy())])
            print("Input shape:", element[0].shape)
            print("Target shape:", element[1].shape)
        ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        model = Sequential([
            LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
            Dropout(temperatur),
            LSTM(256, return_sequences=True),
            Dropout(temperatur),
            LSTM(265),
            Dense(n_unique_chars, activation="softmax"),
        ])
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        if not os.path.isdir("results"):
            os.mkdir("results")
        # train the model
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)

        c=CustomCallback(self.progressBar_8, self.progressBar_9,EPOCHS,(len(encoded_text) - sequence_length) // BATCH_SIZE)
        model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS,
                  callbacks=c)
        model.save(f"results/{BASENAME}-{sequence_length}.h5")
        print("ende")

    def generate(self):
        if (self.funktion == 2):
            t = threading.Thread(target=self.generate2, daemon=True)
            t.start()
        elif (self.funktion == 3):
            t = threading.Thread(target=self.generate3(), daemon=True)
            t.start()
        elif (self.funktion == 4):
            t = threading.Thread(target=self.generate4(), daemon=True)
            t.start()
        elif(self.funktion==5):
            t = threading.Thread(target=self.generate5(), daemon=True)
            t.start()
            print("No funktion selected generator")

    def generate4(self):
        try:
            sequence_length = int(self.linesequenzlange.text())
            batchsize = int(self.linebatchsize.text())
            epochen = int(self.lineepochen.text())
            temperatur = float(self.tempregler.value() / 1000)
            bufsize = int(self.buffersize.text())
            learningrate = temperatur
            lenght = int(self.outputlenth.text())
        except:
            self.lineepochen.setText(str(50))
            self.linesequenzlange.setText(str(100))
            self.linebatchsize.setText(str(128))
            self.textEditgenerated.append("Bitte gültige Werte eingeben")
            return
        # dataset file path
        FILE_PATH = "data/hp.txt"
        # FILE_PATH = "data/python_code.py"
        BASENAME = os.path.basename(FILE_PATH)
        # load vocab dictionaries
        char2int = pickle.load(open(f"{BASENAME}-char2int.pickle", "rb"))
        int2char = pickle.load(open(f"{BASENAME}-int2char.pickle", "rb"))

        sequence_length = sequence_length
        vocab_size = len(char2int)

        # building the model
        model = Sequential([
            LSTM(256, input_shape=(sequence_length, vocab_size), return_sequences=True),
            Dropout(temperatur),
            LSTM(256, return_sequences=True),
            Dropout(temperatur),
            LSTM(256),
            Dense(vocab_size, activation="softmax"),
        ])

        # load the optimal weights
        model.load_weights(f"results/{BASENAME}-{sequence_length}.h5")
        # specify the feed to first characters to generate
        seed = "stoffner"
        s = seed
        n_chars = 400
        # generate 400 characters
        generated = seed + ""
        for i in tqdm.tqdm(range(n_chars), "Generating text"):
            # make the input sequence
            X = np.zeros((1, sequence_length, vocab_size))
            for t, char in enumerate(seed):
                X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
            # predict the next character
            predicted = model.predict(X, verbose=0)[0]
            # converting the vector to an integer
            next_index = np.argmax(predicted)
            # converting the integer to a character
            next_char = int2char[next_index]
            # add the character to results
            generated += next_char
            # shift seed and the predicted character
            seed = seed[1:] + next_char

        print("Seed:", s)
        print("Generated text:")
        print(generated)
        self.textEditgenerated.setText(generated)

    def generate5(self):
        try:
            sl = int(self.linesequenzlange.text())
            bs = int(self.linebatchsize.text())
            e = int(self.lineepochen.text())
            temperatur = float(self.tempregler.value() / 1000)
            bs = int(self.buffersize.text())
            br = temperatur
            l = int(self.outputlenth.text())
        except:
            self.lineepochen.setText(str(50))
            self.linesequenzlange.setText(str(100))
            self.linebatchsize.setText(str(128))
            self.textEditgenerated.append("Bitte gültige Werte eingeben")
            return
        sequence_length = sl
        # dataset file path
        FILE_PATH = "data/data.txt"
        # FILE_PATH = "data/python_code.py"
        BASENAME = os.path.basename(FILE_PATH)

        seed = "seed"
        char2int = pickle.load(open(f"{BASENAME}-char2int.pickle", "rb"))
        int2char = pickle.load(open(f"{BASENAME}-int2char.pickle", "rb"))
        vocab_size = len(char2int)
        model = Sequential([
            LSTM(256, input_shape=(sequence_length, vocab_size), return_sequences=True),
            Dropout(0.1),
            LSTM(256, return_sequences=True),
            Dropout(0.1),
            Dense(vocab_size, activation="softmax"),
        ])
        model.load_weights(f"results/{BASENAME}-{sequence_length}.h5")
        s = seed
        generated = ""
        for i in tqdm.tqdm(range(l), "Generating text"):
            # make the input sequence
            X = np.zeros((1, sequence_length, vocab_size))
            for t, char in enumerate(seed):
                X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
            # predict the next character
            predicted = model.predict(X, verbose=0)[0]
            # converting the vector to an integer
            next_index = np.argmax(predicted)
            # converting the integer to a character
            next_char = int2char[next_index]
            # add the character to results
            generated += next_char
            # shift seed and the predicted character
            seed = seed[1:] + next_char

        print("Seed:", s)
        print("Generated text:")
        print(generated)
        self.textEditgenerated.setText(generated)

    def generate3(self):
        try:
            lenght = int(self.outputlenth.text())
        except:
            print("bitte richtige werte eingeben")
            return
        chars_from_ids = self.int2char
        ids_from_chars = self.char2int
        self.progressBar_7.setValue(self.progressBar_7.value() + 1)
        if (self.model != None):
            model = self.model
        else:
            try:
                model = MyModel()
                model.compile(optimizer="adam", loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))
                model.load_weights("weights.ckpt")
            except:
                print("Error loadint model")
                return
        print("generator started")
        try:
            ranger = self.settingsinteration.text()
            print(ranger)
            ranger = int(ranger)
        except:
            self.settingsinteration.setText("1000")

        print("fertig generiert oder geladen")
        one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
        start = time.time()
        states = None

        next_char = tf.constant(["Haus"])
        result = [next_char]

        for n in range(ranger):  # lange
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)
            self.progressBar.setValue(n)
        result = tf.strings.join(result)
        end = time.time()
        exitcode = result[0].numpy().decode('utf-8')
        print(exitcode, '\n\n' + '_' * 80)
        print('\nRun time:', end - start)
        self.textEditgenerated.append(exitcode)

    def generate2(self):
        FILE_PATH = "virus.txt"
        BASENAME = os.path.basename(FILE_PATH)
        n_unique_chars = self.unikechars
        char2int = self.char2int
        int2char = self.int2char
        try:
            sequence_length = int(self.linesequenzlange.text())
            BATCH_SIZE = int(self.linebatchsize.text())
            EPOCHS = int(self.lineepochen.text())
            temp = float(self.tempregler.value() / 100)
            if (temp == 1):
                temp = temp - 0.01
        except:
            self.lineepochen.setText(str(1))
            self.linesequenzlange.setText(str(100))
            self.linebatchsize.setText(str(128))
            self.textEditgenerated.append("Bitte gültige Werte eingeben")
            return
        if (self.model == None):
            self.textEditgenerated.append("Bitte model zuerst trainieren")
            return

        print("generate button", n_unique_chars)
        """model = Sequential()
        model.add(LSTM(256, input_shape=(sequence_length, len(n_unique_chars)), return_sequences=True))
        #model.add(tf.keras.layers.GaussianNoise(temp))
        model.add(Dropout(temp))
        model.add(LSTM(256, use_bias=True, return_sequences=True))
        model.add(Dropout(temp))
        model.add(Dense(256))
        model.add(LSTM(256))
        model.add(Dense(len(n_unique_chars), activation="softmax"))
        model.summary()
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        """
        model = self.model
        text = self.inputtext.toPlainText()
        if (self.checkboxlower.isChecked()):
            text = text.lower()
        if (self.checkbox.isChecked()):
            text = text.translate(str.maketrans("", "", punctuation))
        a = text.split(" ")
        seed = a[random.randint(0, len(a))]
        print("Seed", seed)
        print("model created")
        # load the optimal weights
        # model.load_weights(f"modelweight.h5")
        print("after loading")
        s = seed
        n_chars = 400
        # generate 400 characters
        vocab_size = len(char2int)
        generated = ""
        generated += seed
        print("vor for")
        print("cahr to int", char2int)
        print("int to cahr,", int2char)
        for i in tqdm.tqdm(range(n_chars), "Generating text"):
            # make the input sequence
            X = np.zeros((1, sequence_length, vocab_size))
            for t, char in enumerate(seed):
                X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
            print("x after :", X)
            predicted2 = model.predict(X, verbose=1)
            predicted = predicted2[0]
            next_index = np.argmax(predicted)
            print(next_index)
            next_char = int2char[next_index]
            if (i > 3):
                if (generated[-1] == generated[-2] == next_char):
                    i += -1
            generated += next_char
            # shift seed and the predicted character
            seed = seed[1:] + next_char

        print("Seed:", s)
        print("Generated text:")
        print("[" + generated + "]")
        self.textEditgenerated.append(generated + " ENDE")

    def selecttextfunk(self):
        print("select text")
        root = tk.Tk()
        root.withdraw()
        files = filedialog.askopenfilenames()
        print("file:", files)
        textlist = []
        for i in range(len(files)):
            print(files[i])
            f = open(files[i], encoding="utf8")
            self.text = f.read()
            textlist.append(self.text)
        self.inputtext.setText("\n----\n".join(textlist))

    def tempreglerfunk(self):
        self.temperaturfeld.setText(str(self.tempregler.value()))

    def tempfeldfunk(self):
        try:
            self.tempregler.setValue(int(self.temperaturfeld.text()))
        except:
            self.tempregler.setValue(0)
        if (int(self.temperaturfeld.text()) > 1000):
            self.tempregler.setValue(0)
            self.temperaturfeld.setText(str(1000))


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, pboben, unten,max1,max2):
        super().__init__()
        self.pb1 = pboben
        self.pb2 = unten
        self.pb2.setMaximum(max2)
        self.pb1.setMaximum(max1)
    """
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))
    """

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        self.pb1.setValue(epoch + 1)

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        #print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
        self.pb2.setValue(batch + 1)


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        if return_state:
            return x, states
        else:
            return x


class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.2):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "" or "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        print(inputs, "inputs lost")
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        print(input_chars)
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        print(input_ids, "input ids")
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        print(states, "------", predicted_logits)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
