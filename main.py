import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QGroupBox, QSizePolicy, QSpacerItem, QLineEdit, QSlider
)
from module import basic, emoji
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageQt
import os


class mainclass(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DCGAN Emoji Generator")
        self.initUI()
        self.net = emoji.emoji()

        self.InputImg = None

    def initUI(self):
        gLayout = QHBoxLayout(self)
        gLayout.addWidget(self.CreatGroup_button())


    def CreatGroup_button(self):
        group = QGroupBox("DcGAN")
        layout = QVBoxLayout(group)

        self.create_btn_showimg(layout)
        self.create_btn_showMod(layout)
        self.create_btn_showLoss(layout)
        self.create_btn_Inference(layout)
        return group
    
    def create_btn_showimg(self, layout):
        btn = QPushButton("1.Show Training Images")
        layout.addWidget(btn)
        def event():
            self.net.show_example()
            pass
        btn.clicked.connect(event)

    def create_btn_showMod(self, layout):
        btn = QPushButton("2. Show Model Structure")
        layout.addWidget(btn)
        def event():
            self.net.print_model()
            pass
        btn.clicked.connect(event)

    def create_btn_showLoss(self, layout):
        btn = QPushButton("3. Show Training Loss")
        layout.addWidget(btn)
        def event():
            self.net.train()
            pass
        btn.clicked.connect(event)

    def create_btn_Inference(self, layout):
        btn = QPushButton("4. Inference")
        layout.addWidget(btn)
        def event():
            self.net.show_gen()
            self.net.train_animation()
            pass
        btn.clicked.connect(event)




    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainclass()
    window.show()
    sys.exit(app.exec())

