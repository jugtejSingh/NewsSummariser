import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore
import inference
import news
import secondAImodel
import re

dataframe = news.gettingNews()
print(dataframe.head(5))
dataframe["categories"] = dataframe["articles"].apply(lambda x: secondAImodel.gettingCategories(x))


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("News4Everyone")
        self.setGeometry(0, 0, 500, 900)

        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(central_widget)
        self.setCentralWidget(self.scroll_area)

        # Layouts
        layoutForWholeApp = QVBoxLayout(central_widget)
        layoutForWholeApp.setContentsMargins(0, 0, 0, 0)
        layoutForTop = QHBoxLayout()
        layoutForTop.setContentsMargins(0, 0, 0, 0)
        self.layoutForTitlesAndArticles = QVBoxLayout()
        self.layoutForTitlesAndArticles.setContentsMargins(0, 0, 0, 0)
        layoutForButtons = QHBoxLayout()
        layoutForButtons.setContentsMargins(0, 0, 0, 0)

        # name of app
        appName = QLabel("News4Everyone")
        appName.setFixedHeight(50)
        appName.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # the dropdown menu for business and shit
        self.comboBox = QComboBox()
        self.comboBox.addItems(pd.unique(dataframe["categories"]).tolist())
        self.comboBox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.comboBox.setFixedHeight(30)
        # Adding widgets to the top layout
        layoutForTop.addStretch(1)
        layoutForTop.addWidget(appName)
        layoutForTop.addSpacing(120)
        layoutForTop.addWidget(self.comboBox)
        # for the image
        self.image = QLabel(self)
        pixmap = QPixmap('twitter-modal.jpg').scaled(480, 350)
        self.image.setPixmap(pixmap)
        self.image.resize(pixmap.width(), pixmap.height())
        self.image.setAlignment(Qt.AlignCenter)
        # Titles
        self.titles = QLabel("")
        self.titles.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.titles.setAlignment(Qt.AlignCenter)
        self.titles.setWordWrap(True)
        # Articles
        #Make title and article multiline
        self.articles = QLabel("")
        self.articles.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.articles.setAlignment(Qt.AlignCenter)
        self.articles.setWordWrap(True)


        self.summary_button = QPushButton()
        self.summary_button.setText("Show Summary")
        self.summary_button.clicked.connect(self.show_summary)
        self.summary_button.setFixedSize(120,40)


        self.summary = QLabel("")
        self.summary.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.articles.setAlignment(Qt.AlignCenter)
        self.summary.setWordWrap(True)

        self.combo_box_change()

        #Buttons
        forward = QPushButton()
        forward.setText("Next Article")
        forward.clicked.connect(self.forward)
        forward.setFixedSize(120,40)
        #CSS FOR BUTTONS
        #Summary button CSS
        self.summary_button.setStyleSheet("background-color: white;border-radius: 20px; margin-bottom: 20px")
        forward.setStyleSheet("background-color: white;border-radius: 20px; margin-bottom: 20px")

        layoutForButtons.addWidget(self.summary_button)
        layoutForButtons.addWidget(forward)
        # adding to the whole layout
        self.layoutForTitlesAndArticles.addWidget(self.image)
        self.layoutForTitlesAndArticles.addWidget(self.titles,Qt.AlignTop)
        self.layoutForTitlesAndArticles.addSpacing(50)
        self.layoutForTitlesAndArticles.addWidget(self.articles, Qt.AlignTop)
        self.layoutForTitlesAndArticles.addSpacing(50)
        self.layoutForTitlesAndArticles.addWidget(self.summary, Qt.AlignTop)
        self.layoutForTitlesAndArticles.addSpacing(50)
        layoutForWholeApp.addLayout(layoutForTop)
        layoutForWholeApp.addLayout(self.layoutForTitlesAndArticles)
        layoutForWholeApp.addLayout(layoutForButtons)
        layoutForWholeApp.addStretch(1)

        print(self.comboBox.currentText())
        self.comboBox.currentIndexChanged.connect(self.combo_box_change)

        self.show()


    def show_summary(self):
        article = self.articles.text()
        print(article)
        preproped_article = inference.preprocess_new_article(article)
        summary = inference.inference(preproped_article)
        self.summary.setText(summary)


    def combo_box_change(self):
        selected_category = self.comboBox.currentText()
        filtered_df = dataframe[dataframe["categories"] == selected_category]
        self.current_row = 0
        self.summary.setText("")

        if not filtered_df.empty:
            self.update_article_display(filtered_df)

    def forward(self):
        selected_category = self.comboBox.currentText()
        filtered_df = dataframe[dataframe["categories"] == selected_category]
        if not filtered_df.empty:
            self.current_row = (self.current_row + 1) % len(filtered_df)
            self.summary.setText("")
            self.update_article_display(filtered_df)

    def update_article_display(self, filtered_df):
        self.titles.setText(repr(filtered_df.iloc[self.current_row, 0]))
        sentence = repr(filtered_df.iloc[self.current_row, 1])
        sentence = re.sub(r"\\n", ' ', sentence)
        sentence = re.sub(r"\\s", ' ', sentence)
        sentence = re.sub(r"\\1\\", ' ', sentence)
        sentence = re.sub(r'\\n\\n', ' ', sentence)
        sentence = re.sub(r'\\', '', sentence)
        sentence = re.sub(r"'", '', sentence)

        self.articles.setText(sentence)
        imagesData = filtered_df.iloc[self.current_row, 2]
        pixmap = QPixmap(imagesData).scaled(480, 350)
        self.image.setPixmap(pixmap)




App = QApplication([])

window = Window()

App.exec()

