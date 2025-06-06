# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'rms_model_editor.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout,
    QHeaderView, QListView, QSizePolicy, QSplitter,
    QTableView, QVBoxLayout, QWidget)
from .icons_rc import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(841, 518)
        icon = QIcon()
        icon.addFile(u":/Icons/icons/dyn.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        self.actionCopy_to_clipboard = QAction(MainWindow)
        self.actionCopy_to_clipboard.setObjectName(u"actionCopy_to_clipboard")
        icon1 = QIcon()
        icon1.addFile(u":/Icons/icons/copy.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionCopy_to_clipboard.setIcon(icon1)
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName(u"actionSave")
        icon2 = QIcon()
        icon2.addFile(u":/Icons/icons/import_profiles.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionSave.setIcon(icon2)
        self.verticalLayout_2 = QVBoxLayout(MainWindow)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.splitter_3 = QSplitter(MainWindow)
        self.splitter_3.setObjectName(u"splitter_3")
        self.splitter_3.setOrientation(Qt.Orientation.Horizontal)
        self.frame_8 = QFrame(self.splitter_3)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setMaximumSize(QSize(400, 16777215))
        self.frame_8.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_8.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.frame_8)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(-1, 8, -1, -1)
        self.listView = QListView(self.frame_8)
        self.listView.setObjectName(u"listView")

        self.verticalLayout_7.addWidget(self.listView)

        self.splitter_3.addWidget(self.frame_8)
        self.PlotFrame = QFrame(self.splitter_3)
        self.PlotFrame.setObjectName(u"PlotFrame")
        self.PlotFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.PlotFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.PlotFrame)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(8, 8, 8, 8)
        self.tableView = QTableView(self.PlotFrame)
        self.tableView.setObjectName(u"tableView")

        self.horizontalLayout.addWidget(self.tableView)

        self.splitter_3.addWidget(self.PlotFrame)

        self.verticalLayout.addWidget(self.splitter_3)


        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"RMS Model editor", None))
        self.actionCopy_to_clipboard.setText(QCoreApplication.translate("MainWindow", u"Copy to clipboard", None))
        self.actionSave.setText(QCoreApplication.translate("MainWindow", u"Save", None))
    # retranslateUi

