# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'generator_editor_gui.ui'
##
## Created by: Qt User Interface Compiler version 6.11.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout,
    QHeaderView, QPushButton, QSizePolicy, QSpacerItem,
    QSplitter, QTableView, QVBoxLayout, QWidget)

from VeraGrid.Gui.PlotDialogue.qt_chart_widget import GraphsWidget
from VeraGrid.Gui.Icons.icons_rc import *

class Ui_GeneratorQCurveEditorDialog(object):
    def setupUi(self, GeneratorQCurveEditorDialog):
        if not GeneratorQCurveEditorDialog.objectName():
            GeneratorQCurveEditorDialog.setObjectName(u"GeneratorQCurveEditorDialog")
        GeneratorQCurveEditorDialog.resize(802, 545)
        self.verticalLayout = QVBoxLayout(GeneratorQCurveEditorDialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(GeneratorQCurveEditorDialog)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.leftFrame = QFrame(self.splitter)
        self.leftFrame.setObjectName(u"leftFrame")
        self.leftLayout = QVBoxLayout(self.leftFrame)
        self.leftLayout.setObjectName(u"leftLayout")
        self.leftLayout.setContentsMargins(0, 0, 0, 0)
        self.tableView = QTableView(self.leftFrame)
        self.tableView.setObjectName(u"tableView")

        self.leftLayout.addWidget(self.tableView)

        self.buttonsFrame = QFrame(self.leftFrame)
        self.buttonsFrame.setObjectName(u"buttonsFrame")
        self.buttonsFrame.setMaximumSize(QSize(16777215, 40))
        self.buttonsLayout = QHBoxLayout(self.buttonsFrame)
        self.buttonsLayout.setObjectName(u"buttonsLayout")
        self.buttonsLayout.setContentsMargins(0, 0, 0, 0)
        self.addRowButton = QPushButton(self.buttonsFrame)
        self.addRowButton.setObjectName(u"addRowButton")
        icon = QIcon()
        icon.addFile(u":/Icons/icons/plus.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.addRowButton.setIcon(icon)

        self.buttonsLayout.addWidget(self.addRowButton)

        self.delRowButton = QPushButton(self.buttonsFrame)
        self.delRowButton.setObjectName(u"delRowButton")
        icon1 = QIcon()
        icon1.addFile(u":/Icons/icons/minus.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.delRowButton.setIcon(icon1)

        self.buttonsLayout.addWidget(self.delRowButton)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.buttonsLayout.addItem(self.horizontalSpacer)

        self.applyButton = QPushButton(self.buttonsFrame)
        self.applyButton.setObjectName(u"applyButton")
        icon2 = QIcon()
        icon2.addFile(u":/Icons/icons/accept.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.applyButton.setIcon(icon2)

        self.buttonsLayout.addWidget(self.applyButton)


        self.leftLayout.addWidget(self.buttonsFrame)

        self.splitter.addWidget(self.leftFrame)
        self.rightFrame = QFrame(self.splitter)
        self.rightFrame.setObjectName(u"rightFrame")
        self.rightLayout = QVBoxLayout(self.rightFrame)
        self.rightLayout.setObjectName(u"rightLayout")
        self.rightLayout.setContentsMargins(0, 0, 0, 0)
        self.plotter = GraphsWidget(self.rightFrame)
        self.plotter.setObjectName(u"plotter")

        self.rightLayout.addWidget(self.plotter)

        self.splitter.addWidget(self.rightFrame)

        self.verticalLayout.addWidget(self.splitter)


        self.retranslateUi(GeneratorQCurveEditorDialog)

        QMetaObject.connectSlotsByName(GeneratorQCurveEditorDialog)
    # setupUi

    def retranslateUi(self, GeneratorQCurveEditorDialog):
        GeneratorQCurveEditorDialog.setWindowTitle(QCoreApplication.translate("GeneratorQCurveEditorDialog", u"Reactive power curve editor", None))
#if QT_CONFIG(tooltip)
        self.addRowButton.setToolTip(QCoreApplication.translate("GeneratorQCurveEditorDialog", u"Add entry", None))
#endif // QT_CONFIG(tooltip)
        self.addRowButton.setText("")
#if QT_CONFIG(tooltip)
        self.delRowButton.setToolTip(QCoreApplication.translate("GeneratorQCurveEditorDialog", u"Delete selected", None))
#endif // QT_CONFIG(tooltip)
        self.delRowButton.setText("")
#if QT_CONFIG(tooltip)
        self.applyButton.setToolTip(QCoreApplication.translate("GeneratorQCurveEditorDialog", u"Apply curve to Qmin, Qmax, Pmin, Pmax", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.applyButton.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.applyButton.setText("")
    # retranslateUi

