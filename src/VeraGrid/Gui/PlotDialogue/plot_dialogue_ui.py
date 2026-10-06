# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'plot_dialogue.ui'
##
## Created by: Qt User Interface Compiler version 6.11.2
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
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QHeaderView,
    QLineEdit, QMainWindow, QPushButton, QSizePolicy,
    QSplitter, QTabWidget, QToolBar, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget)

from VeraGrid.Gui.PlotDialogue.qt_chart_widget import GraphsWidget
from VeraGrid.Gui.Icons import icons_rc

class Ui_PlotDialogue(object):
    def setupUi(self, PlotDialogue):
        if not PlotDialogue.objectName():
            PlotDialogue.setObjectName(u"PlotDialogue")
        PlotDialogue.resize(900, 620)
        self.actionSaveImage = QAction(PlotDialogue)
        self.actionSaveImage.setObjectName(u"actionSaveImage")
        icon = QIcon()
        icon.addFile(u":/Icons/icons/savec.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionSaveImage.setIcon(icon)
        self.actionResetView = QAction(PlotDialogue)
        self.actionResetView.setObjectName(u"actionResetView")
        icon1 = QIcon()
        icon1.addFile(u":/Icons/icons/resize.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionResetView.setIcon(icon1)
        self.actionopencloseTree = QAction(PlotDialogue)
        self.actionopencloseTree.setObjectName(u"actionopencloseTree")
        self.actionopencloseTree.setCheckable(True)
        icon2 = QIcon()
        icon2.addFile(u":/Icons/icons/tree.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionopencloseTree.setIcon(icon2)
        self.actionopencloseTree.setMenuRole(QAction.MenuRole.NoRole)
        self.actionAddPlot = QAction(PlotDialogue)
        self.actionAddPlot.setObjectName(u"actionAddPlot")
        icon3 = QIcon()
        icon3.addFile(u":/Icons/icons/plus.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionAddPlot.setIcon(icon3)
        self.centralwidget = QWidget(PlotDialogue)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.plotToolBar = QToolBar(self.centralwidget)
        self.plotToolBar.setObjectName(u"plotToolBar")
        self.plotToolBar.setMovable(False)
        self.plotToolBar.setFloatable(False)

        self.verticalLayout.addWidget(self.plotToolBar)

        self.plotSplitter = QSplitter(self.centralwidget)
        self.plotSplitter.setObjectName(u"plotSplitter")
        self.plotSplitter.setOrientation(Qt.Orientation.Horizontal)
        self.seriesSelectorFrame = QFrame(self.plotSplitter)
        self.seriesSelectorFrame.setObjectName(u"seriesSelectorFrame")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.seriesSelectorFrame.sizePolicy().hasHeightForWidth())
        self.seriesSelectorFrame.setSizePolicy(sizePolicy)
        self.seriesSelectorFrame.setMinimumSize(QSize(220, 0))
        self.seriesSelectorFrame.setVisible(True)
        self.seriesSelectorFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.seriesSelectorFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.seriesSelectorLayout = QVBoxLayout(self.seriesSelectorFrame)
        self.seriesSelectorLayout.setObjectName(u"seriesSelectorLayout")
        self.frame = QFrame(self.seriesSelectorFrame)
        self.frame.setObjectName(u"frame")
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QFrame.Shape.NoFrame)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.seriesSearchLineEdit = QLineEdit(self.frame)
        self.seriesSearchLineEdit.setObjectName(u"seriesSearchLineEdit")

        self.horizontalLayout.addWidget(self.seriesSearchLineEdit)

        self.selectAllButton = QPushButton(self.frame)
        self.selectAllButton.setObjectName(u"selectAllButton")
        icon4 = QIcon()
        icon4.addFile(u":/Icons/icons/check_all.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.selectAllButton.setIcon(icon4)

        self.horizontalLayout.addWidget(self.selectAllButton)

        self.selectNoneButton = QPushButton(self.frame)
        self.selectNoneButton.setObjectName(u"selectNoneButton")
        icon5 = QIcon()
        icon5.addFile(u":/Icons/icons/uncheck_all.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.selectNoneButton.setIcon(icon5)

        self.horizontalLayout.addWidget(self.selectNoneButton)


        self.seriesSelectorLayout.addWidget(self.frame)

        self.seriesTreeWidget = QTreeWidget(self.seriesSelectorFrame)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1")
        self.seriesTreeWidget.setHeaderItem(__qtreewidgetitem)
        self.seriesTreeWidget.setObjectName(u"seriesTreeWidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.seriesTreeWidget.sizePolicy().hasHeightForWidth())
        self.seriesTreeWidget.setSizePolicy(sizePolicy1)
        self.seriesTreeWidget.setFrameShape(QFrame.Shape.NoFrame)
        self.seriesTreeWidget.setHeaderHidden(True)

        self.seriesSelectorLayout.addWidget(self.seriesTreeWidget)

        self.plotSplitter.addWidget(self.seriesSelectorFrame)
        self.plotTabs = QTabWidget(self.plotSplitter)
        self.plotTabs.setObjectName(u"plotTabs")
        self.plotTabs.setTabPosition(QTabWidget.TabPosition.South)
        self.plotTabs.setTabsClosable(True)
        self.plotTab = QWidget()
        self.plotTab.setObjectName(u"plotTab")
        self.plotTabLayout = QVBoxLayout(self.plotTab)
        self.plotTabLayout.setObjectName(u"plotTabLayout")
        self.plotTabLayout.setContentsMargins(0, 0, 0, 0)
        self.plotwidget = GraphsWidget(self.plotTab)
        self.plotwidget.setObjectName(u"plotwidget")

        self.plotTabLayout.addWidget(self.plotwidget)

        self.plotTabs.addTab(self.plotTab, "")
        self.plotSplitter.addWidget(self.plotTabs)

        self.verticalLayout.addWidget(self.plotSplitter)

        PlotDialogue.setCentralWidget(self.centralwidget)

        self.plotToolBar.addAction(self.actionopencloseTree)
        self.plotToolBar.addSeparator()
        self.plotToolBar.addAction(self.actionAddPlot)
        self.plotToolBar.addAction(self.actionSaveImage)
        self.plotToolBar.addAction(self.actionResetView)

        self.retranslateUi(PlotDialogue)

        QMetaObject.connectSlotsByName(PlotDialogue)
    # setupUi

    def retranslateUi(self, PlotDialogue):
        PlotDialogue.setWindowTitle(QCoreApplication.translate("PlotDialogue", u"Plot", None))
        self.actionSaveImage.setText(QCoreApplication.translate("PlotDialogue", u"Save image", None))
#if QT_CONFIG(tooltip)
        self.actionSaveImage.setToolTip(QCoreApplication.translate("PlotDialogue", u"Save the current plot as SVG or PNG", None))
#endif // QT_CONFIG(tooltip)
        self.actionResetView.setText(QCoreApplication.translate("PlotDialogue", u"Center data", None))
#if QT_CONFIG(tooltip)
        self.actionResetView.setToolTip(QCoreApplication.translate("PlotDialogue", u"Reset zoom and pan to show all data", None))
#endif // QT_CONFIG(tooltip)
        self.actionopencloseTree.setText(QCoreApplication.translate("PlotDialogue", u"Series list", None))
#if QT_CONFIG(tooltip)
        self.actionopencloseTree.setToolTip(QCoreApplication.translate("PlotDialogue", u"Show or hide the series list", None))
#endif // QT_CONFIG(tooltip)
        self.actionAddPlot.setText(QCoreApplication.translate("PlotDialogue", u"Add plot", None))
#if QT_CONFIG(tooltip)
        self.actionAddPlot.setToolTip(QCoreApplication.translate("PlotDialogue", u"Add another plot tab", None))
#endif // QT_CONFIG(tooltip)
        self.seriesSearchLineEdit.setPlaceholderText(QCoreApplication.translate("PlotDialogue", u"Search series", None))
#if QT_CONFIG(tooltip)
        self.selectAllButton.setToolTip(QCoreApplication.translate("PlotDialogue", u"Select all series", None))
#endif // QT_CONFIG(tooltip)
        self.selectAllButton.setText("")
#if QT_CONFIG(tooltip)
        self.selectNoneButton.setToolTip(QCoreApplication.translate("PlotDialogue", u"Select no series", None))
#endif // QT_CONFIG(tooltip)
        self.selectNoneButton.setText("")
        self.plotTabs.setTabText(self.plotTabs.indexOf(self.plotTab), QCoreApplication.translate("PlotDialogue", u"Plot", None))
    # retranslateUi
