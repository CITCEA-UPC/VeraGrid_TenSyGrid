# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dynamic_model_comparison_page_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.11.0
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSplitter, QToolBar, QTreeView,
    QVBoxLayout, QWidget)

from VeraGrid.Gui.spread_sheet_table import SpreadsheetTableView
from VeraGrid.Gui.Icons.icons_rc import *

class Ui_DynamicModelComparisonPage(object):
    def setupUi(self, DynamicModelComparisonPage):
        if not DynamicModelComparisonPage.objectName():
            DynamicModelComparisonPage.setObjectName(u"DynamicModelComparisonPage")
        DynamicModelComparisonPage.resize(1000, 600)
        self.actionSave = QAction(DynamicModelComparisonPage)
        self.actionSave.setObjectName(u"actionSave")
        icon = QIcon()
        icon.addFile(u":/Icons/icons/savec.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionSave.setIcon(icon)
        self.mainLayout = QVBoxLayout(DynamicModelComparisonPage)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setObjectName(u"mainLayout")
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.comparisonToolBar = QToolBar(DynamicModelComparisonPage)
        self.comparisonToolBar.setObjectName(u"comparisonToolBar")
        self.comparisonToolBar.setMovable(False)
        self.comparisonToolBar.setIconSize(QSize(24, 24))
        self.comparisonToolBar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.comparisonToolBar.setFloatable(False)

        self.mainLayout.addWidget(self.comparisonToolBar)

        self.statusLabel = QLabel(DynamicModelComparisonPage)
        self.statusLabel.setObjectName(u"statusLabel")
        self.statusLabel.setVisible(False)
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setMargin(4)

        self.mainLayout.addWidget(self.statusLabel)

        self.comparisonSplitter = QSplitter(DynamicModelComparisonPage)
        self.comparisonSplitter.setObjectName(u"comparisonSplitter")
        self.comparisonSplitter.setOrientation(Qt.Orientation.Horizontal)
        self.comparisonSplitter.setChildrenCollapsible(True)
        self.modelsFrame = QFrame(self.comparisonSplitter)
        self.modelsFrame.setObjectName(u"modelsFrame")
        self.modelsFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.modelsLayout = QVBoxLayout(self.modelsFrame)
        self.modelsLayout.setSpacing(0)
        self.modelsLayout.setObjectName(u"modelsLayout")
        self.modelsLayout.setContentsMargins(0, 0, 3, 0)
        self.searchFrame = QFrame(self.modelsFrame)
        self.searchFrame.setObjectName(u"searchFrame")
        self.searchFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.searchLayout = QHBoxLayout(self.searchFrame)
        self.searchLayout.setObjectName(u"searchLayout")
        self.searchLayout.setContentsMargins(0, 4, 0, 4)
        self.searchLineEdit = QLineEdit(self.searchFrame)
        self.searchLineEdit.setObjectName(u"searchLineEdit")

        self.searchLayout.addWidget(self.searchLineEdit)

        self.searchButton = QPushButton(self.searchFrame)
        self.searchButton.setObjectName(u"searchButton")
        self.searchButton.setMaximumSize(QSize(32, 16777215))
        icon1 = QIcon()
        icon1.addFile(u":/Icons/icons/magnifying_glass.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.searchButton.setIcon(icon1)
        self.searchButton.setFlat(True)

        self.searchLayout.addWidget(self.searchButton)


        self.modelsLayout.addWidget(self.searchFrame)

        self.modelsTreeView = QTreeView(self.modelsFrame)
        self.modelsTreeView.setObjectName(u"modelsTreeView")
        self.modelsTreeView.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked|QAbstractItemView.EditTrigger.EditKeyPressed|QAbstractItemView.EditTrigger.SelectedClicked)

        self.modelsLayout.addWidget(self.modelsTreeView)

        self.comparisonSplitter.addWidget(self.modelsFrame)
        self.tableFrame = QFrame(self.comparisonSplitter)
        self.tableFrame.setObjectName(u"tableFrame")
        self.tableFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.tableLayout = QVBoxLayout(self.tableFrame)
        self.tableLayout.setSpacing(0)
        self.tableLayout.setObjectName(u"tableLayout")
        self.tableLayout.setContentsMargins(3, 4, 0, 0)
        self.comparisonTableView = SpreadsheetTableView(self.tableFrame)
        self.comparisonTableView.setObjectName(u"comparisonTableView")
        self.comparisonTableView.setAlternatingRowColors(True)
        self.comparisonTableView.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.comparisonTableView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)

        self.tableLayout.addWidget(self.comparisonTableView)

        self.comparisonSplitter.addWidget(self.tableFrame)

        self.mainLayout.addWidget(self.comparisonSplitter)


        self.comparisonToolBar.addAction(self.actionSave)

        self.retranslateUi(DynamicModelComparisonPage)

        QMetaObject.connectSlotsByName(DynamicModelComparisonPage)
    # setupUi

    def retranslateUi(self, DynamicModelComparisonPage):
        DynamicModelComparisonPage.setWindowTitle(QCoreApplication.translate("DynamicModelComparisonPage", u"Dynamic Model Comparison", None))
        self.actionSave.setText(QCoreApplication.translate("DynamicModelComparisonPage", u"Save", None))
#if QT_CONFIG(tooltip)
        self.actionSave.setToolTip(QCoreApplication.translate("DynamicModelComparisonPage", u"Save all model comparison changes", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionSave.setShortcut(QCoreApplication.translate("DynamicModelComparisonPage", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.statusLabel.setText("")
        self.searchLineEdit.setPlaceholderText(QCoreApplication.translate("DynamicModelComparisonPage", u"Search device or model types", None))
#if QT_CONFIG(tooltip)
        self.searchButton.setToolTip(QCoreApplication.translate("DynamicModelComparisonPage", u"Filter model types", None))
#endif // QT_CONFIG(tooltip)
        self.searchButton.setText("")
#if QT_CONFIG(tooltip)
        self.modelsTreeView.setToolTip(QCoreApplication.translate("DynamicModelComparisonPage", u"Select a model type to compare its devices", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi
