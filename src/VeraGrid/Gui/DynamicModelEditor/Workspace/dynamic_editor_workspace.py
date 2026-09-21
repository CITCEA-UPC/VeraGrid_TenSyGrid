# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dynamic_editor_workspace.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
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
    QLineEdit, QMainWindow, QSizePolicy, QSplitter,
    QToolBar, QTreeView, QVBoxLayout, QWidget)
from VeraGrid.Gui.Icons.icons_rc import *

class Ui_DynamicEditorWorkspaceWindow(object):
    def setupUi(self, DynamicEditorWorkspaceWindow):
        if not DynamicEditorWorkspaceWindow.objectName():
            DynamicEditorWorkspaceWindow.setObjectName(u"DynamicEditorWorkspaceWindow")
        DynamicEditorWorkspaceWindow.resize(1094, 603)
        self.actionview_tree = QAction(DynamicEditorWorkspaceWindow)
        self.actionview_tree.setObjectName(u"actionview_tree")
        icon = QIcon()
        icon.addFile(u":/Icons/icons/tree.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionview_tree.setIcon(icon)
        self.actionview_tree.setMenuRole(QAction.MenuRole.NoRole)
        self.actionRMS_Editor = QAction(DynamicEditorWorkspaceWindow)
        self.actionRMS_Editor.setObjectName(u"actionRMS_Editor")
        icon1 = QIcon()
        icon1.addFile(u":/Icons/icons/dyn.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionRMS_Editor.setIcon(icon1)
        self.actionRMS_Editor.setMenuRole(QAction.MenuRole.NoRole)
        self.actionEMT_Editor = QAction(DynamicEditorWorkspaceWindow)
        self.actionEMT_Editor.setObjectName(u"actionEMT_Editor")
        icon2 = QIcon()
        icon2.addFile(u":/Icons/icons/dyn_emt.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionEMT_Editor.setIcon(icon2)
        self.actionEMT_Editor.setMenuRole(QAction.MenuRole.NoRole)
        self.actionRMS_Events = QAction(DynamicEditorWorkspaceWindow)
        self.actionRMS_Events.setObjectName(u"actionRMS_Events")
        icon3 = QIcon()
        icon3.addFile(u":/Icons/icons/dyn_edit.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionRMS_Events.setIcon(icon3)
        self.actionRMS_Events.setMenuRole(QAction.MenuRole.NoRole)
        self.actionEMT_Events = QAction(DynamicEditorWorkspaceWindow)
        self.actionEMT_Events.setObjectName(u"actionEMT_Events")
        icon4 = QIcon()
        icon4.addFile(u":/Icons/icons/dyn_emt_edit.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionEMT_Events.setIcon(icon4)
        self.actionEMT_Events.setMenuRole(QAction.MenuRole.NoRole)
        self.actionRMS_Plots = QAction(DynamicEditorWorkspaceWindow)
        self.actionRMS_Plots.setObjectName(u"actionRMS_Plots")
        icon5 = QIcon()
        icon5.addFile(u":/Icons/icons/rms_plots.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionRMS_Plots.setIcon(icon5)
        self.actionRMS_Plots.setMenuRole(QAction.MenuRole.NoRole)
        self.actionEMT_Plots = QAction(DynamicEditorWorkspaceWindow)
        self.actionEMT_Plots.setObjectName(u"actionEMT_Plots")
        icon6 = QIcon()
        icon6.addFile(u":/Icons/icons/emt_plots.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionEMT_Plots.setIcon(icon6)
        self.actionEMT_Plots.setMenuRole(QAction.MenuRole.NoRole)
        self.actionRMS_Compare = QAction(DynamicEditorWorkspaceWindow)
        self.actionRMS_Compare.setObjectName(u"actionRMS_Compare")
        icon7 = QIcon()
        icon7.addFile(u":/Icons/icons/rms_compare.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionRMS_Compare.setIcon(icon7)
        self.actionRMS_Compare.setMenuRole(QAction.MenuRole.NoRole)
        self.actionEMT_Compare = QAction(DynamicEditorWorkspaceWindow)
        self.actionEMT_Compare.setObjectName(u"actionEMT_Compare")
        icon8 = QIcon()
        icon8.addFile(u":/Icons/icons/emt_compare.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionEMT_Compare.setIcon(icon8)
        self.actionEMT_Compare.setMenuRole(QAction.MenuRole.NoRole)
        self.centralwidget = QWidget(DynamicEditorWorkspaceWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.treeFrame = QFrame(self.splitter)
        self.treeFrame.setObjectName(u"treeFrame")
        self.treeFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.treeFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.treeFrame)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 4, 6, 0)
        self.frame_2 = QFrame(self.treeFrame)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(6, 0, 0, 0)
        self.searchInTreeLineEdit = QLineEdit(self.frame_2)
        self.searchInTreeLineEdit.setObjectName(u"searchInTreeLineEdit")

        self.horizontalLayout_2.addWidget(self.searchInTreeLineEdit)


        self.verticalLayout_2.addWidget(self.frame_2)

        self.treeView = QTreeView(self.treeFrame)
        self.treeView.setObjectName(u"treeView")
        self.treeView.setFrameShape(QFrame.Shape.NoFrame)

        self.verticalLayout_2.addWidget(self.treeView)

        self.splitter.addWidget(self.treeFrame)
        self.editorFrame = QFrame(self.splitter)
        self.editorFrame.setObjectName(u"editorFrame")
        self.editorFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.editorFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.editorFrame)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(6, 0, 0, 0)
        self.editorFrameLayout = QVBoxLayout()
        self.editorFrameLayout.setObjectName(u"editorFrameLayout")

        self.verticalLayout_3.addLayout(self.editorFrameLayout)

        self.splitter.addWidget(self.editorFrame)

        self.verticalLayout.addWidget(self.splitter)

        DynamicEditorWorkspaceWindow.setCentralWidget(self.centralwidget)
        self.toolBar = QToolBar(DynamicEditorWorkspaceWindow)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setMovable(False)
        self.toolBar.setAllowedAreas(Qt.ToolBarArea.LeftToolBarArea)
        self.toolBar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolBar.setFloatable(False)
        DynamicEditorWorkspaceWindow.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionview_tree)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRMS_Editor)
        self.toolBar.addAction(self.actionEMT_Editor)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRMS_Events)
        self.toolBar.addAction(self.actionEMT_Events)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRMS_Plots)
        self.toolBar.addAction(self.actionEMT_Plots)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRMS_Compare)
        self.toolBar.addAction(self.actionEMT_Compare)

        self.retranslateUi(DynamicEditorWorkspaceWindow)

        QMetaObject.connectSlotsByName(DynamicEditorWorkspaceWindow)
    # setupUi

    def retranslateUi(self, DynamicEditorWorkspaceWindow):
        DynamicEditorWorkspaceWindow.setWindowTitle(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"Dynamic Editor Workspace", None))
        self.actionview_tree.setText(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"view tree", None))
        self.actionRMS_Editor.setText(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"RMS Editor", None))
#if QT_CONFIG(tooltip)
        self.actionRMS_Editor.setToolTip(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"Open the RMS editor for the selected device", None))
#endif // QT_CONFIG(tooltip)
        self.actionEMT_Editor.setText(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"EMT Editor", None))
#if QT_CONFIG(tooltip)
        self.actionEMT_Editor.setToolTip(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"Open the EMT editor for the selected device", None))
#endif // QT_CONFIG(tooltip)
        self.actionRMS_Events.setText(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"RMS Events", None))
#if QT_CONFIG(tooltip)
        self.actionRMS_Events.setToolTip(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"Open the RMS events editor", None))
#endif // QT_CONFIG(tooltip)
        self.actionEMT_Events.setText(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"EMT Events", None))
#if QT_CONFIG(tooltip)
        self.actionEMT_Events.setToolTip(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"Open the EMT events editor", None))
#endif // QT_CONFIG(tooltip)
        self.actionRMS_Plots.setText(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"RMS Plots", None))
#if QT_CONFIG(tooltip)
        self.actionRMS_Plots.setToolTip(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"Open the RMS plots editor", None))
#endif // QT_CONFIG(tooltip)
        self.actionEMT_Plots.setText(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"EMT Plots", None))
#if QT_CONFIG(tooltip)
        self.actionEMT_Plots.setToolTip(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"Open the EMT plots editor", None))
#endif // QT_CONFIG(tooltip)
        self.actionRMS_Compare.setText(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"RMS Compare", None))
#if QT_CONFIG(tooltip)
        self.actionRMS_Compare.setToolTip(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"Compare saved RMS models and parameters", None))
#endif // QT_CONFIG(tooltip)
        self.actionEMT_Compare.setText(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"EMT Compare", None))
#if QT_CONFIG(tooltip)
        self.actionEMT_Compare.setToolTip(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"Compare saved EMT models and parameters", None))
#endif // QT_CONFIG(tooltip)
        self.searchInTreeLineEdit.setPlaceholderText(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"Type to search the device", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("DynamicEditorWorkspaceWindow", u"toolBar", None))
    # retranslateUi

