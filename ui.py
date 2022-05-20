from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
import test


class Stats:

    def __init__(self):
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.y = None
        self.x = None
        self.raster_num = None
        self.contours_area = None
        self.ui = QUiLoader().load('windows.ui')

        self.ui.Button1.clicked.connect(self.upload_img)
        self.ui.Button2.clicked.connect(self.upload_message)

    def upload_img(self):
        try:
            self.x = int(self.ui.lineEdit_2.text())
            self.y = int(self.ui.lineEdit.text())
            self.raster_num = int(self.ui.lineEdit_3.text())
            self.contours_area = int(self.ui.lineEdit_4.text())
            # 选择图片
            filePath, _ = QFileDialog.getOpenFileName(
                self.ui,
                "选择你要上传的图片",
                r"C:\\Users\\29878\\Desktop\\project\\code",
                "图片类型 (*.png *.jpg *.bmp)"
            )

            # print(filePath, type(filePath))
            if filePath == '':
                QMessageBox.information(self.ui, '图片上传失败', '请重新上传图片')
            else:
                flag = test.upload(filePath, self.x, self.y, self.raster_num,self.contours_area)
        except:
            QMessageBox.critical(self.ui, '错误', '请输入正确的数据')

    def upload_message(self):
        try:
            self.x = int(self.ui.lineEdit_2.text())
            self.y = int(self.ui.lineEdit.text())
            self.raster_num = int(self.ui.lineEdit_3.text())
            self.contours_area = int(self.ui.lineEdit_4.text())
            QMessageBox.information(self.ui, '上传成功', '数据上传成功')
        except:
            QMessageBox.critical(self.ui, '错误', '数据上传失败')


app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()
