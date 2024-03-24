import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 640, 480)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)

        self.image_paths = ["1.jpg"]  # 图片路径列表
        self.image_index = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(1000)  # 定时器间隔，单位毫秒

    def update_image(self):
        # 加载下一张图片
        pixmap = QPixmap(self.image_paths[self.image_index])
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

        # 更新图片索引
        self.image_index = (self.image_index + 1) % len(self.image_paths)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
