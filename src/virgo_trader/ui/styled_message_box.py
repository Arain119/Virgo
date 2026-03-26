"""Styled message box dialog for the Qt UI.

Provides a custom look-and-feel (shadow/animations) while preserving
QMessageBox-like behavior for confirmations and alerts.
"""

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)


class StyledMessageBox(QDialog):
    """现代化的自定义消息框，支持毛玻璃效果和流畅动画"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # 优先使用顶层窗口作为定位参照，避免子控件过小或隐藏导致位置异常
        if parent is not None and parent.window() is not None:
            self.parent_widget = parent.window()
        else:
            self.parent_widget = parent
        self.result_button = None

        # 设置窗口属性
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )

        # Windows特定属性
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.setObjectName("styledMessageBox")
        self.setModal(True)

        # 拖动相关
        self.old_pos = None

        self.setup_ui()
        self.setup_animations()
        self.setup_shadow_effect()

    def setup_ui(self):
        """设置现代化UI"""
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 主容器 - 用于毛玻璃效果和阴影
        self.main_container = QFrame()
        self.main_container.setObjectName("styledMessageBoxContainer")
        self.main_container.setStyleSheet("background-color: transparent;")

        container_layout = QVBoxLayout(self.main_container)
        container_layout.setContentsMargins(20, 15, 20, 20)
        container_layout.setSpacing(10)

        # 内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        # 图标标签
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(50, 50)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("background-color: transparent;")
        content_layout.addWidget(self.icon_label)

        # 文本区域
        text_area = QVBoxLayout()
        text_area.setSpacing(8)

        # 标题标签
        self.title_label = QLabel()
        self.title_label.setObjectName("messageTitle")
        self.title_label.setStyleSheet("""
            QLabel#messageTitle {
                color: #f8f8f2;
                font-size: 16px;
                font-weight: bold;
                background-color: transparent;
            }
        """)

        # 文本标签
        self.text_label = QLabel()
        self.text_label.setObjectName("messageText")
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("""
            QLabel#messageText {
                color: #f8f8f2;
                font-size: 14px;
                background-color: transparent;
            }
        """)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.text_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        text_area.addWidget(self.title_label)
        text_area.addWidget(self.text_label)

        content_layout.addLayout(text_area, 1)
        container_layout.addLayout(content_layout)

        # 按钮区域
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(15)
        self.button_layout.addStretch()

        container_layout.addLayout(self.button_layout)
        layout.addWidget(self.main_container)

    def setup_animations(self):
        """设置入场和退场动画"""
        # 缩放动画 - now targets the container
        self.scale_animation = QPropertyAnimation(self.main_container, b"geometry")
        self.scale_animation.setDuration(250)
        self.scale_animation.setEasingCurve(QEasingCurve.Type.OutBack)

        # 透明度动画 - still targets the whole window
        self.opacity_animation = QPropertyAnimation(self, b"windowOpacity")
        self.opacity_animation.setDuration(200)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # 动画结束后重新启用阴影
        self.scale_animation.finished.connect(self.on_animation_finished)

    def on_animation_finished(self):
        """动画结束后重新启用阴影"""
        # 仅在显示动画完成后才重新启用阴影
        if self.windowOpacity() > 0.9 and self.main_container.graphicsEffect():
            self.main_container.graphicsEffect().setEnabled(True)

    def _animate_and_close(self, result_code):
        """执行退出动画并关闭对话框"""
        # 在动画开始前禁用阴影，防止闪烁
        if self.main_container.graphicsEffect():
            self.main_container.graphicsEffect().setEnabled(False)

        # 仅执行淡出动画以提高稳定性
        self.opacity_animation.setStartValue(1.0)
        self.opacity_animation.setEndValue(0.0)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.InCubic)

        # 当透明度动画完成时，关闭对话框
        try:
            # 先断开所有连接，避免重复触发
            self.opacity_animation.finished.disconnect()
        except TypeError:
            pass  # 如果之前没有连接，则忽略错误

        self.opacity_animation.finished.connect(lambda: self._finish_close(result_code))

        # 停止任何可能正在运行的缩放动画
        self.scale_animation.stop()
        # 启动透明度动画
        self.opacity_animation.start()

    def accept(self):
        """重写accept以触发退出动画"""
        self._animate_and_close(QDialog.DialogCode.Accepted)

    def reject(self):
        """重写reject以触发退出动画"""
        self._animate_and_close(QDialog.DialogCode.Rejected)

    def _finish_close(self, result_code):
        """完成关闭过程，确保窗口被销毁"""
        super().done(result_code)
        self.deleteLater()

    def setup_shadow_effect(self):
        """添加阴影效果"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.main_container.setGraphicsEffect(shadow)

    def set_icon(self, icon_type):
        """设置现代化图标"""
        pixmap = QPixmap(50, 50)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 根据类型设置颜色和符号
        if icon_type == QMessageBox.Icon.Information:
            color = QColor("#50fa7b")
            symbol = "ℹ"
            bg_color = QColor("#50fa7b")
        elif icon_type == QMessageBox.Icon.Warning:
            color = QColor("#f1fa8c")
            symbol = "⚠"
            bg_color = QColor("#f1fa8c")
        elif icon_type == QMessageBox.Icon.Critical:
            color = QColor("#ff5555")
            symbol = "✕"
            bg_color = QColor("#ff5555")
        elif icon_type == QMessageBox.Icon.Question:
            color = QColor("#8be9fd")
            symbol = "?"
            bg_color = QColor("#8be9fd")
        else:
            color = QColor("#f8f8f2")
            symbol = "●"
            bg_color = QColor("#f8f8f2")

        # 绘制现代化圆形背景
        bg_color.setAlpha(40)
        painter.setBrush(bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, 50, 50)

        # 绘制图标符号
        painter.setPen(color)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(0, 0, 50, 50, Qt.AlignmentFlag.AlignCenter, symbol)

        painter.end()
        self.icon_label.setPixmap(pixmap)

    def set_text(self, text):
        """设置文本内容并智能调整大小"""
        self.text_label.setText(text)

        # --- 精确计算所有组件和布局的高度 ---

        # 1. 布局定义的固定尺寸
        margins = self.main_container.layout().contentsMargins()
        vertical_margins = margins.top() + margins.bottom()
        container_spacing = self.main_container.layout().spacing()
        text_area_spacing = self.text_label.parent().layout().spacing()
        button_height = 36  # 按钮固定高度

        # 2. 使用QFontMetrics精确计算动态文本高度
        # 标题高度
        title_font_metrics = QFontMetrics(self.title_label.font())
        title_height = title_font_metrics.height()

        # 正文高度
        # 窗口宽度(400) - 左右边距(40) - 内容区HBox布局间距(15) - 图标宽度(50)
        width = 400
        text_width = width - vertical_margins - 15 - self.icon_label.width()
        text_font_metrics = QFontMetrics(self.text_label.font())
        bounding_rect = text_font_metrics.boundingRect(
            0, 0, text_width, 0, Qt.TextFlag.TextWordWrap, text
        )
        text_height = bounding_rect.height()

        # 3. 累加所有部分计算总高度
        total_height = (
            vertical_margins
            + title_height
            + text_area_spacing
            + text_height
            + container_spacing
            + button_height
            + 15
        )  # 增加一个15px的视觉缓冲

        # 4. 设置最终尺寸，确保不小于一个合理的最小值
        final_height = max(180, total_height)
        self.setFixedSize(width, final_height)

    def set_title(self, title):
        """设置标题"""
        self.title_label.setText(title)
        self.setWindowTitle(title)

    def add_button(self, text, role):
        """添加现代化按钮"""
        button = QPushButton(text)
        button.setObjectName("modernMessageButton")
        button.setFixedHeight(36)
        button.setMinimumWidth(80)

        # 根据角色设置按钮样式
        if role in [QMessageBox.ButtonRole.AcceptRole, QMessageBox.ButtonRole.YesRole]:
            button.setStyleSheet("""
                QPushButton#modernMessageButton {
                    background-color: #50fa7b;
                    color: #282a36;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 10px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton#modernMessageButton:hover {
                    background-color: #69ff8a;
                }
                QPushButton#modernMessageButton:pressed {
                    background-color: #40c861;
                }
            """)
        else:  # NoRole, RejectRole等
            button.setStyleSheet("""
                QPushButton#modernMessageButton {
                    background-color: #6272a4;
                    color: #f8f8f2;
                    border: 1px solid rgba(98, 114, 164, 0.5);
                    padding: 10px 20px;
                    border-radius: 10px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton#modernMessageButton:hover {
                    background-color: #7582b4;
                    border: 1px solid rgba(98, 114, 164, 0.8);
                }
                QPushButton#modernMessageButton:pressed {
                    background-color: #5562a4;
                }
            """)

        button.setProperty("role", role)  # 存储角色
        button.clicked.connect(lambda: self.button_clicked(button))
        self.button_layout.addWidget(button)
        return button

    def button_clicked(self, button):
        """处理按钮点击"""
        self.result_button = button
        role = button.property("role")

        if role in [QMessageBox.ButtonRole.AcceptRole, QMessageBox.ButtonRole.YesRole]:
            self.accept()
        else:
            self.reject()

    def clicked_button(self):
        """返回被点击的按钮"""
        return self.result_button

    def set_default_button(self, button):
        """设置默认按钮"""
        button.setDefault(True)
        button.setFocus()

    def showEvent(self, event):
        """显示时的入场动画"""
        super().showEvent(event)

        # 居中到父窗口
        target_x = None
        target_y = None
        if self.parent_widget:
            try:
                parent_rect = self.parent_widget.rect()
                global_top_left = self.parent_widget.mapToGlobal(parent_rect.topLeft())
                target_x = global_top_left.x() + (parent_rect.width() - self.width()) // 2
                target_y = global_top_left.y() + (parent_rect.height() - self.height()) // 2
            except Exception:
                target_x = None
                target_y = None

        if target_x is None or target_y is None:
            screen = self.screen() or (self.parent_widget.screen() if self.parent_widget else None)
            if screen:
                available = screen.availableGeometry()
                target_x = available.x() + (available.width() - self.width()) // 2
                target_y = available.y() + (available.height() - self.height()) // 2
            else:
                target_x = 0
                target_y = 0

        self.move(max(0, target_x), max(0, target_y))

        # The window itself is now at its final size but transparent
        self.setWindowOpacity(0.0)

        # 动画期间禁用阴影以防止渲染错误
        if self.main_container.graphicsEffect():
            self.main_container.graphicsEffect().setEnabled(False)

        # Animate the container's geometry instead of the window's
        final_container_geometry = self.main_container.geometry()

        # Start from a smaller, centered geometry
        start_container_geometry = QRect(
            final_container_geometry.x() + final_container_geometry.width() // 6,
            final_container_geometry.y() + final_container_geometry.height() // 6,
            final_container_geometry.width() * 2 // 3,
            final_container_geometry.height() * 2 // 3,
        )

        # Set initial state for animations
        self.main_container.setGeometry(start_container_geometry)

        # 缩放动画
        self.scale_animation.setStartValue(start_container_geometry)
        self.scale_animation.setEndValue(final_container_geometry)

        # 透明度动画
        self.opacity_animation.setStartValue(0.0)
        self.opacity_animation.setEndValue(1.0)

        # 启动动画
        self.scale_animation.start()
        self.opacity_animation.start()

    def paintEvent(self, event):
        """自定义绘制毛玻璃效果"""
        # 正确的绘制方式，确保painter被销毁
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            rect = self.rect()

            # 绘制毛玻璃背景
            gradient = QLinearGradient(0, 0, 0, rect.height())
            gradient.setColorAt(0, QColor(45, 47, 63, 245))
            gradient.setColorAt(1, QColor(30, 32, 44, 230))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 18, 18)

            # 绘制边框
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(68, 71, 90, 160), 1))
            painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 18, 18)

            # 绘制高光
            highlight_gradient = QLinearGradient(0, 0, 0, rect.height() * 0.35)
            highlight_gradient.setColorAt(0, QColor(255, 255, 255, 12))
            highlight_gradient.setColorAt(1, QColor(255, 255, 255, 0))

            painter.setBrush(highlight_gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, int(-rect.height() * 0.65)), 17, 17)
        finally:
            painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件 - 开始拖动"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件 - 拖动窗口"""
        if self.old_pos:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件 - 结束拖动"""
        self.old_pos = None

    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        elif event.key() in [Qt.Key.Key_Return, Qt.Key.Key_Enter]:
            # Enter键点击默认按钮
            for i in range(self.button_layout.count()):
                widget = self.button_layout.itemAt(i).widget()
                if isinstance(widget, QPushButton) and widget.isDefault():
                    widget.click()
                    return
        super().keyPressEvent(event)

    @staticmethod
    def information(parent, title, text):
        """显示信息消息框"""
        msg_box = StyledMessageBox(parent)
        msg_box.set_icon(QMessageBox.Icon.Information)
        msg_box.set_title(title)
        msg_box.set_text(text)
        ok_button = msg_box.add_button("确定", QMessageBox.ButtonRole.AcceptRole)
        msg_box.set_default_button(ok_button)
        return msg_box.exec()

    @staticmethod
    def warning(parent, title, text):
        """显示警告消息框"""
        msg_box = StyledMessageBox(parent)
        msg_box.set_icon(QMessageBox.Icon.Warning)
        msg_box.set_title(title)
        msg_box.set_text(text)
        ok_button = msg_box.add_button("确定", QMessageBox.ButtonRole.AcceptRole)
        msg_box.set_default_button(ok_button)
        return msg_box.exec()

    @staticmethod
    def critical(parent, title, text):
        """显示严重错误消息框"""
        msg_box = StyledMessageBox(parent)
        msg_box.set_icon(QMessageBox.Icon.Critical)
        msg_box.set_title(title)
        msg_box.set_text(text)
        ok_button = msg_box.add_button("确定", QMessageBox.ButtonRole.AcceptRole)
        msg_box.set_default_button(ok_button)
        return msg_box.exec()

    @staticmethod
    def question(parent, title, text):
        """显示问题消息框，返回True（确认）或False（取消）"""
        msg_box = StyledMessageBox(parent)
        msg_box.set_icon(QMessageBox.Icon.Question)
        msg_box.set_title(title)
        msg_box.set_text(text)
        yes_button = msg_box.add_button("确认", QMessageBox.ButtonRole.YesRole)
        no_button = msg_box.add_button("取消", QMessageBox.ButtonRole.NoRole)
        msg_box.set_default_button(no_button)

        msg_box.exec()
        return msg_box.clicked_button() == yes_button
