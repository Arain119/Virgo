"""Adaptive chat input widgets for the Qt dashboard.

Provides a resizable text editor and an input bar with a send button, optimized
for streaming chat interactions.
"""

from PyQt6.QtCore import QByteArray, QEasingCurve, QPropertyAnimation, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QPainter
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QWidget,
)


class AdaptiveTextEdit(QTextEdit):
    """自适应高度的文本输入框"""

    # 自定义信号
    send_requested = pyqtSignal()
    height_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 设置基本属性
        self.setObjectName("adaptiveChatInput")
        self.setAcceptRichText(False)  # 只接受纯文本
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 高度相关设置
        self.min_height = 40
        self.max_height = 150  # 约5行文本
        self.line_height = 20

        # 设置初始大小
        self.setMinimumHeight(self.min_height)
        self.setMaximumHeight(self.max_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # 连接信号
        self.textChanged.connect(self.adjust_height)

        # 设置字体
        font = QFont("Segoe UI", 10)
        self.setFont(font)

        # 初始高度调整
        self.adjust_height()

    def adjust_height(self):
        """根据文本内容自适应调整高度"""
        # 获取文档高度
        doc = self.document()
        doc.setTextWidth(self.viewport().width())

        # 计算所需高度
        content_height = int(doc.size().height())

        # 添加边距
        margins = self.contentsMargins()
        padding = margins.top() + margins.bottom() + 16  # 额外的内边距

        # 计算新高度，限制在最小最大值之间
        new_height = max(self.min_height, min(content_height + padding, self.max_height))

        # 只有高度变化时才更新
        if new_height != self.height():
            self.setFixedHeight(new_height)
            self.height_changed.emit(new_height)

    def keyPressEvent(self, event: QKeyEvent):
        """处理键盘事件"""
        # Enter键发送消息（不包含Shift+Enter）
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if event.modifiers() == Qt.KeyboardModifier.NoModifier:
                # 纯Enter键：发送消息
                self.send_requested.emit()
                return
            elif event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                # Shift+Enter：插入换行
                super().keyPressEvent(event)
                return

        # Escape键清空输入框
        elif event.key() == Qt.Key.Key_Escape:
            self.clear()
            return

        # 其他键的默认处理
        super().keyPressEvent(event)

    def insertFromMimeData(self, source):
        """处理粘贴内容，确保只粘贴纯文本"""
        if source.hasText():
            self.insertPlainText(source.text())


class SvgSendButton(QToolButton):
    """带SVG图标和缩放动画的发送按钮"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 设置基本属性
        self.setObjectName("svgSendButton")
        self.setFixedSize(40, 40)
        self.setText("")  # 不使用文字，完全自绘
        self.setToolTip("发送消息 (Enter)")

        # 状态管理
        self.is_sending = False

        # SVG内容
        self.svg_content = """<svg viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
            <circle cx="512" cy="512" r="417.96" fill="#16C4AF"/>
            <path d="M444.08 689.63c-5.22 0-10.45-2.09-14.63-6.27-8.36-8.36-8.36-21.42 0-29.78l156.73-156.73c8.36-8.36 21.42-8.36 29.78 0 8.36 8.36 8.36 21.42 0 29.78l-156.73 156.73c-4.7 4.18-9.93 6.27-15.15 6.27z" fill="#DCFFFA"/>
            <path d="M600.82 532.9c-5.22 0-10.45-2.09-14.63-6.27L429.46 369.9c-8.36-8.36-8.36-21.42 0-29.78 8.36-8.36 21.42-8.36 29.78 0l156.73 156.73c8.36 8.36 8.36 21.42 0 29.78-4.7 4.18-9.93 6.27-15.15 6.27z" fill="#DCFFFA"/>
        </svg>"""

        # 创建SVG渲染器
        self.svg_renderer = QSvgRenderer()
        self.svg_renderer.load(QByteArray(self.svg_content.encode()))

        # 缩放动画
        self.scale_animation = QPropertyAnimation(self, b"geometry")
        self.scale_animation.setDuration(150)
        self.scale_animation.setEasingCurve(QEasingCurve.Type.OutBounce)

        # 连接点击事件
        self.clicked.connect(self.animate_click)

    def animate_click(self):
        """点击缩放回弹动画"""
        if self.is_sending:
            return

        # 获取当前几何位置
        current_geo = self.geometry()

        # 缩小动画
        smaller_geo = QRect(
            current_geo.x() + 3,
            current_geo.y() + 3,
            current_geo.width() - 6,
            current_geo.height() - 6,
        )

        # 设置动画
        self.scale_animation.setStartValue(current_geo)
        self.scale_animation.setEndValue(smaller_geo)

        # 动画完成后回弹
        def bounce_back():
            self.scale_animation.setStartValue(smaller_geo)
            self.scale_animation.setEndValue(current_geo)
            self.scale_animation.setEasingCurve(QEasingCurve.Type.OutBounce)

            # 安全地断开连接
            try:
                self.scale_animation.finished.disconnect()
            except TypeError:
                pass  # 没有连接时忽略错误

            self.scale_animation.start()

        # 安全地断开之前的连接
        try:
            self.scale_animation.finished.disconnect()
        except TypeError:
            pass  # 没有连接时忽略错误

        self.scale_animation.finished.connect(bounce_back)
        self.scale_animation.start()

    def set_sending_state(self, sending: bool):
        """设置发送状态"""
        self.is_sending = sending

        if sending:
            self.setEnabled(False)
            self.setToolTip("正在生成...")
        else:
            self.setEnabled(True)
            self.setToolTip("发送消息 (Enter)")

        self.update()

    def paintEvent(self, event):
        """自定义绘制SVG图标"""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # 始终绘制SVG图标，根据状态调整透明度
            rect = QRectF(self.rect())
            if self.svg_renderer.isValid():
                if self.is_sending:
                    # 发送状态：降低透明度变灰
                    painter.setOpacity(0.4)
                else:
                    # 普通状态：正常透明度
                    painter.setOpacity(1.0)

                self.svg_renderer.render(painter, rect)
        finally:
            painter.end()

    def enterEvent(self, event):
        """鼠标进入事件"""
        if not self.is_sending:
            self.setStyleSheet(
                "QToolButton { background-color: rgba(22, 196, 175, 0.1); border-radius: 20px; }"
            )
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开事件"""
        self.setStyleSheet("")
        super().leaveEvent(event)


class AdaptiveInputWidget(QWidget):
    """自适应输入框组件容器"""

    # 定义信号
    send_requested = pyqtSignal(str)  # 发送消息信号
    stop_generation = pyqtSignal()  # 停止生成信号

    def __init__(self, parent=None):
        super().__init__(parent)

        # 创建主布局
        self.setup_ui()
        self.setup_animations()
        self.connect_signals()

    def setup_ui(self):
        """设置用户界面"""
        # 主容器
        self.main_frame = QFrame()
        self.main_frame.setObjectName("adaptiveInputFrame")

        # 主布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.main_frame)

        # 输入区域布局
        input_layout = QHBoxLayout(self.main_frame)
        input_layout.setContentsMargins(15, 8, 15, 8)
        input_layout.setSpacing(12)

        # 创建组件
        self.text_input = AdaptiveTextEdit()
        self.send_button = SvgSendButton()

        # 移除占位符文本
        self.text_input.setPlaceholderText("")

        # 添加到布局
        input_layout.addWidget(self.text_input, 1)  # 文本框占据大部分空间
        input_layout.addWidget(self.send_button, 0)  # 按钮固定大小

        # 添加阴影效果
        self.add_shadow_effect()

    def setup_animations(self):
        """设置动画效果"""
        # 高度变化动画
        self.height_animation = QPropertyAnimation(self, b"maximumHeight")
        self.height_animation.setDuration(200)
        self.height_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def add_shadow_effect(self):
        """添加阴影效果"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 100))  # 半透明黑色
        self.main_frame.setGraphicsEffect(shadow)

    def connect_signals(self):
        """连接信号"""
        # 文本输入信号
        self.text_input.send_requested.connect(self.handle_send_request)
        self.text_input.height_changed.connect(self.update_container_height)

        # 发送按钮信号
        self.send_button.clicked.connect(self.handle_send_request)

    def handle_send_request(self):
        """处理发送请求"""
        text = self.text_input.toPlainText().strip()
        if text:
            # 设置发送状态
            self.send_button.set_sending_state(True)

            # 发射信号
            self.send_requested.emit(text)

            # 清空输入框
            self.text_input.clear()

    def update_container_height(self, text_height):
        """更新容器高度"""
        # 计算新的容器高度（添加边距）
        margins = self.main_frame.contentsMargins()
        new_height = text_height + margins.top() + margins.bottom()

        # 使用动画更新高度
        self.height_animation.setStartValue(self.height())
        self.height_animation.setEndValue(new_height)
        self.height_animation.start()

    def set_sending_complete(self):
        """设置发送完成状态"""
        self.send_button.set_sending_state(False)

    def get_text(self):
        """获取输入文本"""
        return self.text_input.toPlainText()

    def clear_text(self):
        """清空输入文本"""
        self.text_input.clear()

    def set_placeholder_text(self, text: str):
        """设置占位符文本"""
        self.text_input.setPlaceholderText(text)

    def handle_stop_generation(self):
        """处理停止生成请求"""
        self.stop_generation.emit()
        self.set_sending_complete()

    def set_enabled(self, enabled: bool):
        """设置组件启用状态"""
        self.text_input.setEnabled(enabled)
        self.send_button.setEnabled(enabled and not self.send_button.is_sending)
