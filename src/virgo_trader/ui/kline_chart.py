"""K-line / candlestick chart widgets built on PyQtGraph.

Used by the dashboard to visualize market prices, trades, and portfolio value
time series.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen("w"))
        for d in self.data:
            p.drawLine(QtCore.QPointF(d["t"], d["l"]), QtCore.QPointF(d["t"], d["h"]))
            if d["o"] > d["c"]:
                p.setBrush(pg.mkBrush("r"))
            else:
                p.setBrush(pg.mkBrush("g"))
            p.drawRect(QtCore.QRectF(d["t"] - 0.4, d["o"], 0.8, d["c"] - d["o"]))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class KLineChartWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("#282a36")
        self.kline_data = None
        self.index_map = None
        self._index = None

        # --- Main K-line Plot ---
        self.kline_plot = self.addPlot(row=0, col=0)
        self.kline_plot.setLabel("left", "Price", color="#f8f8f2")
        self.kline_plot.setLabel("bottom", "Time", color="#f8f8f2")
        self.kline_plot.showGrid(x=True, y=True, alpha=0.3)

        # --- Portfolio Value Plot ---
        self.portfolio_plot = self.addPlot(row=1, col=0)
        self.portfolio_plot.setMaximumHeight(200)
        self.portfolio_plot.setLabel("left", "Portfolio Value", color="#50fa7b")
        self.portfolio_plot.setXLink(self.kline_plot)
        self.portfolio_plot.showGrid(x=True, y=True, alpha=0.3)

        # --- Create PlotDataItems ---
        self.candlestick_item = None
        self.buy_scatter = pg.ScatterPlotItem()
        self.sell_scatter = pg.ScatterPlotItem()
        self.portfolio_curve = pg.PlotDataItem(pen=pg.mkPen("#50fa7b", width=2))
        self.line_curve = pg.PlotDataItem(pen=pg.mkPen("#8be9fd", width=1.8))

        self.kline_plot.addItem(self.buy_scatter)
        self.kline_plot.addItem(self.sell_scatter)
        self.kline_plot.addItem(self.line_curve)
        self.portfolio_plot.addItem(self.portfolio_curve)

    def clear_background(self):
        """Clears the candlestick background without wiping overlays."""
        if self.candlestick_item:
            self.kline_plot.removeItem(self.candlestick_item)
            self.candlestick_item = None
        self.kline_data = None
        self.line_curve.clear()
        self.index_map = None
        self._index = None
        # Keep scatters/portfolio curve; update_overlays will refresh them.

    def plot_background(self, kline_data):
        """一次性绘制静态的K线图背景"""
        if kline_data.empty:
            self.clear_all()
            return

        self.kline_data = kline_data  # Store for later use
        self.index_map = kline_data.index
        self._index = pd.Index(self.index_map)

        candlestick_data = []
        for i, (_timestamp, row) in enumerate(kline_data.iterrows()):
            candlestick_data.append(
                {"t": i, "o": row["open"], "h": row["high"], "l": row["low"], "c": row["close"]}
            )

        if self.candlestick_item:
            self.kline_plot.removeItem(self.candlestick_item)

        self.candlestick_item = CandlestickItem(candlestick_data)
        self.kline_plot.addItem(self.candlestick_item)

        # Set X-axis ticks to show dates
        x_axis = self.kline_plot.getAxis("bottom")
        num_ticks = 10
        tick_step = max(1, len(kline_data) // num_ticks)
        ticks = [
            (i, kline_data.index[i].strftime("%Y-%m-%d"))
            for i in range(0, len(kline_data), tick_step)
        ]
        x_axis.setTicks([ticks])
        self.line_curve.clear()

    def plot_line_background(self, series: pd.Series):
        if series is None or series.empty:
            self.clear_all()
            return
        self.clear_background()
        self.index_map = series.index
        self._index = pd.Index(self.index_map)
        normalized = series / series.iloc[0]
        x_values = np.arange(len(normalized))
        self.line_curve.setData(x=x_values, y=normalized.values)
        ticks = [
            (i, series.index[i].strftime("%Y-%m-%d"))
            for i in range(0, len(series), max(1, len(series) // 10))
        ]
        self.kline_plot.getAxis("bottom").setTicks([ticks])

    def update_overlays(self, trades, portfolio_history):
        """动态更新买卖点和资金曲线（兼容旧接口）。"""
        self.update_trades(trades)
        self.update_portfolio_curve(portfolio_history)

    def clear_overlays(self):
        """Clears dynamic overlays without removing the background plot."""
        self.buy_scatter.clear()
        self.sell_scatter.clear()
        self.portfolio_curve.clear()

    def update_trades(self, trades):
        """Update buy/sell markers only."""
        if self.index_map is None or len(self.index_map) == 0:
            return

        index = self._index if self._index is not None else pd.Index(self.index_map)

        if trades is not None and not trades.empty:
            buy_trades = trades[trades["side"] == "BUY"]
            sell_trades = trades[trades["side"] == "SELL"]

            buy_indices = index.get_indexer(buy_trades.index)
            if (buy_indices < 0).any():
                buy_indices = index.get_indexer(buy_trades.index, method="nearest")
            sell_indices = index.get_indexer(sell_trades.index)
            if (sell_indices < 0).any():
                sell_indices = index.get_indexer(sell_trades.index, method="nearest")

            if len(buy_trades) > 0:
                self.buy_scatter.setData(
                    x=buy_indices,
                    y=buy_trades["price"].values,
                    symbol="t",
                    size=12,
                    brush=pg.mkBrush(0, 255, 0, 200),
                )
            else:
                self.buy_scatter.clear()

            if len(sell_trades) > 0:
                self.sell_scatter.setData(
                    x=sell_indices,
                    y=sell_trades["price"].values,
                    symbol="t1",
                    size=12,
                    brush=pg.mkBrush(255, 0, 0, 200),
                )
            else:
                self.sell_scatter.clear()
        else:
            self.buy_scatter.clear()
            self.sell_scatter.clear()

    def update_portfolio_curve(self, portfolio_history):
        """Update portfolio curve only."""
        if portfolio_history is not None and len(portfolio_history) > 0:
            portfolio_array = np.asarray(portfolio_history, dtype=float).ravel()
            x_values = np.arange(portfolio_array.shape[0])
            self.portfolio_curve.setData(x=x_values, y=portfolio_array)
        else:
            self.portfolio_curve.clear()

    def clear_all(self):
        if self.candlestick_item:
            self.kline_plot.removeItem(self.candlestick_item)
            self.candlestick_item = None
        self.kline_data = None
        self.index_map = None
        self.line_curve.clear()
        self.buy_scatter.clear()
        self.sell_scatter.clear()
        self.portfolio_curve.clear()
