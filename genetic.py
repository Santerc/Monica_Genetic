import math
import numpy as np
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
import ctypes
import math_tools as mt

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QWidget, QPushButton, QSpinBox, QLabel,
                             QDoubleSpinBox, QHBoxLayout, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QFile
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import sys

@dataclass
class Point:
    """
    单位圆上单点结构体，极坐标表示

    Attributes:
        binary_theta (int): theta的角度的二进制表示。
        binary_gamma (int): gamma的角度的二进制表示。
        theta (float): theta的角度的浮点表示。
        gamma (float): gamma的角度的浮点表示。
    """
    binary_theta: int
    binary_gamma: int
    theta: float
    gamma: float

@dataclass
class Individual:
    """
    个体（N个点）

    Attributes:
        points (list[Point]): 个体的点列表。
        dna_segments (list[np.uint64]): 个体的DNA片段列表。
    """
    points: list[Point]
    dna_segments: list[np.uint64]

class Genetic:
    def __init__(self, point_num, population_size, generation_size, initial_mutation_rate):
        """
        初始化遗传算法。

        Args:
            point_num (int): 每个个体的点的数量。
            population_size (int): 种群大小。
            generation_size (int): 迭代次数。
            initial_mutation_rate (float): 初始变异率。
        """
        self._point_num = point_num
        self._population_size = population_size
        self._generation_size = generation_size
        self._mutation_rate = initial_mutation_rate
        self._population = self.initialize_population()
        self._best_fitness = float('-inf')
        self._generations_without_improvement = 0

    def initialize_population(self):
        """
        初始化种群。

        Returns:
            list[Individual]: 初始化后的种群。
        """
        population = []
        for _ in range(self._population_size):
            points = []
            dna_segments = []

            for i in range(self._point_num):
                u = random.random() * 2 - 1
                theta = math.asin(u)
                gamma = random.random() * 2 * math.pi
                #映射到二进制
                bt = int((theta + math.pi/2) * (pow(2, 32) - 1) / math.pi)
                bg = int(gamma * (pow(2, 32) - 1) / (2 * math.pi))
                # 若存N对极坐标为一二进制数过大，因此将每对坐标存为一个64位整数，基因为一个细胞，包含N对染色体，每个染色体上64个基因
                dna_segment = np.uint64(mt.bit_concat(bt, bg))
                dna_segments.append(dna_segment)
                points.append(Point(bt, bg, theta, gamma))

            population.append(Individual(points=points, dna_segments=dna_segments))
        return population

    def crossover(self, parent1, parent2):
        """
        对两个父代进行交叉操作，生成一个子代。

        Args:
            parent1 (Individual): 第一个父代。
            parent2 (Individual): 第二个父代。

        Returns:
            Individual: 交叉后的子代。
        """
        segment_idx = random.randint(0, 5)
        cross_point = random.randint(0, 63)
        new_segments = parent1.dna_segments.copy()

        # 对选中分段及其后续所有分段进行交叉
        for i in range(segment_idx, self._point_num):
            if i == segment_idx:
                # 第一个分段从选中位置开始交叉
                mask = np.uint64((1 << cross_point) - 1)
                new_segments[i] = (
                        (parent1.dna_segments[i] & ~mask) |
                        (parent2.dna_segments[i] & mask)
                )
            else:
                # 后续分段完全交换
                pos = random.randint(0, 63)
                mask = np.uint64((1 << pos) - 1)
                new_segments[i] = (
                        (parent1.dna_segments[i] & ~mask) |
                        (parent2.dna_segments[i] & mask)
                )
        #计算成员变量
        points = []
        for segment in new_segments:
            theta, gamma = mt.bit_break(int(segment))
            theta_val = mt.mapping(math.pi/2, -math.pi/2, pow(2, 32) - 1, theta)
            gamma_val = mt.mapping(2*math.pi, 0, pow(2, 32) - 1, gamma)
            points.append(Point(theta, gamma, theta_val, gamma_val))

        return Individual(points=points, dna_segments=new_segments)

    def mutate(self, parent):
        """
        对个体进行变异操作。

        Args:
            parent (Individual): 待变异的个体。

        Returns:
            Individual: 变异后的个体。
        """
        if random.random() < self._mutation_rate:
            # 随机选择一个分段和位置进行变异
            segment_idx = random.randint(0, self._point_num - 1)  # Use point_num
            mutation_point = random.randint(0, 63)

            new_segments = parent.dna_segments.copy()

            # 对选中位置及其后续所有分段进行变异
            for i in range(segment_idx, self._point_num):  # Use point_num
                if i == segment_idx:
                    new_segments[i] ^= np.uint64(1 << mutation_point)
                else:
                    pos = random.randint(0, 63)
                    new_segments[i] ^= np.uint64(1 << pos)
            #计算成员变量
            points = []
            for segment in new_segments:
                theta, gamma = mt.bit_break(int(segment))
                theta_val = mt.mapping(math.pi/2, -math.pi/2, pow(2, 32) - 1, theta)
                gamma_val = mt.mapping(2*math.pi, 0, pow(2, 32) - 1, gamma)
                points.append(Point(theta, gamma, theta_val, gamma_val))

            return Individual(points=points, dna_segments=new_segments)
        return parent

    def scaler_function(self, individual):
        """
        标量函数计算，在导航中使用过类似的方式进行规划，这里迁移过来，采用引力和的方式，当各点引力和最小时分布当然最均匀

        Args:
            individual (Individual): 待计算适应度的个体。

        Returns:
            float: 个体的适应度值。
        """
        total_force = 0
        # 计算单个个体内部的6个点之间的力
        for i in range(len(individual.points)):
            for j in range(i + 1, len(individual.points)):
                p1 = individual.points[i]
                p2 = individual.points[j]
                r_sqr = (
                        (math.sin(p1.theta) - math.sin(p2.theta))**2 +
                        (math.cos(p1.theta) * math.cos(p1.gamma) -
                         math.cos(p2.theta) * math.cos(p2.gamma))**2 +
                        (math.cos(p1.theta) * math.sin(p1.gamma) -
                         math.cos(p2.theta) * math.sin(p2.gamma))**2
                )

                # 点之间的排斥力，距离越近力越大，包含除零保护
                force = 65535 if r_sqr < 1e-10 else 1/r_sqr
                total_force += force

        return -total_force

    def tournament_selection(self, fitness_values, tournament_size=3):
        """
        锦标赛选择法

        Args:
            fitness_values (list[float]): 种群中每个个体的适应度值。
            tournament_size (int): 锦标赛的大小。

        Returns:
            tuple[Individual, Individual]: 选择的两个父代。
        """
        parent1 = random.sample(list(zip(self._population, fitness_values)), tournament_size)
        parent2 = random.sample(list(zip(self._population, fitness_values)), tournament_size)
        return max(parent1, key=lambda x: x[1])[0], max(parent2, key=lambda x: x[1])[0]

    def evolve(self):
        """
        执行遗传算法，不断迭代进化种群

        Returns:
            list[Individual]: 最优种群。
        """
        best_generation = None
        best_fitness_sum = float('-inf')

        for generation in range(self._generation_size):
            fitness_values = [self.scaler_function(ind) for ind in self._population]
            current_fitness_sum = sum(fitness_values)

            if current_fitness_sum > best_fitness_sum:
                best_fitness_sum = current_fitness_sum
                best_generation = self._population.copy()
                self._generations_without_improvement = 0
            else:
                self._generations_without_improvement += 1

            elite_count = max(1, self._population_size // 10)
            sorted_population = sorted(zip(self._population, fitness_values),
                                       key=lambda x: x[1],
                                       reverse=True)
            new_population = [ind for ind, _ in sorted_population[:elite_count]]

            while len(new_population) < self._population_size:
                if random.random() < 0.1:
                    new_population.append(self.initialize_population()[0])
                else:
                    parent1, parent2 = self.tournament_selection(fitness_values)
                    offspring = self.crossover(parent1, parent2)
                    if random.random() < self._mutation_rate:
                        offspring = self.mutate(offspring)
                    new_population.append(offspring)

            self._population = new_population
            #参考学习的方法，引入动态变化率
            if self._generations_without_improvement > 20:
                self._mutation_rate = min(0.4, self._mutation_rate * 1.5)
            else:
                self._mutation_rate = max(0.05, self._mutation_rate * 0.9)

        return best_generation

def visualize_solutions(solutions):
    """
    可视化解决方案。

    Args:
        solutions (list[Point]): 解决方案中的点列表。
    """
    # 创建图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    xs = []
    ys = []
    zs = []

    for sol in solutions:
        x = np.cos(sol.theta) * np.cos(sol.gamma)
        y = np.cos(sol.theta) * np.sin(sol.gamma)
        z = np.sin(sol.theta)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # 点
    ax.scatter(xs, ys, zs, c='r', marker='o', s=100)

    # 3D图形化区
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # 标签与题目
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Points Distribution on Unit Sphere')

    plt.show()

class GeneticThread(QThread):
    """
    遗传算法计算线程

    Attributes:
        finished (pyqtSignal): 信号，当算法完成时发出。
        progress (pyqtSignal): 信号，用于报告进度。
    """
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, float)

    def __init__(self, params):
        """
        初始化线程。

        Args:
            params (dict): 遗传算法的参数。
        """
        super().__init__()
        self.params = params

    def run(self):
        """
        执行遗传算法。
        """
        ga = Genetic(**self.params)
        best_population = None

        for gen in range(self.params['generation_size']):
            ga._population = ga.evolve()
            best_ind = min(ga._population, key=lambda x: ga.scaler_function(x))
            fitness = -ga.scaler_function(best_ind)
            self.progress.emit(gen, fitness)
            best_population = ga._population

        self.finished.emit(best_population)

class MainWindow(QMainWindow):
    """
    主窗口类，包含控制面板和可视化面板。
    """
    def __init__(self):
        """
        初始化主窗口。
        """
        super().__init__()
        self.setWindowTitle("球面分布遗传算法优化")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(self.load_styles())

        self.set_window_icon()

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Create control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)

        # Create visualization panel
        viz_panel = self.create_visualization_panel()
        layout.addWidget(viz_panel)

        # Initialize plot
        self.fig = Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.viz_layout.addWidget(self.canvas)

    def set_window_icon(self):
        """
        设置窗口图标，谁不喜欢好看一点的软件呢（GPT教我的）
        """
        from PyQt6.QtGui import QIcon
        try:
            # First try loading from file
            icon_path = './source/Morfonica.png'  # Place your icon file in project root
            if QFile.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
            else:
                # Fallback to built-in icon
                self.setWindowIcon(QIcon.fromTheme('applications-science'))
        except Exception as e:
            print(f"Failed to load window icon: {e}")
    def create_control_panel(self):
        """
        创建控制面板，包含参数设置和状态显示，还有活泼的Emoji）

        Returns:
            QGroupBox: 控制面板。
        """
        panel = QGroupBox("Control Panel")
        layout = QVBoxLayout(panel)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)

        # Point number
        point_layout = QHBoxLayout()
        self.point_spin = QSpinBox()
        self.point_spin.setRange(4, 20)
        self.point_spin.setValue(8)
        point_layout.addWidget(QLabel("🧮 Number of Points:"))
        point_layout.addWidget(self.point_spin)
        param_layout.addLayout(point_layout)

        # Population size
        pop_layout = QHBoxLayout()
        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(6, 20)
        self.pop_spin.setValue(6)
        pop_layout.addWidget(QLabel("👥 Population Size:"))
        pop_layout.addWidget(self.pop_spin)
        param_layout.addLayout(pop_layout)

        # Generation size
        gen_layout = QHBoxLayout()
        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(10, 10000)
        self.gen_spin.setValue(100)
        gen_layout.addWidget(QLabel("📈 Generation Size:"))
        gen_layout.addWidget(self.gen_spin)
        param_layout.addLayout(gen_layout)

        # Mutation rate
        mut_layout = QHBoxLayout()
        self.mut_spin = QDoubleSpinBox()
        self.mut_spin.setRange(0.01, 1.0)
        self.mut_spin.setValue(0.2)
        self.mut_spin.setSingleStep(0.01)
        mut_layout.addWidget(QLabel("🧬 Mutation Rate:"))
        mut_layout.addWidget(self.mut_spin)
        param_layout.addLayout(mut_layout)

        layout.addWidget(param_group)

        # Status display
        self.status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(self.status_group)
        self.generation_label = QLabel("Generation: 0")
        self.fitness_label = QLabel("Best Fitness: 0.0")
        status_layout.addWidget(self.generation_label)
        status_layout.addWidget(self.fitness_label)
        layout.addWidget(self.status_group)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_optimization)
        button_layout.addWidget(self.start_button)
        layout.addLayout(button_layout)

        layout.addStretch()
        return panel

    def create_visualization_panel(self):
        """
        创建可视化面板。

        Returns:
            QGroupBox: 可视化面板。
        """
        panel = QGroupBox("Visualization")
        self.viz_layout = QVBoxLayout(panel)
        return panel

    def start_optimization(self):
        """
        开始优化过程，其实只是前端发信号而已
        """
        self.start_button.setEnabled(False)
        params = {
            'point_num': self.point_spin.value(),
            'population_size': self.pop_spin.value(),
            'generation_size': self.gen_spin.value(),
            'initial_mutation_rate': self.mut_spin.value()
        }
        #进程间传参
        self.thread = GeneticThread(params)
        self.thread.finished.connect(self.optimization_finished)
        self.thread.progress.connect(self.update_progress)
        self.thread.start()

    def update_progress(self, generation, fitness):
        """
        更新进度显示。

        Args:
            generation (int): 当前迭代次数。
            fitness (float): 当前最佳适应度。
        """
        self.generation_label.setText(f"Generation: {generation}")
        self.fitness_label.setText(f"Best Fitness: {fitness:.4f}")

    def optimization_finished(self, population):
        """
        优化结束，更新可视化
        """
        self.start_button.setEnabled(True)
        best_individual = min(population, key=lambda x: Genetic.scaler_function(Genetic, x))
        self.update_visualization(best_individual.points)

    def load_styles(self):
        """
        飞书简约风QSS，把美化的东西丢在一起，不要干扰核心算法

        Returns:
            QSS样式
        """
        return """
        QWidget { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(240, 245, 255, 255), stop:1 rgba(220, 230, 250, 255)); }
        QPushButton { background-color: #0078D4; color: white; border-radius: 8px; padding: 10px; }
        QPushButton:hover { background-color: #005A9E; }
        QLabel { font-size: 14px; }
        """

    def update_visualization(self, points):
        """
        更新可视化
        """
        self.ax.clear()

        # Convert coordinates and plot points
        xs, ys, zs = [], [], []
        for p in points:
            x = np.cos(p.theta) * np.cos(p.gamma)
            y = np.cos(p.theta) * np.sin(p.gamma)
            z = np.sin(p.theta)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            self.ax.plot([0, x], [0, y], [0, z], '--', color='red', alpha=0.3)

        # Plot points
        self.ax.scatter(xs, ys, zs, c=range(len(points)),
                        cmap='viridis', s=200, edgecolor='black')

        # Plot wireframe sphere
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(-np.pi/2, np.pi/2, 40)
        u, v = np.meshgrid(u, v)
        x = np.cos(v) * np.cos(u)
        y = np.cos(v) * np.sin(u)
        z = np.sin(v)
        self.ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

        # Customize appearance
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_box_aspect([1,1,1])

        self.canvas.draw()

def launch_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    launch_gui()