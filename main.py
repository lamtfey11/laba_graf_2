#15. Дано N-дерево. Найти все поддеревья, листья которых находятся в заданном диапазоне высот от корня поддерева.

import random
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import time
import psutil
import os
from typing import List, Optional, Tuple

class TreeNode:
    """Узел N-дерева"""
    __slots__ = ['data', 'children']
    
    def __init__(self, data: int):
        self.data = data          # Значение узла
        self.children = []        # Список дочерних узлов

class NaryTree:
    """Класс для работы с N-деревьями"""
    
    def __init__(self):
        self.root = None          # Корень дерева
        self._size = 0            # Количество узлов

    def get_size(self) -> int:
        """Возвращает количество узлов в дереве"""
        return self._size

    def create_random_tree(self, num_nodes: int, max_children: int = 3, none_prob: float = 0.2) -> None:
        """
        Создает случайное N-дерево
        :param num_nodes: максимальное количество узлов
        :param max_children: максимальное количество потомков у узла
        :param none_prob: вероятность отсутствия узла
        """
        if num_nodes <= 0:
            raise ValueError("Количество узлов должно быть положительным")
        
        values = [random.randint(1, 1000) if random.random() > none_prob else None 
                 for _ in range(num_nodes)]
        
        if values[0] is None:
            values[0] = random.randint(1, 1000)
        
        self.root = TreeNode(values[0])
        self._size = 1
        nodes_queue = deque([self.root])
        current_index = 1
        
        while nodes_queue and current_index < num_nodes:
            current_node = nodes_queue.popleft()
            
            if current_node is None:
                continue
                
            num_children = random.randint(1, max_children)
            
            for _ in range(num_children):
                if current_index >= num_nodes:
                    break
                    
                if values[current_index] is not None:
                    child_node = TreeNode(values[current_index])
                    current_node.children.append(child_node)
                    nodes_queue.append(child_node)
                    self._size += 1
                else:
                    nodes_queue.append(None)
                
                current_index += 1

    def find_subtrees_with_leaf_depths(self, min_depth: int, max_depth: int) -> Tuple[List['NaryTree'], float, float]:
        """
        Находит все поддеревья, где все листья находятся на глубине 
        (в рёбрах) от корня поддерева в диапазоне [min_depth, max_depth]
        Возвращает кортеж: (список поддеревьев, время выполнения в миллисекундах, использованная память в байтах)
        """
        if self.root is None:
            return [], 0.0, 0.0
            
        if min_depth < 0 or max_depth < min_depth:
            raise ValueError("Некорректный диапазон глубин")
        
        start_time = time.perf_counter()
        valid_subtrees = []
        queue = deque([self.root])
        
        while queue:
            current_node = queue.popleft()
            
            if not current_node.children:
                continue
                
            if self._check_leaf_depths(current_node, min_depth, max_depth):
                subtree = NaryTree()
                subtree.root = self._copy_subtree(current_node)
                subtree._size = self._count_nodes(subtree.root)
                valid_subtrees.append(subtree)
            
            for child in current_node.children:
                queue.append(child)
        
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000  # в миллисекундах
        
        # Память, использованная во время работы алгоритма
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss  # в байтах
        
        return valid_subtrees, elapsed_time, memory_usage

    def _check_leaf_depths(self, node: TreeNode, min_d: int, max_d: int) -> bool:
        """
        Проверяет, что ВСЕ листья поддерева находятся на глубине 
        от корня поддерева в диапазоне [min_d, max_d] (в рёбрах)
        """
        stack = [(node, 0)]  # (узел, текущая_глубина)
        
        while stack:
            current_node, depth = stack.pop()
            
            if not current_node.children:  # Лист
                if not (min_d <= depth <= max_d):
                    return False
            else:
                for child in current_node.children:
                    stack.append((child, depth + 1))
        
        return True

    def _copy_subtree(self, node: TreeNode) -> Optional[TreeNode]:
        """Глубокое копирование поддерева"""
        if node is None:
            return None
            
        new_node = TreeNode(node.data)
        new_node.children = [self._copy_subtree(child) for child in node.children]
        return new_node

    def _count_nodes(self, node: TreeNode) -> int:
        """Подсчет узлов в поддереве"""
        if node is None:
            return 0
        return 1 + sum(self._count_nodes(child) for child in node.children)

    def visualize(self, title: str = "N-дерево", node_size: int = 800, level_spacing: float = 1.5, sibling_spacing: float = 1.2) -> None:
        """Визуализация дерева с помощью networkx и matplotlib"""
        if self.root is None:
            print("Дерево пустое")
            return

        plt.clf()
        G = nx.DiGraph()
        pos = {}
        labels = {}

        def _calculate_positions(node, x_offset: float, y: float, spacing: float):
            if node is None:
                return x_offset

            node_id = id(node)
            child_x = x_offset
            child_widths = []
            
            for child in node.children:
                child_width = _calculate_positions(child, child_x, y - level_spacing, spacing * 0.9)
                child_widths.append(child_width - child_x)
                child_x = child_width + sibling_spacing

            total_width = sum(child_widths) + max(0, len(node.children) - 1) * sibling_spacing
            x = x_offset + total_width / 2
            
            pos[node_id] = (x, y)
            labels[node_id] = str(node.data)
            G.add_node(node_id)

            for child in node.children:
                G.add_edge(node_id, id(child))

            return x_offset + total_width

        _calculate_positions(self.root, 0, 0, sibling_spacing)

        if not pos:
            return

        x_values = [p[0] for p in pos.values()]
        y_values = [p[1] for p in pos.values()]
        x_range = max(x_values) - min(x_values)
        y_range = max(y_values) - min(y_values)

        fig_width = max(10, min(20, x_range * 0.5))
        fig_height = max(8, min(15, y_range * 0.7))
        plt.figure(figsize=(fig_width, fig_height))

        node_colors = ['skyblue' if G.out_degree(node) > 0 else 'lightgreen' for node in G.nodes()]

        nx.draw(G, pos,
                labels=labels,
                node_size=node_size,
                node_color=node_colors,
                font_size=10,
                font_weight='bold',
                arrows=False,
                edge_color='gray',
                width=1.5,
                alpha=0.8)

        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show()

def save_tree_to_file(tree: NaryTree, filename: str) -> None:
    """Сохраняет дерево в файл в формате: значение количество_детей"""
    with open(filename, 'w', encoding='utf-8') as f:
        def _write_node(node, level=0):
            if node is None:
                f.write("  " * level + "None\n")
                return
            f.write("  " * level + f"{node.data} {len(node.children)}\n")
            for child in node.children:
                _write_node(child, level + 1)
        
        _write_node(tree.root)
    print(f"Дерево сохранено в {filename}")

def save_subtrees_to_file(subtrees: List[NaryTree], filename: str, exec_time: float, memory_usage: float) -> None:
    """Сохраняет результаты поиска поддеревьев в файл"""
    with open(filename, 'w', encoding='utf-8') as f:
        if exec_time < 0.001:
            time_str = f"{exec_time * 1000:.3f} наносекунд"
        elif exec_time < 1:
            time_str = f"{exec_time:.3f} микросекунд"
        else:
            time_str = f"{exec_time:.3f} миллисекунд"
        
        f.write(f"Результаты поиска поддеревьев\n")
        f.write(f"Время выполнения: {time_str}\n")
        f.write(f"Использованная память: {memory_usage / (1024 ** 2):.2f} МБ\n")
        f.write(f"Найдено поддеревьев: {len(subtrees)}\n\n")
        
        for i, subtree in enumerate(subtrees, 1):
            f.write(f"Поддерево {i}:\n")
            f.write(f"  Корень: {subtree.root.data}\n")
            f.write(f"  Количество узлов: {subtree.get_size()}\n")
            
            f.write("  Структура поддерева:\n")
            stack = [(subtree.root, 0)]
            
            while stack:
                node, level = stack.pop()
                f.write("    " + "  " * level + f"{node.data}\n")
                for child in reversed(node.children):
                    stack.append((child, level + 1))
            
            f.write("\n")
    
    print(f"Результаты поиска сохранены в {filename}")

def manual_tree_creation() -> Optional[NaryTree]:
    """Создание дерева вручную"""
    print("\nСоздание дерева вручную (префиксный обход)")

    def _build_tree():
        try:
            line = input().strip()
            if line.lower() == 'none':
                return None
            parts = line.split()
            data = int(parts[0])
            num_children = int(parts[1])
            
            node = TreeNode(data)
            print(f"Узел {data}. Введите {num_children} детей:")
            node.children = [_build_tree() for _ in range(num_children)]
            return node
        except (ValueError, IndexError):
            print("Ошибка ввода! Формат: 'значение количество_детей'")
            return _build_tree()
    
    print("Введите корень дерева:")
    tree = NaryTree()
    tree.root = _build_tree()
    if tree.root:
        tree._size = tree._count_nodes(tree.root)
        print(f"Дерево создано. Узлов: {tree._size}")
        return tree
    return None

def main():
    """Основная функция с интерфейсом командной строки"""
    current_tree = None
    subtrees = []
    exec_time = 0.0
    memory_usage = 0.0
    
    while True:
        print("\nМеню:")
        print("1. Создать случайное дерево")
        print("2. Загрузить дерево из файла")
        print("3. Создать дерево вручную")
        print("4. Визуализировать дерево")
        print("5. Найти поддеревья по глубине листьев")
        print("6. Сохранить данные")
        print("8. Выход")
        
        choice = input("Выберите действие: ").strip()
        
        if choice == '1':
            try:
                num_nodes = int(input("Количество узлов: "))
                max_children = int(input("Максимальное число детей у узла: "))
                current_tree = NaryTree()
                current_tree.create_random_tree(num_nodes, max_children)
                print(f"Создано дерево с {current_tree.get_size()} узлами")
            except Exception as e:
                print(f"Ошибка: {e}")
        
        elif choice == '2':
            filename = input("Имя файла: ").strip()
            current_tree = NaryTree()
            current_tree.load_from_file(filename)
        
        elif choice == '3':
            current_tree = manual_tree_creation()
        
        elif choice == '4':
            if current_tree:
                current_tree.visualize()
            else:
                print("Дерево не загружено!")
        
        elif choice == '5':
            if not current_tree:
                print("Дерево не загружено!")
                continue
                
            try:
                min_d = int(input("Минимальная глубина листьев: "))
                max_d = int(input("Максимальная глубина листьев: "))
                
                subtrees, exec_time, memory_usage = current_tree.find_subtrees_with_leaf_depths(min_d, max_d)
                
                if exec_time < 0.001:
                    time_str = f"{exec_time * 1000:.3f} наносекунд"
                elif exec_time < 1:
                    time_str = f"{exec_time:.3f} микросекунд"
                else:
                    time_str = f"{exec_time:.3f} миллисекунд"
                
                print(f"\nНайдено {len(subtrees)} поддеревьев за {time_str}, использовано памяти: {memory_usage / (1024 ** 2):.2f} МБ:")
                
                if len(subtrees) > 20:
                    answer = input(f"Найдено {len(subtrees)} поддеревьев. Показать первое? (y/n): ").lower()
                    if answer == 'y':
                        subtrees[0].visualize(f"Поддерево 1 (корень {subtrees[0].root.data})")
                    else:
                        print("Вывод поддеревьев пропущен.")
                else:
                    for i, subtree in enumerate(subtrees, 1):
                        print(f"{i}. Корень: {subtree.root.data}, узлов: {subtree.get_size()}")
                        if input("Показать? (y/n): ").lower() == 'y':
                            subtree.visualize(f"Поддерево {i} (корень {subtree.root.data})")
            except ValueError as e:
                print(f"Ошибка: {e}")
        
        elif choice == '6':
            if current_tree:
                print("1. Сохранить основное дерево")
                print("2. Сохранить результаты последнего поиска")
                save_choice = input("Выберите что сохранить: ").strip()
                
                if save_choice == '1':
                    filename = input("Введите имя файла для сохранения дерева: ").strip()
                    save_tree_to_file(current_tree, filename)
                elif save_choice == '2':
                    if subtrees:
                        filename = input("Введите имя файла для сохранения результатов: ").strip()
                        save_subtrees_to_file(subtrees, filename, exec_time, memory_usage)
                    else:
                        print("Сначала выполните поиск поддеревьев!")
                else:
                    print("Неверный выбор!")
            else:
                print("Дерево не загружено!")
        
        elif choice == '8':
            print("Выход")
            break
        
        else:
            print("Неверный ввод!")

if __name__ == "__main__":
    main()
