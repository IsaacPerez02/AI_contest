# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
from util import PriorityQueue
from util import manhattan_distance
from util import Counter
#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveAStarAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    def get_boundary_pos(self, game_state):
        """
        finds a list of positions along the boundary that the pacman can use to go home or the ghost can defend
        """
        layout = game_state.get_walls() # returns game state with true if wall and false if not wall
        width = layout.width
        height = layout.height
        boundary_x = width // 2 - 1 if self.red else width // 2
        boundary_pos = [(boundary_x, y) for y in range(height) if not layout[boundary_x][y]]

        return boundary_pos

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        food_eaten = successor.get_agent_state(self.index).num_carrying
        food_list_defense = self.get_food_you_are_defending(successor).as_list()
        features['successor_score'] = -len(food_list)
        my_pos = successor.get_agent_state(self.index).get_position()
        boundary_pos = self.get_boundary_pos(successor)
        min_boundary_distance = min([self.get_maze_distance(my_pos, boundary) for boundary in boundary_pos])
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
    
        if len(food_list) > 1:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
            if food_eaten > 4:
                features['high_carrying_risk'] = min_boundary_distance
            elif food_eaten > 1:
                features['distance_to_home'] = min_boundary_distance
        else:
            features['high_carrying_risk'] = min_boundary_distance

        if len(ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, ght.get_position()) for ght in ghosts]
            features['ghosts_distance'] = min(dists)

            if (min(dists) <= 2):
                features['run_away'] = 1
                features['high_carrying_risk'] = 2 * min_boundary_distance
            elif (min(dists) <= 5):
                features['high_carrying_risk'] = min_boundary_distance

        if action == Directions.STOP: features['stop'] = 1 

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1000, 'distance_to_food': -10, 'high_carrying_risk': -50, 'distance_to_home': -1, 'ghosts_distance': 10, 'stop': -10}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that actively stays close to the nearest invader
    in its field and minimizes invader presence, unless scared.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0

        # Get scared timer
        scared_timer = my_state.scared_timer

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            # Find the distance to the closest invader
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            closest_invader_distance = min(dists)

            features['invader_distance'] = closest_invader_distance

        # Penalize stopping or reversing direction
        if action == Directions.STOP: 
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -1000,
            'stop': -200, 
            'reverse': -50
        }
        
        
class OptimizedDefensiveAgent(ReflexCaptureAgent):
    """
    Un agente defensivo optimizado que protege comida y minimiza la presencia de invasores.
    """

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.previous_food = None  # Para rastrear comida desaparecida

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Identificar si el agente está en defensa o ataque
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # Detectar enemigos visibles
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible_invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        # Número de invasores detectados
        features['num_invaders'] = len(visible_invaders)

        if len(visible_invaders) > 0:
            # Calcular distancia al invasor más cercano
            invader_distances = [self.get_maze_distance(my_pos, inv.get_position()) for inv in visible_invaders]
            features['invader_distance'] = min(invader_distances)

        # Detectar comida desaparecida
        food_defending = self.get_food_you_are_defending(game_state).as_list()
        if self.previous_food:
            missing_food = [food for food in self.previous_food if food not in food_defending]
        else:
            missing_food = []
        self.previous_food = food_defending  # Actualizar estado previo de comida

        # Priorizar comida desaparecida si existe
        if len(missing_food) > 0:
            food_distances = [self.get_maze_distance(my_pos, food) for food in missing_food]
            features['missing_food_distance'] = min(food_distances)
        else:
            features['missing_food_distance'] = 0

        # Calcular distancia al límite (patrulla cuando no hay objetivos claros)
        boundary_positions = self.get_boundary_pos(successor)
        boundary_distances = [self.get_maze_distance(my_pos, pos) for pos in boundary_positions]
        features['boundary_distance'] = min(boundary_distances)

        # Penalizar quedarse quieto o moverse en reversa
        if action == Directions.STOP:
            features['stop'] = 1
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Ajusta los pesos de cada característica.
        """
        return {
            'num_invaders': -1000,             # Prioridad para minimizar invasores
            'invader_distance': -500,         # Atrapar invasores cercanos
            'on_defense': 100,                # Mantenerse en defensa
            'missing_food_distance': -300,    # Proteger comida desaparecida
            'boundary_distance': -10,         # Patrullar la frontera
            'stop': -100,                     # Penalizar quedarse quieto
            'reverse': -50                    # Penalizar movimientos reversos
        }

class DefensiveAStarAgent(ReflexCaptureAgent):
    """
    Un agente defensivo que protege comida, patrulla la frontera y persigue invasores
    utilizando A* para planificación eficiente.
    """

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.previous_food = None  # Rastrea comida protegida desaparecida
        self.target = None  # Objetivo actual del agente

    def register_initial_state(self, game_state):
        """
        Inicializa el estado del agente y registra la comida inicial.
        """
        super().register_initial_state(game_state)
        self.previous_food = self.get_food_you_are_defending(game_state).as_list()
        self.target = None

    def choose_action(self, game_state):
        """
        Decide la mejor acción basándose en prioridades defensivas y utiliza A* si es necesario.
        """
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Actualizar comida protegida
        current_food = self.get_food_you_are_defending(game_state).as_list()
        missing_food = [food for food in self.previous_food if food not in current_food]
        self.previous_food = current_food

        # Detectar invasores visibles
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        # Prioridades:
        # 1. Perseguir invasores visibles
        if visible_invaders:
            invader_positions = [inv.get_position() for inv in visible_invaders]
            self.target = min(invader_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))

        # 2. Ir a la comida desaparecida
        elif missing_food:
            self.target = missing_food[0]

        # 3. Patrullar la frontera
        elif not self.target:
            boundary_positions = self.get_boundary_pos(game_state)
            self.target = random.choice(boundary_positions)

        # Usar A* para calcular la ruta hacia el objetivo
        if self.target:
            path = self.a_star(game_state, self.target)
            if path:
                return path[0]  # Siguiente acción en la ruta

        # Si no hay objetivos claros, utiliza la evaluación
        values = [self.evaluate(game_state, action) for action in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def a_star(self, game_state, target):
        """
        Implementa el algoritmo A* para calcular la ruta más corta al objetivo.
        """
        frontier = PriorityQueue()
        start_pos = game_state.get_agent_state(self.index).get_position()
        frontier.push((start_pos, []), 0)
        explored = set()

        while not frontier.is_empty():
            current_pos, path = frontier.pop()

            if current_pos in explored:
                continue
            explored.add(current_pos)

            if current_pos == target:
                return path

            for action in game_state.get_legal_actions(self.index):
                successor = self.get_successor(game_state, action)
                successor_pos = successor.get_agent_state(self.index).get_position()
                if successor_pos not in explored:
                    cost = len(path) + 1
                    priority = cost + manhattan_distance(successor_pos, target)
                    frontier.push((successor_pos, path + [action]), priority)

        return []

    def get_features(self, game_state, action):
        """
        Calcula características relevantes para la defensa.
        """
        features = Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        # Característica: número de invasores visibles
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible_invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        features['num_invaders'] = len(visible_invaders)

        if visible_invaders:
            invader_distances = [self.get_maze_distance(my_pos, inv.get_position()) for inv in visible_invaders]
            features['invader_distance'] = min(invader_distances)

        # Característica: comida desaparecida
        if self.previous_food:
            food_defending = self.get_food_you_are_defending(game_state).as_list()
            missing_food = [food for food in self.previous_food if food not in food_defending]
            if missing_food:
                food_distances = [self.get_maze_distance(my_pos, food) for food in missing_food]
                features['missing_food_distance'] = min(food_distances)

        # Característica: distancia al objetivo actual
        if self.target:
            features['target_distance'] = self.get_maze_distance(my_pos, self.target)

        # Penalizar quedarse quieto o moverse en reversa
        if action == Directions.STOP:
            features['stop'] = 1
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Define los pesos asociados a las características.
        """
        return {
            'num_invaders': -1000,             # Prioridad máxima para invasores
            'invader_distance': -500,         # Aproximarse a invasores
            'missing_food_distance': -300,    # Proteger comida desaparecida
            'target_distance': -100,          # Moverse hacia el objetivo
            'stop': -200,                     # Penalizar quedarse quieto
            'reverse': -50                    # Penalizar movimientos reversos
        }

