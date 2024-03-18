import math
import random
import curses
from collections import defaultdict

class Node:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.results = defaultdict(int)
        self.visit_count = 0
        self.untried_actions = state.get_legal_actions()

    def q(self):
        return self.results[1] - self.results[-1]  # wins minus losses

    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    # Step 1: Select a node to expand

    def select_node(self):
        current_node = self
        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_child(self, C=1.4):
        # Negate q because we want to choose the child with the worst value for our opponent
        return max(self.children, key=lambda c: (-c.q() / c.visit_count) + C * math.sqrt(2 * math.log(self.visit_count) / c.visit_count))

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.get_next_state(action)
        child = Node(next_state, self, action)
        self.children.append(child)
        return child

    # Step 2: Rollout to the end of the game

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_terminal():
            action = self.rollout_policy(current_rollout_state.get_legal_actions())
            current_rollout_state = current_rollout_state.get_next_state(action)
        reward = current_rollout_state.get_result()
        return reward * (1 if self.state.player_turn == current_rollout_state.player_turn else -1)

    def rollout_policy(self, legal_actions):
        return random.choice(legal_actions)

    # Step 3: Backpropagate the results

    def backpropagate(self, result):
        self.visit_count += 1
        self.results[result] += 1
        if self.parent:
            self.parent.backpropagate(-result)  # Negative because we're minimizing the opponent's reward

    # Tree Search

    def best_action(self, n_simulations=100):
        for _ in range(n_simulations):
            leaf_node = self.select_node()
            result = leaf_node.rollout()
            leaf_node.backpropagate(result)
        return max(self.children, key=lambda c: c.visit_count).parent_action


class State:
    def __init__(self, board=None, player_turn='X'):
        self.board = board or [[' ' for _ in range(3)] for _ in range(3)]
        self.player_turn = player_turn

    def __str__(self):
        return '\n'.join(' '.join(row) for row in self.board)

    def get_legal_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']

    def is_terminal(self):
        # Check for a win or a draw
        return self.get_result() != 0 or all(self.board[i][j] != ' ' for i in range(3) for j in range(3))

    def get_result(self):
        # Check rows and columns for a win
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return -1  # It's always the opponent who won, we can't have won already on our turn
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return -1
        # Check diagonals for a win
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return -1
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return -1
        return 0

    def get_next_state(self, action):
        new_board = [row[:] for row in self.board]
        new_board[action[0]][action[1]] = self.player_turn
        next_player = 'O' if self.player_turn == 'X' else 'X'
        return State(new_board, next_player)


# TESTING

def main(stdscr):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()  # Clear the window
    state = State()
    current_pos = (0, 0)  # Starting position

    while not state.is_terminal():
        stdscr.clear()
        stdscr.addstr(0, 0, "+---+---+---+")
        for i in range(3):
            for j in range(3):
                if (i, j) == current_pos:
                    stdscr.addstr(i*2+1, j*4, f"|[{state.board[i][j]}]")
                else:
                    stdscr.addstr(i*2+1, j*4, f"| {state.board[i][j]} ")
            stdscr.addstr(i*2+1, 12, "|")
            stdscr.addstr(i*2+2, 0, "+---+---+---+")
        stdscr.refresh()

        if state.player_turn == 'X':
            key = stdscr.getch()
            if key == curses.KEY_UP and current_pos[0] > 0:
                current_pos = (current_pos[0] - 1, current_pos[1])
            elif key == curses.KEY_DOWN and current_pos[0] < 2:
                current_pos = (current_pos[0] + 1, current_pos[1])
            elif key == curses.KEY_LEFT and current_pos[1] > 0:
                current_pos = (current_pos[0], current_pos[1] - 1)
            elif key == curses.KEY_RIGHT and current_pos[1] < 2:
                current_pos = (current_pos[0], current_pos[1] + 1)
            elif key in [curses.KEY_ENTER, 10, 13]:
                if current_pos in state.get_legal_actions():
                    state = state.get_next_state(current_pos)
        else:
            node = Node(state)
            action = node.best_action(n_simulations = 100)
            state = state.get_next_state(action)


    stdscr.clear()
    result = state.get_result()
    if result == -1 and state.player_turn == 'O':
        stdscr.addstr(1, 0, "Player X wins!")
    elif result == -1 and state.player_turn == 'X':
        stdscr.addstr(1, 0, "Player O wins!")
    else:
        stdscr.addstr(1, 0, "It's a draw!")
    stdscr.refresh()
    stdscr.getch()


if __name__ == "__main__":
    curses.wrapper(main)

    # board = [
    #     ['X', 'O', 'X'],
    #     [' ', 'O', ' '],
    #     [' ', ' ', ' ']
    # ]
    # state = State(board=board, player_turn='X')
    # node = Node(state)
    # node.best_action(n_simulations=1000)
    # for child in node.children:
    #     print(f"Action: {child.parent_action}, Visit Count: {child.visit_count}")
