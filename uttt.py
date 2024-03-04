import math

#0-80 squares, 81-89 result of each subgame, 90 next symbol, 91 subgame constraint, 92 result of uttt
SIZE = 93
NEXT_SYMBOL_INDEX = 90
CONSTRAINT_INDEX = 91
RESULT_INDEX = 92

X_STATE_VALUE = 1
O_STATE_VALUE = 2
DRAW_STATE_VALUE = 3
UNCONSTRAINED_STATE_VALUE = 9
MAPPING ={
#First subgame
(1,1): 0, (1,2): 1, (1,3): 2,
(2,1):3, (2,2): 4, (2,3): 5,
(3,1):6, (3,2):7, (3,3):8,

#Second subgame
(1,4):9, (1,5):10, (1,6):11,
(2,4):12, (2,5):13, (2,6):14,
(3,4):15, (3,5):16, (3,6):17,

#Third subgame
(1,7):18, (1,8):19, (1,9):20,
(2,7):21, (2,8):22, (2,9):23,
(3,7):24, (3,8):25, (3,9):26,

#Fourth subgame
(4,1):27, (4,2):28, (4,3):29,
(5,1):30, (5,2):31, (5,3):32,
(6,1):33, (6,2):34, (6,3):35,

#Fifth subgame
(4,4):36, (4,5):37, (4,6):38,
(5,4):39, (5,5):40, (5,6):41,
(6,4):42, (6,5):43, (6,6):44,

#Sixth subgame
(4,7):45, (4,8):46, (4,9):47,
(5,7):48, (5,8):49, (5,9):50,
(6,7):51, (6,8):52, (6,9):53,

#Seventh subgame
(7,1):54, (7,2):55, (7,3):56,
(8,1):57, (8,2):58, (8,3):59,
(9,1):60, (9,2):61, (9,3):62,

#Eighth subgame
(7,4):63, (7,5):64, (7,6):65,
(8,4):66, (8,5):67, (8,6):68,
(9,4):69, (9,5):70, (9,6):71,

#Ninth subgame
(7,7):72, (7,8):73, (7,9):74,
(8,7):75, (8,8):76, (8,9):77,
(9,7):78, (9,8):79, (9,9):80,
}

REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}


class Move:
    def __init__(self,
                symbol: int, # X_STATE_VALUE = 1 or O_STATE_VALUE = 2
                index: int): # int from 0 to 80
        '''A move contains the symbol (represented as an int) and the index (int from 0 to 80) where the symbol will be placed.'''
        self.symbol = symbol  # X_STATE_VALUE or O_STATE_VALUE
        self.index = index  # int from 0 to 80

    def __str__(self):
        output = '{cls}(symbol={symbol}, index={index})'
        output = output.format(
            cls='Move',
            symbol={X_STATE_VALUE: 'X', O_STATE_VALUE: 'O'}[self.symbol],
            index=self.index,
        )
        return output

class UltimateTicTacToe:
    def __init__(self,
                state = None): #If no state is given, it generates a new one.
        '''The state is a list of 93 elements. \n
        The first 81 elements are the state of each square, 0 for empty, 1 for X and 2 for O. \n
        The next 9 elements are the result of each subgame: 0 while being played, 1 is win for X, 2 is a win for O and 3 for draw.\n
        The next element is the next symbol to play: 1 for X and 2 for O.\n
        The next element is the index of the subgame that is constrained, 9 for no subgame constraint. \n
        The last element is the result of the UTTT: 0 while being played, 1 is win for X, 2 is a win for O and 3 for draw.'''
        if state:
            self.state = state
        else:
            self.state = [0] * SIZE
            self.state[NEXT_SYMBOL_INDEX] = X_STATE_VALUE #X always starts the game
            self.state[CONSTRAINT_INDEX] = UNCONSTRAINED_STATE_VALUE #no subgame is constrained at the beginning


    @property
    def result(self) -> int:
        return self.state[RESULT_INDEX]

    @property
    def next_symbol(self) -> int:
        return self.state[NEXT_SYMBOL_INDEX]

    @property
    def constraint(self) -> int:
        return self.state[CONSTRAINT_INDEX]

    def copy(self):
        '''Returns a copy of the game.'''
        return UltimateTicTacToe(self.state)

    def get_winner(self) -> int:
        '''Returns the winner of the game.'''
        return self.state[RESULT_INDEX]

    def _get_state(self) -> list:
        '''Returns the state of the game.'''
        return self.state

    def is_game_over(self) -> bool:
        '''Returns True if the game is over, False otherwise.'''
        return bool(self.state[RESULT_INDEX])

    def is_constrained(self) -> bool:
        '''Returns True if a subgame is constrained, False otherwise.'''
        return self.state[CONSTRAINT_INDEX] != UNCONSTRAINED_STATE_VALUE

    def _verify_move(self,
                     move: Move): #Verifies if the move is valid.
        '''Verifies if the move is valid: if the index is in the valid range,
        if the subgame is not over, if the index is not already taken and if the subgame is not constrained.
          If it is not, it raises an exception.'''
        illegal_move = f"Illegal move {move} - "
        if not (0 <= move.index < 81):
            raise utttError(illegal_move + "index outside the valid range")
        if self.is_constrained() and self.constraint != move.index // 9:
            raise utttError(illegal_move + f"violated constraint = {self.constraint}")
        if self.state[81 + move.index // 9]:
            raise utttError(illegal_move + "index from terminated subgame")
        if self.state[move.index]:
            raise utttError(illegal_move + "index already taken")

    def _get_subgame_result(self,
                            subgame_index: int) -> int:   #Index of the subgame from 0 to 8
        '''Returns the result of the subgame.'''
        return self.state[81 + subgame_index]

    def _update_state(self,
                      move: Move): #Updates the state of the game after a move.
        '''Updates the state of the game after a move. It also verifies if the subgame and the game are over.'''
        self.state[move.index] = move.symbol
        self.state[NEXT_SYMBOL_INDEX] = X_STATE_VALUE + O_STATE_VALUE - move.symbol

        self._verify_subgame_result(move)
        self._verify_game_result(move)

        if not self.is_game_over():
            #Check if the subgame on index move.index % 9 is still being played. If it is, constraint to it. Else, unconstrain the game.
            subgame_index = move.index % 9
            if not self._get_subgame_result(subgame_index):
                self.state[CONSTRAINT_INDEX] = subgame_index
            else:
                self.state[CONSTRAINT_INDEX] = UNCONSTRAINED_STATE_VALUE

    def _make_move(self,
                move: Move, #Receives a move and updates the state of the game.
                verify: bool = True): #A boolean to verify if the move is valid.
        '''Makes a move in the game. It verifies if the move is valid'''
        if verify:
            if self.is_game_over():
                raise utttError("Illegal move " + str(move) + ' - The game is over')
            self._verify_move(move)

        self._update_state(move)
        self._verify_game_result(move)
        '''
        if self.is_game_over():
            print('The game is over')
            if self.state[RESULT_INDEX] == DRAW_STATE_VALUE:
                print('The game is a draw')
            elif self.state[RESULT_INDEX] == X_STATE_VALUE:
                print('X is the winner')
            else:
                print('O is the winner')
        '''
    def is_winning_position(self,
                            game:list) -> bool: #Receives a list of 9 elements and returns True if it is a winning position, False otherwise.
        '''Returns True if the game is a winning position, False otherwise.'''
        return game[0] == game[1] == game[2] != 0 or game[3] == game[4] == game[5] != 0 or game[6] == game[7] == game[8] != 0 or game[0] == game[3] == game[6] != 0 or game[1] == game[4] == game[7] != 0 or game[2] == game[5] == game[8] != 0 or game[0] == game[4] == game[8] != 0 or game[2] == game[4] == game[6] != 0

    def is_drawn_position(self,
                          game:list) -> bool: #Receives a list of 9 elements and returns True if it is a drawn position, False otherwise.
        '''Returns True if the game is a drawn position, False otherwise.'''
        return 0 not in game

    def _verify_subgame_result(self, move):
        '''Verifies if the subgame is over and updates the state of the subgame.'''
        subgame_index = move.index // 9
        subgame = self.state[subgame_index * 9 : subgame_index * 9 + 9]
        if self.is_winning_position(subgame):
            self.state[81 + subgame_index] = move.symbol
        elif self.is_drawn_position(subgame):
            self.state[81 + subgame_index] = DRAW_STATE_VALUE

    def _verify_game_result(self,move):
        '''Verifies if the game is over and updates the state of the game.'''
        symbol = move.symbol
        game = self.state[81:90]
        if self.is_winning_position(game):
            self.state[RESULT_INDEX] = symbol
        elif self.is_drawn_position(game):
            self.state[RESULT_INDEX] = DRAW_STATE_VALUE

    def _get_legal_indexes(self) -> list:
        '''Returns a list with the indexes of the legal moves.'''
        if not self.is_constrained():
            ## If the game is not constrained, all the empty squares are legal moves, except the ones from subgames that are already over.
            legal = []
            for subgame_index in range(9):
                if not self._get_subgame_result(subgame_index):
                    for i in range(subgame_index * 9, subgame_index * 9 + 9):
                        if not self.state[i]:
                            legal.append(i)
            return legal
        else:
            subgame_index = self.state[CONSTRAINT_INDEX]
            return [i for i in range(subgame_index * 9, subgame_index * 9 + 9) if not self.state[i]]

    def get_indexes_from_constraint(self):
      constraint = self.state[91]
      if constraint == 0:
        winning_lines = [[self.state[0], self.state[1], self.state[2]],
                          [self.state[3], self.state[4], self.state[5]],
                          [self.state[6], self.state[7], self.state[8]],
                          [self.state[0], self.state[3], self.state[6]],
                          [self.state[1], self.state[4], self.state[7]],
                          [self.state[2], self.state[5], self.state[8]],
                          [self.state[0], self.state[4], self.state[8]],
                          [self.state[2], self.state[4], self.state[8]]]
      elif constraint == 1:
        winning_lines= [[self.state[9], self.state[10], self.state[11]],
                        [self.state[12], self.state[13], self.state[14]],
                        [self.state[15], self.state[16], self.state[17]],
                        [self.state[9], self.state[12], self.state[15]],
                        [self.state[10], self.state[13], self.state[16]],
                        [self.state[11], self.state[14], self.state[17]],
                        [self.state[9], self.state[13], self.state[17]],
                        [self.state[11], self.state[13], self.state[15]]]

      elif constraint == 2:
        winning_lines= [[self.state[18], self.state[19], self.state[20]],
                        [self.state[21], self.state[22], self.state[23]],
                        [self.state[24], self.state[25], self.state[26]],
                        [self.state[18], self.state[21], self.state[24]],
                        [self.state[19], self.state[22], self.state[25]],
                        [self.state[20], self.state[23], self.state[26]],
                        [self.state[18], self.state[22], self.state[26]],
                        [self.state[20], self.state[22], self.state[24]]]

      elif constraint == 3:
        winning_lines= [[self.state[27], self.state[28], self.state[29]],
                        [self.state[30], self.state[31], self.state[32]],
                        [self.state[33], self.state[34], self.state[35]],
                        [self.state[27], self.state[30], self.state[33]],
                        [self.state[28], self.state[31], self.state[34]],
                        [self.state[29], self.state[32], self.state[35]],
                        [self.state[27], self.state[31], self.state[35]],
                        [self.state[29], self.state[31], self.state[33]]]

      elif constraint == 4:
        winning_lines= [[self.state[36], self.state[37], self.state[38]],
                        [self.state[39], self.state[40], self.state[41]],
                        [self.state[42], self.state[43], self.state[44]],
                        [self.state[36], self.state[39], self.state[42]],
                        [self.state[37], self.state[40], self.state[43]],
                        [self.state[38], self.state[41], self.state[44]],
                        [self.state[36], self.state[40], self.state[44]],
                        [self.state[38], self.state[40], self.state[42]]]

      elif constraint == 5:
        winning_lines = [[self.state[45], self.state[46], self.state[47]],
                        [self.state[48], self.state[49], self.state[50]],
                        [self.state[51], self.state[52], self.state[53]],
                        [self.state[45], self.state[48], self.state[51]],
                        [self.state[46], self.state[49], self.state[52]],
                        [self.state[47], self.state[50], self.state[53]],
                        [self.state[45], self.state[49], self.state[53]],
                        [self.state[47], self.state[49], self.state[51]]]

      elif constraint == 6:
        winning_lines= [[self.state[54], self.state[55], self.state[56]],
                        [self.state[57], self.state[58], self.state[59]],
                        [self.state[60], self.state[61], self.state[62]],
                        [self.state[54], self.state[57], self.state[60]],
                        [self.state[55], self.state[58], self.state[61]],
                        [self.state[56], self.state[59], self.state[62]],
                        [self.state[54], self.state[58], self.state[62]],
                        [self.state[56], self.state[58], self.state[60]]]

      elif constraint == 7:
        winning_lines= [[self.state[63], self.state[64], self.state[65]],
                        [self.state[66], self.state[67], self.state[68]],
                        [self.state[69], self.state[70], self.state[71]],
                        [self.state[63], self.state[66], self.state[69]],
                        [self.state[64], self.state[67], self.state[70]],
                        [self.state[65], self.state[68], self.state[71]],
                        [self.state[63], self.state[67], self.state[71]],
                        [self.state[65], self.state[67], self.state[69]]]

      elif constraint == 8:
        winning_lines = [[self.state[72], self.state[73], self.state[74]],
                        [self.state[75], self.state[76], self.state[77]],
                        [self.state[78], self.state[79], self.state[80]],
                        [self.state[72], self.state[75], self.state[78]],
                        [self.state[73], self.state[76], self.state[79]],
                        [self.state[74], self.state[77], self.state[80]],
                        [self.state[72], self.state[76], self.state[80]],
                        [self.state[74], self.state[76], self.state[78]]]
      return winning_lines

    def __str__(self):
        state_values_map = {
            X_STATE_VALUE: 'X',
            O_STATE_VALUE: 'O',
            DRAW_STATE_VALUE: '=',
            0: '·',
        }

        subgames = [state_values_map[s] for s in self.state[:81]]
        supergame = [state_values_map[s] for s in self.state[81:90]]

        if not self.is_game_over():
            legal_indexes = self._get_legal_indexes()
            for legal_index in legal_indexes:
                subgames[legal_index] = '•'

            if self.is_constrained():
                supergame[self.constraint] = '•'
            elif self.constraint == UNCONSTRAINED_STATE_VALUE:
                supergame = ['•' if s == '·' else s for s in supergame]

        sb = lambda l, r: ' '.join(subgames[l : r + 1])
        sp = lambda l, r: ' '.join(supergame[l : r + 1])

        subgames_str = [
            '    1 2 3   4 5 6   7 8 9',
            '  1 ' + sb(0, 2)   + ' │ ' + sb(9, 11) +  ' │ ' + sb(18, 20),
            '  2 ' + sb(3, 5)   + ' │ ' + sb(12, 14) + ' │ ' + sb(21, 23),
            '  3 ' + sb(6, 8)   + ' │ ' + sb(15, 17) + ' │ ' + sb(24, 26),
            '    ' + '—' * 21,
            '  4 ' + sb(27, 29) + ' │ ' + sb(36, 38) + ' │ ' + sb(45, 47),
            '  5 ' + sb(30, 32) + ' │ ' + sb(39, 41) + ' │ ' + sb(48, 50),
            '  6 ' + sb(33, 35) + ' │ ' + sb(42, 44) + ' │ ' + sb(51, 53),
            '    ' + '—' * 21,
            '  7 ' + sb(54, 56) + ' │ ' + sb(63, 65) + ' │ ' + sb(72, 74),
            '  8 ' + sb(57, 59) + ' │ ' + sb(66, 68) + ' │ ' + sb(75, 77),
            '  9 ' + sb(60, 62) + ' │ ' + sb(69, 71) + ' │ ' + sb(78, 80),
        ]

        supergame_str = [
            '  ' + sp(0, 2),
            '  ' + sp(3, 5),
            '  ' + sp(6, 8),
        ]

        subgames_str = '\n'.join(subgames_str)
        supergame_str = '\n'.join(supergame_str)

        next_symbol = state_values_map[self.next_symbol]
        constraint = 'None' if self.constraint == UNCONSTRAINED_STATE_VALUE else str(self.constraint+1)
        result = 'In Game'
        if self.result == X_STATE_VALUE:
            result = 'X won'
        elif self.result == O_STATE_VALUE:
            result = 'O won'
        elif self.result == DRAW_STATE_VALUE:
            result = 'Draw'

        output = f'{self.__class__.__name__}(\n'
        output += f'  subgames:\n{subgames_str}\n'
        if not self.is_game_over():
            output += f'  next player: {next_symbol}\n'
            output += f'  constraint: {constraint}\n'
        output += f'  supergame:\n{supergame_str}\n'
        output += f'  result: {result}\n)'
        return output

    def map_matrix_to_subgame(self, matrix_index: int) -> int:
        digit_1 = matrix_index // 10
        digit_2 = matrix_index % 10
        subgame_index = MAPPING[(digit_1, digit_2)]
        return subgame_index

    def play(self, matrix_index: int):
        '''Plays the game. Receives a matrix index and makes a move in the corresponding state index.'''
        index = self.map_matrix_to_subgame(matrix_index)
        self._make_move(Move(self.next_symbol, index))
        #print(self)
        
class utttError(Exception):
    pass

def subgame_heuristic(nuevo_juego, ficha):
  if nuevo_juego.state[91] != 9:
    winning_lines_list = nuevo_juego.get_indexes_from_constraint()
    winning_lines = 0
    winning_lines_opponent = 0

    if ficha == 'X':
      for line in winning_lines_list:
        sum = 0
        for num in line:
          sum += num
        if (sum == 2) & (2 not in line):
          winning_lines += 10
        elif (sum == 4) & (1 not in line):
          winning_lines_opponent -= 10

    else:
          for line in winning_lines_list:
            sum = 0
            for num in line:
              sum += num
            if (sum == 4) & (1 not in line):
              winning_lines += 10
            elif (sum == 2) & (2 not in line):
              winning_lines_opponent -= 10

    result = winning_lines + winning_lines_opponent

  else:
    result = -80

  return result

def macrogame_winning_lines(nuevo_juego, ficha):
  
  winning_lines_list = [[nuevo_juego.state[81], nuevo_juego.state[82], nuevo_juego.state[83]],
                        [nuevo_juego.state[84], nuevo_juego.state[85], nuevo_juego.state[86]],
                        [nuevo_juego.state[87], nuevo_juego.state[88], nuevo_juego.state[89]],
                        [nuevo_juego.state[81], nuevo_juego.state[84], nuevo_juego.state[87]],
                        [nuevo_juego.state[82], nuevo_juego.state[85], nuevo_juego.state[88]],
                        [nuevo_juego.state[83], nuevo_juego.state[86], nuevo_juego.state[89]],
                        [nuevo_juego.state[81], nuevo_juego.state[85], nuevo_juego.state[89]],
                        [nuevo_juego.state[83], nuevo_juego.state[85], nuevo_juego.state[87]]]

  winning_lines = 0
  winning_lines_opponent = 0

  if ficha == 'X':
    for line in winning_lines_list:
      sum = 0
      for num in line:
        sum += num
      if (sum == 2) & (2 not in line):
        winning_lines += 25
      elif (sum == 4) & (1 not in line):
        winning_lines_opponent -= 25

  else:
        for line in winning_lines_list:
          sum = 0
          for num in line:
            sum += num
          if (sum == 4) & (1 not in line):
            winning_lines += 25
          elif (sum == 2) & (2 not in line):
            winning_lines_opponent -= 25

  result = winning_lines + winning_lines_opponent

  
  return result


def macrogame_heuristic(nuevo_juego, ficha):

 # Define los valores de las piezas
  if ficha == 'X':
      my_value = 1
      opponent_value = 2
  else:
      my_value = 2
      opponent_value = 1


  if nuevo_juego.state[92] == 0:
    macrogame_state = nuevo_juego.state[81:90]

    # Define las posiciones de las esquinas, los medios y el centro
    corners = [0, 2, 6, 8]
    middles = [1, 3, 5, 7]
    center = 4

    # Inicializa la puntuación total
    score = 0

    # Revisa las esquinas
    for i in corners:
        if macrogame_state[i] == my_value:
            score += 50
        elif macrogame_state[i] == opponent_value:
            score -= 30

    # Revisa el centro
    if macrogame_state[center] == my_value:
        score += 30
    elif macrogame_state[center] == opponent_value:
        score -= 50

    # Revisa los medios
    for i in middles:
        if macrogame_state[i] == my_value:
            score += 10
        elif macrogame_state[i] == opponent_value:
            score -= 10

  elif nuevo_juego.state[91] == my_value:
    score = math.inf
  else:
    score = -math.inf

  return score

def game_heuristic(nuevo_juego,ficha):
  return subgame_heuristic(nuevo_juego,ficha) + macrogame_heuristic(nuevo_juego,ficha)/2 + macrogame_winning_lines(nuevo_juego, ficha)


def solucion_corregida(index):
    if index in REVERSE_MAPPING:
        row, col = REVERSE_MAPPING[index]
        return row * 10 + col
    else:
        return None  # Return None if index is not valid


def _minimax(juego, profundidad, turno, ficha):
    global movimiento_optimo

    # Condición de parada: profundidad cero o juego terminado.
    if profundidad == 0 or juego.is_game_over():
        return game_heuristic(juego, ficha)

    # Obtiene todos los movimientos legales posibles.
    frontera = juego._get_legal_indexes().copy()

    if turno:  # Turno de la IA.
        best_value = -math.inf
        for elemento in frontera:
            nuevo_juego = UltimateTicTacToe(state=juego.state.copy())
            nuevo_juego.play(solucion_corregida(elemento))  # Realiza el movimiento en una copia del juego.
            valor_heuristico = _minimax(nuevo_juego, profundidad - 1, False, ficha)
            if best_value <= valor_heuristico:
                if profundidad == 5:  # Cambia esto por la profundidad inicial si es diferente.
                    movimiento_optimo = elemento
                best_value = valor_heuristico
        return best_value
    else:  # Turno del oponente.
        best_value = math.inf
        for elemento in frontera:
            nuevo_juego = UltimateTicTacToe(state=juego.state.copy())
            nuevo_juego.play(solucion_corregida(elemento))  # Realiza el movimiento en una copia del juego.
            valor_heuristico = _minimax(nuevo_juego, profundidad - 1, True, ficha)
            if best_value >= valor_heuristico:
                best_value = valor_heuristico
        return best_value

def minimax(uttt, profundidad, turno, ficha):
    global movimiento_optimo
    movimiento_optimo = None  # Asegúrate de reiniciar el movimiento óptimo antes de empezar.
    _minimax(uttt, profundidad, turno, ficha)
    return movimiento_optimo


class Ejecutable:
    @staticmethod
    def main():
        juego = UltimateTicTacToe()

        # quien inicia
        s = input("¿Quién inicia? Si quieres que inicie la inteligencia artificial, escribe IA. De lo contrario, escribe YO: ")
        if s == 'IA':
            qi = 1
            ficha = "X"
            # Hacer que la IA realice el primer movimiento
            jugada_ia = solucion_corregida(minimax(juego, 5, True, ficha))
            if jugada_ia is not None:
                juego.play(jugada_ia)
                print("Movimiento de la IA:")
                print(juego)
            else:
                print("No se pudo obtener una jugada inicial de la IA.")
        else:
            qi = 0  # si qi = 0, iniciamos nosotros; si qi = 1 inicia la IA
            ficha = "O"

        print(juego)

        game_over = False
        while not game_over:
            # Turno del jugador humano
            flag = True
            while flag:
                print("Es tu turno")
                jugada = input("Ingresa la casilla que quieras jugar: ")
                flag = False
                try:
                    juego.play(int(jugada))
                    print(juego)
                    game_over = juego.is_game_over()
                except utttError as e:
                    print('Jugada no válida')
                    flag = True
                if game_over:
                    break

            # Turno de la IA
            if not game_over:
                print("Turno de la IA")
                jugada_ia = solucion_corregida(minimax(juego, 5, True, ficha))
                if jugada_ia is not None:
                    juego.play(jugada_ia)
                    print("Movimiento de la IA:", jugada_ia)
                    print(juego)
                    game_over = juego.is_game_over()
                else:
                    print("La IA no pudo encontrar una jugada válida.")

        # Determinar el ganador
        if juego.get_winner() == X_STATE_VALUE:
            ganador = "X"
        elif juego.get_winner() == O_STATE_VALUE:
            ganador = "O"
        else:
            ganador = "Nadie (empate)"
        print("El juego ha terminado. Ganador: ", ganador)



     
class EjecutableDosJugadores:
    def __init__(self) -> None:
        self.jugador_actual = None
  
    def main():
        juego = UltimateTicTacToe()
        print(juego)
        
        game_over = False
        while not game_over:
            flag = True
            while flag:
                jugada = input("Ingresa la casilla en la que juega X: ") 
                flag = False
                try:  
                    juego.play(int(jugada))
                    game_over = juego.is_game_over()
                    print(juego)
                except utttError as e:
                    print('jugada no valida')
                    flag = True
                if game_over:
                    break

                flag = True
                while flag:
                    jugada = input("Ingresa la casilla en la que juega O: ") 
                    flag = False
                    try:  
                        juego.play(int(jugada))
                        game_over = juego.is_game_over()
                        print(juego)
                    except utttError as e:
                        print('jugada no valida')
                        flag = True
        if juego.get_winner() == 1:
            ganador = "X"
        else:
            ganador = "O"
        print("El juego ha terminado. Ganador: ", ganador)
  

# Ejecutar el juego
Ejecutable.main()