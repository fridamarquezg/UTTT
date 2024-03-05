import math  # Import the math module for mathematical operations.

# Constants representing the structure and state of an ultimate tic-tac-toe game board.
SIZE = 93  # Total number of indices used to represent the game state.
NEXT_SYMBOL_INDEX = 90  # Index for storing the next player symbol (X or O).
CONSTRAINT_INDEX = 91  # Index for storing the constraint for the next move (which subgame to play).
RESULT_INDEX = 92  # Index for storing the overall game result.

# Constants representing the state values for the game.
X_STATE_VALUE = 1  # State value representing the 'X' player.
O_STATE_VALUE = 2  # State value representing the 'O' player.
DRAW_STATE_VALUE = 3  # State value representing a draw.
UNCONSTRAINED_STATE_VALUE = 9  # State value when there are no constraints on the next move.

# Mapping from (row, column) positions in the ultimate tic-tac-toe board to their respective index in the game state array.
MAPPING = {
    # First subgame (top-left 3x3 grid)
    (1, 1): 0, (1, 2): 1, (1, 3): 2,
    (2, 1): 3, (2, 2): 4, (2, 3): 5,
    (3, 1): 6, (3, 2): 7, (3, 3): 8,

    # Second subgame (top-middle 3x3 grid)
    (1, 4): 9, (1, 5): 10, (1, 6): 11,
    (2, 4): 12, (2, 5): 13, (2, 6): 14,
    (3, 4): 15, (3, 5): 16, (3, 6): 17,

    # Third subgame (top-right 3x3 grid)
    (1,7):18, (1,8):19, (1,9):20,
    (2,7):21, (2,8):22, (2,9):23,
    (3,7):24, (3,8):25, (3,9):26,

    # Fourth subgame (middle-left 3x3 grid)
    (4,1):27, (4,2):28, (4,3):29,
    (5,1):30, (5,2):31, (5,3):32,
    (6,1):33, (6,2):34, (6,3):35,

    # Fifth subgame (center 3x3 grid)
    (4,4):36, (4,5):37, (4,6):38,
    (5,4):39, (5,5):40, (5,6):41,
    (6,4):42, (6,5):43, (6,6):44,

    # Sixth subgame (middle-rigth 3x3 grid)
    (4,7):45, (4,8):46, (4,9):47,
    (5,7):48, (5,8):49, (5,9):50,
    (6,7):51, (6,8):52, (6,9):53,

    # Seventh subgame (bottom-left 3x3 grid)
    (7,1):54, (7,2):55, (7,3):56,
    (8,1):57, (8,2):58, (8,3):59,
    (9,1):60, (9,2):61, (9,3):62,

    # Eighth subgame (bottom-middle 3x3 grid)
    (7,4):63, (7,5):64, (7,6):65,
    (8,4):66, (8,5):67, (8,6):68,
    (9,4):69, (9,5):70, (9,6):71,

    # Ninth subgame (bottom-right 3x3 grid)
    (7, 7): 72, (7, 8): 73, (7, 9): 74,
    (8, 7): 75, (8, 8): 76, (8, 9): 77,
    (9, 7): 78, (9, 8): 79, (9, 9): 80,
}
# Reverse mapping for converting back from indices to (row, column) positions.
REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}  # Create a reverse mapping from index to (row, column).

class Move:
    def __init__(self, symbol: int, index: int):
        """
        Initializes a new instance of the Move class.

        Parameters:
        symbol (int): Represents the player making the move, using constants X_STATE_VALUE (1) or O_STATE_VALUE (2).
        index (int): The position (ranging from 0 to 80) on the ultimate tic-tac-toe board where the symbol will be placed.
        """
        self.symbol = symbol  # Store the player's symbol (X or O) as an integer (X_STATE_VALUE = 1 or O_STATE_VALUE = 2).
        self.index = index    # Store the board position for the move (int from 0 to 80).

    def __str__(self):
        """
        Returns a string representation of the Move instance, which is useful for debugging.

        The output will be in the format 'Move(symbol=X, index=index)' or 'Move(symbol=O, index=index)',
        where 'X' or 'O' corresponds to the player's symbol and 'index' is the position on the board.
        """
        output = '{cls}(symbol={symbol}, index={index})'  # Template for the output string.
        output = output.format(
            cls='Move',  # Class name
            symbol={X_STATE_VALUE: 'X', O_STATE_VALUE: 'O'}[self.symbol],  # Convert the symbol from int to string ('X' or 'O').
            index=self.index,  # Include the move index (board position).
        )
        return output  # Return the formatted string representation of the move.


class UltimateTicTacToe:
    def __init__(self, state=None):  # Initialize a new Ultimate Tic Tac Toe game.
        '''The state is a list of 93 elements representing the entire game board.
        The first 81 elements represent the state of each square: 0 for empty, 1 for X, and 2 for O.
        The next 9 elements represent the result of each subgame: 0 while being played, 1 is win for X, 2 is win for O, and 3 for a draw.
        The 91st element is the next symbol to play: 1 for X and 2 for O.
        The 92nd element is the index of the constrained subgame, with 9 representing no constraints.
        The last element (93rd) represents the result of the Ultimate Tic Tac Toe game: 0 while being played, 1 is win for X, 2 is win for O, and 3 for a draw.'''
        if state:
            self.state = state  # Use the provided state if one is given.
        else:
            self.state = [0] * SIZE  # Otherwise, initialize a new game state.
            self.state[NEXT_SYMBOL_INDEX] = X_STATE_VALUE  # X always starts the game.
            self.state[CONSTRAINT_INDEX] = UNCONSTRAINED_STATE_VALUE  # No subgame constraint at the beginning.

    @property
    def result(self) -> int:
        '''Returns the overall game result.'''
        return self.state[RESULT_INDEX]

    @property
    def next_symbol(self) -> int:
        '''Returns the symbol (X or O) of the player who is next to move.'''
        return self.state[NEXT_SYMBOL_INDEX]

    @property
    def constraint(self) -> int:
        '''Returns the index of the currently constrained subgame, or 9 if there is no constraint.'''
        return self.state[CONSTRAINT_INDEX]

    def copy(self):
        '''Returns a copy of the current game state, allowing for simulation or backtracking without altering the original state.'''
        return UltimateTicTacToe(list(self.state))  # Pass a copy of the state to ensure the original is not modified.

    def get_winner(self) -> int:
        '''Returns the winner of the game, if there is one.'''
        return self.state[RESULT_INDEX]

    def _get_state(self) -> list:
        '''Returns the current game state.'''
        return self.state

    def is_game_over(self) -> bool:
        '''Checks if the game has ended, which is true if the RESULT_INDEX is not zero (indicating a win or a draw).'''
        return bool(self.state[RESULT_INDEX])

    def is_constrained(self) -> bool:
        '''Checks if the current move is constrained to a specific subgame (if CONSTRAINT_INDEX is not 9).'''
        return self.state[CONSTRAINT_INDEX] != UNCONSTRAINED_STATE_VALUE
 
    def _verify_move(self, move: Move):  
        '''Checks if the move is valid: it checks if the index is within the valid range (0 to 80),
        if the subgame is not already over, if the index is not already taken, and if the move does not violate the current subgame constraint.
        If the move is invalid, it raises an exception.'''
        illegal_move = f"Illegal move {move} - "
        if not (0 <= move.index < 81):
            raise utttError(illegal_move + "index outside the valid range")
        if self.is_constrained() and self.constraint != move.index // 9:
            raise utttError(illegal_move + f"violated constraint = {self.constraint}")
        if self.state[81 + move.index // 9]:
            raise utttError(illegal_move + "index from terminated subgame")
        if self.state[move.index]:
            raise utttError(illegal_move + "index already taken")

    def _get_subgame_result(self, subgame_index: int) -> int:  
        '''Returns the result of the specified subgame, indicated by the subgame_index (range 0 to 8).
        The result can be 0 (still playing), 1 (win for X), 2 (win for O), or 3 (draw).'''
        return self.state[81 + subgame_index]

    def _update_state(self, move: Move):  
        '''Updates the game state after a move has been made. This includes setting the square to the symbol of the player,
        updating the player who has the next move, checking if the current subgame or the entire game has ended,
        and updating the subgame constraint accordingly.'''
        self.state[move.index] = move.symbol
        self.state[NEXT_SYMBOL_INDEX] = X_STATE_VALUE + O_STATE_VALUE - move.symbol

        self._verify_subgame_result(move)
        self._verify_game_result(move)

        if not self.is_game_over():
            # If the game is not over, check if the subgame of the current move is still active. If so, constrain to it; otherwise, remove constraints.
            subgame_index = move.index % 9
            if not self._get_subgame_result(subgame_index):
                self.state[CONSTRAINT_INDEX] = subgame_index
            else:
                self.state[CONSTRAINT_INDEX] = UNCONSTRAINED_STATE_VALUE

    def _make_move(self, move: Move, verify: bool = True):  
        '''Performs a move in the game after validating it (if verification is enabled).
        It checks whether the game is already over; if so, it raises an exception. Otherwise, it updates the state of the game accordingly.'''
        if verify:
            if self.is_game_over():
                raise utttError("Illegal move " + str(move) + ' - The game is over')
            self._verify_move(move)

        self._update_state(move)
        self._verify_game_result(move)

    def is_winning_position(self, game: list) -> bool:  
        '''Evaluates a list representing a subgame (9 elements) and returns True if it represents a winning configuration,
        otherwise returns False.'''
        return game[0] == game[1] == game[2] != 0 or game[3] == game[4] == game[5] != 0 or game[6] == game[7] == game[8] != 0 or game[0] == game[3] == game[6] != 0 or game[1] == game[4] == game[7] != 0 or game[2] == game[5] == game[8] != 0 or game[0] == game[4] == game[8] != 0 or game[2] == game[4] == game[6] != 0

    def is_drawn_position(self, game: list) -> bool:  
        '''Evaluates a list representing a subgame (9 elements) and returns True if it represents a drawn configuration,
        otherwise returns False.'''
        return 0 not in game

    def _verify_subgame_result(self, move): 
        '''Checks if the subgame to which the move was made is over and updates the game state accordingly.'''
        subgame_index = move.index // 9
        subgame = self.state[subgame_index * 9 : subgame_index * 9 + 9]
        if self.is_winning_position(subgame):
            self.state[81 + subgame_index] = move.symbol
        elif self.is_drawn_position(subgame):
            self.state[81 + subgame_index] = DRAW_STATE_VALUE

    def _verify_game_result(self, move):  
        '''Checks if the entire game is over following a move and updates the overall game result accordingly.'''
        symbol = move.symbol
        game = self.state[81:90]  # Check the results of all subgames.
        if self.is_winning_position(game):
            self.state[RESULT_INDEX] = symbol
        elif self.is_drawn_position(game):
            self.state[RESULT_INDEX] = DRAW_STATE_VALUE

    def _get_legal_indexes(self) -> list:  
        '''Generates and returns a list of indexes that represent legal moves based on the current game state and constraints.'''
        if not self.is_constrained():
            ## If the game is not constrained, all the empty squares are legal moves, except the ones from subgames that are already taken.
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
        '''Returns the list of 'winning lines' (lists) of the current constraint subgame. We define the winning lines as sets of three indexes that form a line.'''
        constraint = self.state[CONSTRAINT_INDEX]  # Retrieves the current subgame constraint from the game state.
        winning_lines = []  # This will store the lines (rows, columns, diagonals) of the constrained subgame.

        # Depending on which subgame is constrained, select the corresponding squares from the game state.
        # Each 'if' block corresponds to one of the nine subgames in Ultimate Tic Tac Toe.
        # For each subgame, we define the winning lines.
        if constraint == 0:
            winning_lines = [
                [self.state[0], self.state[1], self.state[2]],  # First row
                [self.state[3], self.state[4], self.state[5]],  # Second row
                [self.state[6], self.state[7], self.state[8]],  # Third row
                [self.state[0], self.state[3], self.state[6]],  # First column
                [self.state[1], self.state[4], self.state[7]],  # Second column
                [self.state[2], self.state[5], self.state[8]],  # Third column
                [self.state[0], self.state[4], self.state[8]],  # First diagonal
                [self.state[2], self.state[4], self.state[6]]]  # Second diagonal
        
        # And so on for each subgame
        elif constraint == 1:
            winning_lines= [
                [self.state[9], self.state[10], self.state[11]],
                [self.state[12], self.state[13], self.state[14]],
                [self.state[15], self.state[16], self.state[17]],
                [self.state[9], self.state[12], self.state[15]],
                [self.state[10], self.state[13], self.state[16]],
                [self.state[11], self.state[14], self.state[17]],
                [self.state[9], self.state[13], self.state[17]],
                [self.state[11], self.state[13], self.state[15]]]

        elif constraint == 2:
            winning_lines= [
                [self.state[18], self.state[19], self.state[20]],
                [self.state[21], self.state[22], self.state[23]],
                [self.state[24], self.state[25], self.state[26]],
                [self.state[18], self.state[21], self.state[24]],
                [self.state[19], self.state[22], self.state[25]],
                [self.state[20], self.state[23], self.state[26]],
                [self.state[18], self.state[22], self.state[26]],
                [self.state[20], self.state[22], self.state[24]]]

        elif constraint == 3:
            winning_lines= [
                [self.state[27], self.state[28], self.state[29]],
                [self.state[30], self.state[31], self.state[32]],
                [self.state[33], self.state[34], self.state[35]],
                [self.state[27], self.state[30], self.state[33]],
                [self.state[28], self.state[31], self.state[34]],
                [self.state[29], self.state[32], self.state[35]],
                [self.state[27], self.state[31], self.state[35]],
                [self.state[29], self.state[31], self.state[33]]]

        elif constraint == 4:
            winning_lines= [
                [self.state[36], self.state[37], self.state[38]],
                [self.state[39], self.state[40], self.state[41]],
                [self.state[42], self.state[43], self.state[44]],
                [self.state[36], self.state[39], self.state[42]],
                [self.state[37], self.state[40], self.state[43]],
                [self.state[38], self.state[41], self.state[44]],
                [self.state[36], self.state[40], self.state[44]],
                [self.state[38], self.state[40], self.state[42]]]

        elif constraint == 5:
            winning_lines = [
                [self.state[45], self.state[46], self.state[47]],
                [self.state[48], self.state[49], self.state[50]],
                [self.state[51], self.state[52], self.state[53]],
                [self.state[45], self.state[48], self.state[51]],
                [self.state[46], self.state[49], self.state[52]],
                [self.state[47], self.state[50], self.state[53]],
                [self.state[45], self.state[49], self.state[53]],
                [self.state[47], self.state[49], self.state[51]]]

        elif constraint == 6:
            winning_lines= [
                [self.state[54], self.state[55], self.state[56]],
                [self.state[57], self.state[58], self.state[59]],
                [self.state[60], self.state[61], self.state[62]],
                [self.state[54], self.state[57], self.state[60]],
                [self.state[55], self.state[58], self.state[61]],
                [self.state[56], self.state[59], self.state[62]],
                [self.state[54], self.state[58], self.state[62]],
                [self.state[56], self.state[58], self.state[60]]]

        elif constraint == 7:
            winning_lines= [
                [self.state[63], self.state[64], self.state[65]],
                [self.state[66], self.state[67], self.state[68]],
                [self.state[69], self.state[70], self.state[71]],
                [self.state[63], self.state[66], self.state[69]],
                [self.state[64], self.state[67], self.state[70]],
                [self.state[65], self.state[68], self.state[71]],
                [self.state[63], self.state[67], self.state[71]],
                [self.state[65], self.state[67], self.state[69]]]

        elif constraint == 8:
            winning_lines = [
                [self.state[72], self.state[73], self.state[74]],
                [self.state[75], self.state[76], self.state[77]],
                [self.state[78], self.state[79], self.state[80]],
                [self.state[72], self.state[75], self.state[78]],
                [self.state[73], self.state[76], self.state[79]],
                [self.state[74], self.state[77], self.state[80]],
                [self.state[72], self.state[76], self.state[80]],
                [self.state[74], self.state[76], self.state[78]]]
        
        # The function then returns the winning lines for the current constrained subgame.
        return winning_lines

    def __str__(self):
        # Map game symbols to their respective string representations.
        state_values_map = {
            X_STATE_VALUE: 'X',
            O_STATE_VALUE: 'O',
            DRAW_STATE_VALUE: '=',
            0: '·',  # Represents an empty space.
        }

        # Convert the game state into string representations for subgames and the supergame.
        subgames = [state_values_map[s] for s in self.state[:81]]  # The individual squares of all nine subgames.
        supergame = [state_values_map[s] for s in self.state[81:90]]  # The results of each of the nine subgames.

        # Mark legal moves in the subgames and supergame if the game is still ongoing.
        if not self.is_game_over():
            legal_indexes = self._get_legal_indexes()  # Get all legal moves based on current game state.
            for legal_index in legal_indexes:  # Highlight legal moves in the subgames.
                subgames[legal_index] = '•'

            # Highlight the constrained subgame or all possible moves in the supergame if no constraint.
            if self.is_constrained():
                supergame[self.constraint] = '•'
            elif self.constraint == UNCONSTRAINED_STATE_VALUE:
                supergame = ['•' if s == '·' else s for s in supergame]

        # Helper functions to format rows of the subgames and supergame for printing.
        sb = lambda l, r: ' '.join(subgames[l : r + 1])  # Formats a row of a subgame.
        sp = lambda l, r: ' '.join(supergame[l : r + 1])  # Formats a row of the supergame.

        # Format the string representation of the subgames and supergame.
        # Formatted strings for each row of the subgames.
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

        # Formatted strings for each row of the supergame.
        supergame_str = [
            '  ' + sp(0, 2),
            '  ' + sp(3, 5),
            '  ' + sp(6, 8),
        ]

        # Concatenate all parts to form the complete game board representation.
        subgames_str = '\n'.join(subgames_str)
        supergame_str = '\n'.join(supergame_str)

        # Determine the next symbol to play, the current constraint, and the game result.
        next_symbol = state_values_map[self.next_symbol]
        constraint = 'None' if self.constraint == UNCONSTRAINED_STATE_VALUE else str(self.constraint + 1)
        result = 'In Game'  # Default result if the game is still ongoing.
        if self.result == X_STATE_VALUE:
            result = 'X won'
        elif self.result == O_STATE_VALUE:
            result = 'O won'
        elif self.result == DRAW_STATE_VALUE:
            result = 'Draw'

        # Compile the final string representation of the game state, including board state and game info.
        output = f'{self.__class__.__name__}(\n'
        output += f'  subgames:\n{subgames_str}\n'
        if not self.is_game_over():
            output += f'  next player: {next_symbol}\n'
            output += f'  constraint: {constraint}\n'
        output += f'  supergame:\n{supergame_str}\n'
        output += f'  result: {result}\n)'
        return output

    def map_matrix_to_subgame(self, matrix_index: int) -> int:
        '''Converts a matrix index to a subgame index using the MAPPING dictionary.'''
        digit_1 = matrix_index // 10  # Extracts the tens place.
        digit_2 = matrix_index % 10   # Extracts the units place.
        subgame_index = MAPPING[(digit_1, digit_2)]  # Maps the two digits to the corresponding subgame index.
        return subgame_index

    def play(self, matrix_index: int):
        '''Receives a matrix index and performs the corresponding move in the game.'''
        index = self.map_matrix_to_subgame(matrix_index)  # Converts matrix index to game state index.
        self._make_move(Move(self.next_symbol, index))  # Executes the move in the game.

class utttError(Exception):
    # Custom exception class for Ultimate Tic Tac Toe errors.
    # Inherits from the built-in Exception class.
    pass


def subgame_heuristic(nuevo_juego, symbol):
    ''' Evaluates the current subgame board based on the given symbol. 
        It calculates potential winning lines for the current player and the opponent.'''
    if nuevo_juego.state[91] != 9:  # Check if the game is constrained to a specific subgame.
        winning_lines_list = nuevo_juego.get_indexes_from_constraint()  # Get the winning lines for the constrained subgame.
        winning_lines = 0  # Initialize the count of potential winning lines for the current player.
        winning_lines_opponent = 0  # Initialize the count of potential winning lines for the opponent.

        # Calculate scores based on the current player's symbol.
        if symbol == 'X':
            for line in winning_lines_list:
                sum = 0
                for num in line:
                    sum += num  # Sum the values in the current line.
                if (sum == 2) & (2 not in line):  # Check for two 'X's and no 'O's.
                    winning_lines += 10  # Gives a positive value for each potential winning line for 'X'.
                elif (sum == 4) & (1 not in line):  # Check for two 'O's and no 'X's.
                    winning_lines_opponent -= 10  # Gives a negative value for each potential winning line for 'O'.
        else:  # If the current player is 'O'.
            for line in winning_lines_list:
                sum = 0
                for num in line:
                    sum += num  # Sum the values in the current line.
                if (sum == 4) & (1 not in line):  # Check for two 'O's and no 'X's.
                    winning_lines += 10  # Gives a positive value for each potential winning line for 'O'.
                elif (sum == 2) & (2 not in line):  # Check for two 'X's and no 'O's.
                    winning_lines_opponent -= 10  # Gives a negative value for each potential winning line for 'X'.

        result = winning_lines + winning_lines_opponent  # Calculate the subgame heuristic value.

    else:
        result = -80  # If the game is not constrained, return a default heuristic value.

    return result

def macrogame_winning_lines(nuevo_juego, symbol):
    ''' Evaluates the entire board (macrogame) based on the given symbol.
        It calculates potential winning lines for the current player and the opponent.'''
    
    # Define all possible winning lines in the macrogame (supergame).
    winning_lines_list = [[nuevo_juego.state[81], nuevo_juego.state[82], nuevo_juego.state[83]],
                            [nuevo_juego.state[84], nuevo_juego.state[85], nuevo_juego.state[86]],
                            [nuevo_juego.state[87], nuevo_juego.state[88], nuevo_juego.state[89]],
                            [nuevo_juego.state[81], nuevo_juego.state[84], nuevo_juego.state[87]],
                            [nuevo_juego.state[82], nuevo_juego.state[85], nuevo_juego.state[88]],
                            [nuevo_juego.state[83], nuevo_juego.state[86], nuevo_juego.state[89]],
                            [nuevo_juego.state[81], nuevo_juego.state[85], nuevo_juego.state[89]],
                            [nuevo_juego.state[83], nuevo_juego.state[85], nuevo_juego.state[87]]]
    
    winning_lines = 0  # Initialize the count of potential winning lines for the current player.
    winning_lines_opponent = 0  # Initialize the count of potential winning lines for the opponent.

    # Calculate scores based on the current player's symbol.
    if symbol == 'X':
        for line in winning_lines_list:
            sum = 0
            for num in line:
                sum += num  # Sum the values in the current line.
            if (sum == 2) & (2 not in line):  # Check for two 'X's and no 'O's.
                winning_lines += 25  # Gives a positive value for each potential winning line for 'X'.
            elif (sum == 4) & (1 not in line):  # Check for two 'O's and no 'X's.
                winning_lines_opponent -= 25  # Gives a negative value for each potential winning line for 'O'.
    else:  # If the current player is 'O'.
        for line in winning_lines_list:
            sum = 0
            for num in line:
                sum += num  # Sum the values in the current line.
            if (sum == 4) & (1 not in line):  # Check for two 'O's and no 'X's.
                winning_lines += 25  # Gives a positive value for each potential winning line for 'O'.
            elif (sum == 2) & (2 not in line):  # Check for two 'X's and no 'O's.
                winning_lines_opponent -= 25  # Gives a negative value for each potential winning line for 'X'.

    result = winning_lines + winning_lines_opponent  # Calculate the macrogame heuristic value.

    return result


def macrogame_heuristic(nuevo_juego, symbol):
    # Defines piece values based on the player.
    if symbol == 'X':
        my_value = 1
        opponent_value = 2
    else:
        my_value = 2
        opponent_value = 1

    # If the overall game result is still undecided, calculate score based on board positions.
    if nuevo_juego.state[92] == 0:  # Game is ongoing.
        macrogame_state = nuevo_juego.state[81:90]  # Get the state of the macrogame (results of subgames).

        # Define positions for corners, middles, and the center.
        corners = [0, 2, 6, 8]
        middles = [1, 3, 5, 7]
        center = 4

        # Initialize total score.
        score = 0

        # Check corners.
        for i in corners:
            if macrogame_state[i] == my_value:
                score += 30  # Favorable position for the player.
            elif macrogame_state[i] == opponent_value:
                score -= 30  # Unfavorable position for the player.

        # Check the center.
        if macrogame_state[center] == my_value:
            score += 50  # The center is highly valued.
        elif macrogame_state[center] == opponent_value:
            score -= 50

        # Check middles.
        for i in middles:
            if macrogame_state[i] == my_value:
                score += 10  # Middles have lower value compared to corners and the center.
            elif macrogame_state[i] == opponent_value:
                score -= 10

    # If the game is already won by the player, assign an infinitely high score.
    elif nuevo_juego.state[92] == my_value:
        score = math.inf
    
    # If the game is lost, assign an infinitely low score.
    else:
        score = -math.inf

    return score

def game_heuristic(nuevo_juego, symbol):
    ''' Combines various heuristic evaluations for the current game state.
        Returns the sum of the values obtained by each heuristic function. 
        The macrogame_heuristic value is halved to prioritize the value obtained in the other functions'''
    return subgame_heuristic(nuevo_juego, symbol) + macrogame_heuristic(nuevo_juego, symbol)/2 + macrogame_winning_lines(nuevo_juego, symbol)

def reverse_mapping(index):
    ''' Converts a game state index back to its matrix index form.
        This function is useful for translating internal game indices (used in the minimax algorithm).'''
    
    if index in REVERSE_MAPPING:
        # If the index exists in the reverse mapping dictionary, retrieve the (row, col) tuple.
        row, col = REVERSE_MAPPING[index]
        # Convert the (row, col) coordinates back to a single matrix index.
        # This is typically in the form 'rowcol' (e.g., row 3, column 4 becomes 34).
        return row * 10 + col
    else:
        # If the index does not exist in the reverse mapping (meaning it's not valid),
        # return None to indicate that no valid matrix position corresponds to this index.
        return None


def _minimax(juego, profundidad_total, profundidad_actual, turno, ficha):
    ''' Minimax algorithm implementation for the game. It evaluates possible moves and chooses the best one.
        'juego' is the current game state, 'profundidad_total' is the total depth, 'profundidad_actual' is the current depth,
        'turno' indicates whose turn it is (True for AI, False for opponent), and 'ficha' is the piece type ('X' or 'O').'''

    global movimiento_optimo  # Use a global variable to store the best move.

    # Base case: if the current depth is zero or the game is over, return the heuristic value of the board.
    if profundidad_actual == 0 or juego.is_game_over():
        return game_heuristic(juego, ficha)
        
    # Get all possible legal moves for the current game state.
    frontera = juego._get_legal_indexes().copy()

    if turno:  # AI's turn.
        best_value = -math.inf  # Initialize the best value as negative infinity.
        for elemento in frontera:
            # Create a new game state by copying the current one and making a move.
            nuevo_juego = UltimateTicTacToe(state=juego.state.copy())
            nuevo_juego.play(reverse_mapping(elemento))  # Make the move in the copied game state.
            # Recursively call the minimax function for the opponent's turn.
            valor_heuristico = _minimax(nuevo_juego, profundidad_total, profundidad_actual - 1, False, ficha)
            # Update the best value and optimal move if the current heuristic value is better.
            if best_value <= valor_heuristico:
                if profundidad_actual == profundidad_total:  # Update the optimal move only at the top level of recursion.
                    movimiento_optimo = elemento
                best_value = valor_heuristico
        return best_value
    else:  # Opponent's turn.
        best_value = math.inf  # Initialize the best value as infinity.
        for elemento in frontera:
            # Similar process as above, but for the opponent's move.
            nuevo_juego = UltimateTicTacToe(state=juego.state.copy())
            nuevo_juego.play(reverse_mapping(elemento))
            # Recursively call the minimax function for the AI's turn.
            valor_heuristico = _minimax(nuevo_juego, profundidad_total, profundidad_actual - 1, True, ficha)
            # Update the best value if the current heuristic value is better.
            if best_value > valor_heuristico:
                best_value = valor_heuristico
        return best_value

def minimax(uttt, profundidad, turno, ficha):
    ''' Public function to start the minimax algorithm.
        'uttt' is the Ultimate Tic Tac Toe game instance, 'profundidad' is the depth for the algorithm,
        'turno' indicates whose turn it is, and 'ficha' represents the piece type.'''
    global movimiento_optimo
    movimiento_optimo = None  # Reset the optimal move before starting.
    _minimax(uttt, profundidad, profundidad, turno, ficha)  # Start the recursive minimax process.
    return movimiento_optimo  # Return the optimal move after the minimax process is complete.


class Executable:
    @staticmethod
    def main():
        # Initialize the Ultimate Tic Tac Toe game.
        juego = UltimateTicTacToe()

        # Ask the user who should start the game.
        s = input("¿Quién inicia? Si quieres que inicie la inteligencia artificial, escribe IA. De lo contrario, escribe YO: ")
        if s == 'IA': # IA starts.
            symbol = "X"  # IA plays as X.
            # Make the AI perform the first move, chosen here as the middle of the board.
            juego.play(55)
        else: # Human starts.
            symbol = "O"  # Human plays as O.

        # Print the initial state of the game.
        print(juego)

        game_over = False
        while not game_over:  # Continue playing until the game is over.
            # Human player's turn.
            flag = True
            while flag:
                print("Es tu turno")  # It's your turn.
                jugada = input("Ingresa la casilla que quieras jugar: ")  # Enter the square you want to play.
                flag = False
                try:
                    juego.play(int(jugada))  # Attempt to make the entered move.
                    print(juego)  # Print the game state after the move.
                    game_over = juego.is_game_over()  # Check if the game is over.
                except utttError as e:
                    print('Jugada no válida')  # Invalid move.
                    flag = True  # Allow the player to make another move.
                if game_over:
                    break  # Exit the loop if the game is over.

            # AI's turn.
            if not game_over:
                print("Turno de la IA")  # It's the AI's turn.
                jugada_ia = reverse_mapping(minimax(juego, 5, True, symbol))  # AI calculates its move.
                if jugada_ia is not None:
                    juego.play(jugada_ia)  # Make the AI's move.
                    print("Movimiento de la IA:", jugada_ia)  # Print the AI's move.
                    print(juego)  # Print the game state after the AI's move.
                    game_over = juego.is_game_over()  # Check if the game is over.
                else:
                    print("La IA no pudo encontrar una jugada válida.")  # The AI could not find a valid move.

        # Determine the winner.
        if juego.get_winner() == X_STATE_VALUE:
            ganador = "X"
        elif juego.get_winner() == O_STATE_VALUE:
            ganador = "O"
        else:
            ganador = "Nadie (empate)"  # Nobody (tie).
        print("El juego ha terminado. Ganador: ", ganador)  # The game is over.


class EjecutableDosJugadores:
    def __init__(self) -> None:
        # Initializes the class; currently, it only sets the current player to None.
        self.jugador_actual = None
  
    def main():
        # Start a new game of Ultimate Tic Tac Toe.
        juego = UltimateTicTacToe()
        # Print the initial state of the game.
        print(juego)
        
        game_over = False  # Flag to check if the game has ended.
        while not game_over:  # Continue the game until it's over.
            flag = True
            while flag:  # Loop for player X's turn.
                jugada = input("Ingresa la casilla en la que juega X: ")  # Ask player X for a move.
                flag = False  # Assume the move is valid initially.
                try:  
                    juego.play(int(jugada))  # Try to make the move.
                    game_over = juego.is_game_over()  # Check if the game is over after the move.
                    print(juego)  # Print the game state.
                except utttError as e:
                    print('jugada no valida')  # Print an error message if the move is invalid.
                    flag = True  # Allow the player to try a different move.
                if game_over:  # If the game is over, exit the loop.
                    break

            if game_over:  # Check again if the game has ended to avoid switching to player O unnecessarily.
                continue  # Skip player O's turn if the game has already ended.

            flag = True  # Reset flag for player O's turn.
            while flag:  # Loop for player O's turn.
                jugada = input("Ingresa la casilla en la que juega O: ")  # Ask player O for a move.
                flag = False  # Assume the move is valid initially.
                try:  
                    juego.play(int(jugada))  # Try to make the move.
                    game_over = juego.is_game_over()  # Check if the game is over after the move.
                    print(juego)  # Print the game state.
                except utttError as e:
                    print('jugada no valida')  # Print an error message if the move is invalid.
                    flag = True  # Allow the player to try a different move.

        # Determine the winner.
        if juego.get_winner() == X_STATE_VALUE:
            ganador = "X"  # X is the winner.
        elif juego.get_winner() == O_STATE_VALUE:
            ganador = "O"  # O is the winner.
        else:
            ganador = "Nadie (empate)"  # No one wins (tie).
        print("El juego ha terminado. Ganador: ", ganador)  # Print the game result.


# Run game
Executable.main()