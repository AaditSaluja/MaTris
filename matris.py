#!/usr/bin/env python
import pygame
from pygame import Rect, Surface
import random
import numpy as np
import os

from tetrominoes import list_of_tetrominoes
from tetrominoes import rotate

from scores import load_score, write_score

class GameOver(Exception):
    """Exception used for its control flow properties"""

def get_sound(filename):
    return pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "resources", filename))

BGCOLOR = (15, 15, 20)
BORDERCOLOR = (140, 140, 140)

BLOCKSIZE = 30
BORDERWIDTH = 10

MATRIS_OFFSET = 20

MATRIX_WIDTH = 10
MATRIX_HEIGHT = 22

LEFT_MARGIN = 340

WIDTH = MATRIX_WIDTH*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2 + LEFT_MARGIN
HEIGHT = (MATRIX_HEIGHT-2)*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2

TRICKY_CENTERX = WIDTH-(WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2))/2

VISIBLE_MATRIX_HEIGHT = MATRIX_HEIGHT - 2

class Matris(object):
    def __init__(self, screen=None, render=True):
        self.screen = screen
        self.render = render
        if self.render and self.screen:
            self.surface = self.screen.subsurface(Rect((MATRIS_OFFSET+BORDERWIDTH, MATRIS_OFFSET+BORDERWIDTH),
                                                  (MATRIX_WIDTH * BLOCKSIZE, (MATRIX_HEIGHT-2) * BLOCKSIZE)))
        else:
            self.surface = None

        self.matrix = dict()
        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                self.matrix[(y,x)] = None

        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.hold_tetromino = None
        self.hold_used = False
        self.set_tetrominoes()
        self.tetromino_rotation = 0
        self.downwards_timer = 0
        self.base_downwards_speed = 0.4  # Move down every 400 ms
        

        self.movement_keys = {'left': 0, 'right': 0}
        self.movement_keys_speed = 0.05
        self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.level = 1
        self.score = 0
        self.lines = 0

        self.combo = 1  # Combo will increase when you clear lines with several tetrominos in a row

        self.paused = False

        self.highscore = load_score()
        self.played_highscorebeaten_sound = False

        self.levelup_sound  = get_sound("levelup.wav")
        self.gameover_sound = get_sound("gameover.wav")
        self.linescleared_sound = get_sound("linecleared.wav")
        self.highscorebeaten_sound = get_sound("highscorebeaten.wav")

        self.reward = 0  # Initialize reward to display alongside score
        self.num_blocks_played = 0
        self.total_rewards = 0
        self.num_updates = 0
        self.last_average_reward = 0
        self.average_reward = 0
        self.reward_rate_of_increase = 0
        self.lines_cleared_last = 0

    def set_tetrominoes(self):
        """
        Sets information for the current and next tetrominos
        """
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.surface_of_hold_tetromino = self.construct_surface_of_hold_tetromino()
        self.tetromino_position = (0, 4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)
        self.hold_used = False

    def hold_piece(self):
        """
        Allows the player to hold the current tetromino
        """
        if self.hold_used:
            return
        self.hold_used = True
        if self.hold_tetromino is None:
            self.hold_tetromino = self.current_tetromino
            self.current_tetromino = self.next_tetromino
            self.next_tetromino = random.choice(list_of_tetrominoes)
        else:
            self.current_tetromino, self.hold_tetromino = self.hold_tetromino, self.current_tetromino

        self.tetromino_position = (0, 4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)

        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.surface_of_hold_tetromino = self.construct_surface_of_hold_tetromino()

    def hard_drop(self):
        """
        Instantly places tetrominos in the cells below
        """
        while self.request_movement('down'):
            continue  # Simply move the piece down without incrementing the score

        self.lock_tetromino()

    def update(self, timepassed):
        """
        Main game loop
        """
        self.needs_redraw = False

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.gameover(full_exit=True)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    if self.surface:
                        self.surface.fill((0,0,0))
                    self.needs_redraw = True
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.gameover()
                elif event.key == pygame.K_SPACE:
                    self.hard_drop()
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    self.request_rotation()
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    self.request_movement('left')
                    self.movement_keys['left'] = 1
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    self.request_movement('right')
                    self.movement_keys['right'] = 1
                elif event.key == pygame.K_c:
                    self.hold_piece()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    self.movement_keys['left'] = 0
                    self.movement_keys_timer = (-self.movement_keys_speed)*2
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    self.movement_keys['right'] = 0
                    self.movement_keys_timer = (-self.movement_keys_speed)*2

        if self.paused:
            return self.needs_redraw

        self.downwards_speed = self.base_downwards_speed ** (1 + self.level / 10)

        self.downwards_timer += timepassed
        keypressed = pygame.key.get_pressed()
        downwards_speed = self.downwards_speed * 0.10 if any([keypressed[pygame.K_DOWN], keypressed[pygame.K_s]]) else self.downwards_speed
        if self.downwards_timer > downwards_speed:
            if not self.request_movement('down'):
                self.lock_tetromino()
            self.downwards_timer %= downwards_speed

        if any(self.movement_keys.values()):
            self.movement_keys_timer += timepassed
        if self.movement_keys_timer > self.movement_keys_speed:
            self.request_movement('right' if self.movement_keys['right'] else 'left')
            self.movement_keys_timer %= self.movement_keys_speed

        return self.needs_redraw

    def draw_surface(self):
        """
        Draws the image of the current tetromino
        """
        if not self.render or not self.surface:
            return  # Skip rendering if not enabled

        with_tetromino = self.blend(matrix=self.place_shadow())

        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                block_location = Rect(x * BLOCKSIZE, (y * BLOCKSIZE - 2 * BLOCKSIZE), BLOCKSIZE, BLOCKSIZE)
                if with_tetromino[(y, x)] is None:
                    self.surface.fill(BGCOLOR, block_location)
                else:
                    if with_tetromino[(y, x)][0] == 'shadow':
                        self.surface.fill(BGCOLOR, block_location)
                    self.surface.blit(with_tetromino[(y, x)][1], block_location)

    def gameover(self, full_exit=False):
        """
        Gameover occurs when a new tetromino does not fit after the old one has died, either
        after a "natural" drop or a hard drop by the player. That is why `self.lock_tetromino`
        is responsible for checking if it's game over.
        """
        write_score(self.score)
        if full_exit:
            exit()
        else:
            raise GameOver("Game Over")

    def place_shadow(self):
        """
        Draws shadow of tetromino so player can see where it will be placed
        """
        posY, posX = self.tetromino_position
        while self.blend(position=(posY, posX)):
            posY += 1

        position = (posY - 1, posX)
        return self.blend(position=position, shadow=True)

    def fits_in_matrix(self, shape, position):
        """
        Checks if tetromino fits on the board
        """
        posY, posX = position
        for x in range(posX, posX + len(shape)):
            for y in range(posY, posY + len(shape)):
                if self.matrix.get((y, x), False) is False and shape[y - posY][x - posX]:
                    return False
        return position

    def request_rotation(self):
        """
        Checks if tetromino can rotate
        Returns the tetromino's rotation position if possible
        """
        rotation = (self.tetromino_rotation + 1) % 4
        shape = self.rotated(rotation)

        y, x = self.tetromino_position

        position = (self.fits_in_matrix(shape, (y, x)) or
                    self.fits_in_matrix(shape, (y, x + 1)) or
                    self.fits_in_matrix(shape, (y, x - 1)) or
                    self.fits_in_matrix(shape, (y, x + 2)) or
                    self.fits_in_matrix(shape, (y, x - 2)))
        # ^ That's how wall-kick is implemented

        if position and self.blend(shape, position):
            self.tetromino_rotation = rotation
            self.tetromino_position = position

            self.needs_redraw = True
            return self.tetromino_rotation
        else:
            return False

    def request_movement(self, direction):
        """
        Checks if tetromino can move in the given direction and returns its new position if movement is possible
        """
        posY, posX = self.tetromino_position
        if direction == 'left' and self.blend(position=(posY, posX - 1)):
            self.tetromino_position = (posY, posX - 1)
            self.needs_redraw = True
            return self.tetromino_position
        elif direction == 'right' and self.blend(position=(posY, posX + 1)):
            self.tetromino_position = (posY, posX + 1)
            self.needs_redraw = True
            return self.tetromino_position
        elif direction == 'up' and self.blend(position=(posY - 1, posX)):
            self.needs_redraw = True
            self.tetromino_position = (posY - 1, posX)
            return self.tetromino_position
        elif direction == 'down' and self.blend(position=(posY + 1, posX)):
            self.tetromino_position = (posY + 1, posX)
            self.needs_redraw = True
            return self.tetromino_position
        else:
            return False

    def rotated(self, rotation=None):
        """
        Rotates tetromino
        """
        if rotation is None:
            rotation = self.tetromino_rotation
        return rotate(self.current_tetromino.shape, rotation)

    def block(self, color, shadow=False):
        """
        Sets visual information for tetromino
        """
        colors = {'blue': (105, 105, 255),
                  'yellow': (225, 242, 41),
                  'pink': (242, 41, 195),
                  'green': (22, 181, 64),
                  'red': (204, 22, 22),
                  'orange': (245, 144, 12),
                  'cyan': (10, 255, 226)}

        if shadow:
            end = [90]  # end is the alpha value
        else:
            end = []  # Adding this to the end will not change the array, thus no alpha value

        border = Surface((BLOCKSIZE, BLOCKSIZE), pygame.SRCALPHA, 32)
        border.fill(list(map(lambda c: c * 0.5, colors[color])) + end)

        borderwidth = 2

        box = Surface((BLOCKSIZE - borderwidth * 2, BLOCKSIZE - borderwidth * 2), pygame.SRCALPHA, 32)
        boxarr = pygame.PixelArray(box)
        for x in range(len(boxarr)):
            for y in range(len(boxarr)):
                boxarr[x][y] = tuple(list(map(lambda c: min(255, int(c * random.uniform(0.8, 1.2))), colors[color])) + end)

        del boxarr  # deleting boxarr or else the box surface will be 'locked' or something like that and won't blit.
        border.blit(box, Rect(borderwidth, borderwidth, 0, 0))

        return border

    def lock_tetromino(self):
        """
        This method is called whenever the falling tetromino "dies". `self.matrix` is updated,
        the lines are counted and cleared, and a new tetromino is chosen.
        Scoring is implemented as in the second code version.
        """
        # Increment the number of blocks played each time a tetromino is locked
        self.num_blocks_played += 1

        # Store the old state before locking the tetromino
        old_state = {
            'lines': self.lines,
            'score': self.score,
            'level': self.level,
            'holes': self.count_holes()  # Capture the number of holes before the new block is placed
        }

        # Lock the tetromino and update the matrix
        blended_matrix = self.blend()
        if blended_matrix:
            self.matrix = blended_matrix
        else:
            self.gameover_sound.play()
            self.gameover()

        # Remove lines and update lines count
        lines_cleared = self.remove_lines()
        self.lines += lines_cleared

        # Implement scoring as before
        if lines_cleared:
            if lines_cleared >= 4:
                self.linescleared_sound.play()
            self.score += 100 * (lines_cleared ** 2) * self.combo

            # Check and play high score related sounds
            if not self.played_highscorebeaten_sound and self.score > self.highscore:
                if self.highscore != 0:
                    self.highscorebeaten_sound.play()
                self.played_highscorebeaten_sound = True

        # Level up logic
        if self.lines >= self.level * 10:
            self.levelup_sound.play()
            self.level += 1

        # Combo logic remains the same
        self.combo = self.combo + 1 if lines_cleared else 1

        # Compute the reward using compute_reward (but don't use it for scoring)
        new_state = {
            'lines': self.lines,
            'score': self.score,
            'level': self.level,
            'lines_cleared': lines_cleared
        }
        self.reward = self.compute_reward(old_state, new_state)

        # Update cumulative rewards for average calculation
        self.total_rewards += self.reward
        self.num_updates += 1
        self.update_average_reward()

        # Set up next tetromino
        self.set_tetrominoes()

        # Game over check after setting new tetromino
        if not self.blend():
            self.gameover_sound.play()
            self.gameover()

        self.needs_redraw = True

    def update_average_reward(self):
        # Update the average reward and its rate of change
        current_average_reward = self.total_rewards / self.num_updates if self.num_updates > 0 else 0
        self.reward_rate_of_increase = current_average_reward - self.last_average_reward
        self.last_average_reward = current_average_reward
        self.average_reward = current_average_reward

    def remove_lines(self):
        lines = []
        for y in range(MATRIX_HEIGHT):
            if all(self.matrix[(y, x)] is not None for x in range(MATRIX_WIDTH)):
                lines.append(y)
        
        # Number of lines cleared in the last move
        self.lines_cleared_last = len(lines)

        for line in sorted(lines, reverse=True):
            # Move all rows above 'line' one row down
            for y in range(line, 0, -1):
                for x in range(MATRIX_WIDTH):
                    self.matrix[(y, x)] = self.matrix[(y - 1, x)]
            # Clear the topmost row
            for x in range(MATRIX_WIDTH):
                self.matrix[(0, x)] = None
        
        return len(lines)


    def blend(self, shape=None, position=None, matrix=None, shadow=False):
        """
        Does `shape` at `position` fit in `matrix`? If so, return a new copy of `matrix` where all
        the squares of `shape` have been placed in `matrix`. Otherwise, return False.

        This method is often used simply as a test, for example to see if an action by the player is valid.
        It is also used in `self.draw_surface` to paint the falling tetromino and its shadow on the screen.
        """
        if shape is None:
            shape = self.rotated()
        if position is None:
            position = self.tetromino_position

        copy = dict(self.matrix if matrix is None else matrix)
        posY, posX = position
        for x in range(posX, posX + len(shape)):
            for y in range(posY, posY + len(shape)):
                if (copy.get((y, x), False) is False and shape[y - posY][x - posX]  # shape is outside the matrix
                    or  # coordinate is occupied by something else which isn't a shadow
                    copy.get((y, x)) and shape[y - posY][x - posX] and copy[(y, x)][0] != 'shadow'):

                    return False  # Blend failed; `shape` at `position` breaks the matrix

                elif shape[y - posY][x - posX]:
                    copy[(y, x)] = ('shadow', self.shadow_block) if shadow else ('block', self.tetromino_block)

        return copy

    def construct_surface_of_next_tetromino(self):
        """
        Draws the image of the next tetromino
        """
        if not self.render:
            return None

        shape = self.next_tetromino.shape
        surf = Surface((len(shape) * BLOCKSIZE, len(shape) * BLOCKSIZE), pygame.SRCALPHA, 32)

        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.next_tetromino.color), (x * BLOCKSIZE, y * BLOCKSIZE))
        return surf

    def construct_surface_of_hold_tetromino(self):
        """
        Draws the image of the hold tetromino
        """
        if not self.render:
            return None

        if self.hold_tetromino is None:
            return Surface((0, 0), pygame.SRCALPHA, 32)
        shape = self.hold_tetromino.shape
        surf = Surface((len(shape) * BLOCKSIZE, len(shape) * BLOCKSIZE), pygame.SRCALPHA, 32)
        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.hold_tetromino.color), (x * BLOCKSIZE, y * BLOCKSIZE))
        return surf

    def place_shadow(self):
        """
        Draws shadow of tetromino so player can see where it will be placed
        """
        posY, posX = self.tetromino_position
        while self.blend(position=(posY, posX)):
            posY += 1

        position = (posY - 1, posX)
        return self.blend(position=position, shadow=True)

    def compute_reward(self, old_state, new_state):
        lines_cleared = new_state['lines_cleared']
        holes_after = self.count_holes()
        heights = self.get_column_heights()
        max_height = max(heights)  # Maximum stack height

        # Define reward constants
        line_clear_reward = 100   # Basic reward for clearing each line
        tetris_bonus = 800        # Large bonus for clearing four lines at once
        hole_penalty = -50        # Penalty for each hole created
        height_penalty = -10      # Penalty for increased stack height, to encourage low plays
        score_fraction = 100     # Fraction of the game score to include in the reward

        # Calculate dynamic rewards
        reward = (line_clear_reward * lines_cleared +
                tetris_bonus * (1 if lines_cleared == 4 else 0) +  # Apply Tetris bonus only for 4 lines
                hole_penalty * holes_after +
                height_penalty * max_height +
                score_fraction * self.score)  # Include a fraction of the score in the reward

        # Normalize the reward by the number of blocks played to avoid rewarding just longer play
        reward /= (self.num_blocks_played if self.num_blocks_played > 0 else 1)

        return reward


    def compute_contiguity_reward(self):
        """
        Compute a reward based on contiguous filled cells in each row, weighted more heavily for lower rows.
        """
        reward = 0
        for y in range(MATRIX_HEIGHT):
            max_contiguity = 0
            current_contiguity = 0
            for x in range(MATRIX_WIDTH):
                if self.matrix[(y, x)] is not None:
                    current_contiguity += 1
                    max_contiguity = max(max_contiguity, current_contiguity)
                else:
                    current_contiguity = 0
            # Apply higher weights for lower rows
            row_weight = 1 + (MATRIX_HEIGHT - y) / MATRIX_HEIGHT
            reward += (max_contiguity ** 2) * row_weight
        return reward

    def count_holes(self):
        """
        Count the number of 'holes' in the stack. A hole is defined as an empty space below at least one block.
        """
        holes = 0
        for x in range(MATRIX_WIDTH):
            block_found = False
            for y in range(MATRIX_HEIGHT):
                if self.matrix[(y, x)] is not None:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def get_column_heights(self):
        """
        Get the height of each column in the play field. The height is defined as the row index of the lowest filled cell.
        """
        heights = [0] * MATRIX_WIDTH
        for x in range(MATRIX_WIDTH):
            for y in range(MATRIX_HEIGHT):
                if self.matrix[(y, x)] is not None:
                    heights[x] = MATRIX_HEIGHT - y
                    break
        return heights

    def count_fills_in_column(self, x):
        """
        Count the number of filled cells in a specific column.
        """
        count = 0
        for y in range(MATRIX_HEIGHT):
            if self.matrix[(y, x)] is not None:
                count += 1
        return count

    def compute_height_differences(self):
        """
        Compute the total difference in height between adjacent columns to measure the 'bumpiness' of the stack.
        """
        heights = self.get_column_heights()
        height_diff = 0
        for i in range(MATRIX_WIDTH - 1):
            diff = heights[i] - heights[i + 1]
            if 0 <= diff <= 2:
                height_diff += 1
            else:
                height_diff -= 1
        return height_diff

    def get_state_features(self):
        """
        Extracts features from the current state for the RL agent. Features include column heights, aggregate height, number of holes, bumpiness, and lines cleared.
        """
        heights = [0] * MATRIX_WIDTH
        num_holes = 0
        for x in range(MATRIX_WIDTH):
            column_filled = False
            for y in range(MATRIX_HEIGHT):
                cell = self.matrix[(y, x)]
                if cell is not None and cell[0] == 'block':
                    if not column_filled:
                        heights[x] = MATRIX_HEIGHT - y
                        column_filled = True
                elif column_filled:
                    num_holes += 1

        aggregate_height = sum(heights)
        bumpiness = sum([abs(heights[i] - heights[i + 1]) for i in range(MATRIX_WIDTH - 1)])
        lines_cleared = self.lines_cleared_last

        features = np.array(heights + [aggregate_height, num_holes, bumpiness, lines_cleared])
        return features

    def reset(self):
        """
        Resets the game state for a new episode.
        """
        self.__init__(screen=self.screen, render=self.render)

class Game(object):
    def __init__(self, render=True):
        self.render = render
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("MaTris")
            self.screen.blit(construct_nightmare(self.screen.get_size()), (0,0))
            matris_border = Surface((MATRIX_WIDTH * BLOCKSIZE + BORDERWIDTH * 2, VISIBLE_MATRIX_HEIGHT * BLOCKSIZE + BORDERWIDTH * 2))
            matris_border.fill(BORDERCOLOR)
            self.screen.blit(matris_border, (MATRIS_OFFSET, MATRIS_OFFSET))
        else:
            self.screen = None
        self.matris = Matris(screen=self.screen, render=self.render)
        self.clock = pygame.time.Clock()
        self.running = True

    def main(self):
        """
        Main loop for game. Redraws scores and next tetromino each time the loop is passed through.
        """
        if self.render:
            self.redraw()

        while self.running:
            try:
                timepassed = self.clock.tick(50)
                if self.matris.update((timepassed / 1000.) if not self.matris.paused else 0):
                    if self.render:
                        self.redraw()
            except GameOver:
                self.running = False

        if self.render:
            pygame.quit()

    def reset(self):
        """
        Resets the game for a new episode.
        """
        self.matris.reset()
        self.running = True

    def step(self, action_key):
        """
        Processes one step in the game based on the action taken, typically triggered by a key press.
        """
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': action_key}))
        try:
            if self.matris.update(0.02):
                if self.render:
                    self.redraw()
                    pygame.display.flip()  # Ensure the display updates
            done = False
        except GameOver:
            done = True
        reward = self.matris.reward
        score = self.matris.score
        state = self.matris.get_state_features()
        return state, reward, done, score

    def redraw(self):
        """
        Redraws the information panel and next tetromino panel.
        """
        if not self.matris.paused and self.render:
            self.blit_tetromino_area(self.matris.surface_of_next_tetromino, "Next", {'top': MATRIS_OFFSET, 'centerx': TRICKY_CENTERX + 30})
            self.blit_tetromino_area(self.matris.surface_of_hold_tetromino, "Hold", {'top': MATRIS_OFFSET, 'centerx': TRICKY_CENTERX - 110})
            self.blit_info()

            self.matris.draw_surface()

            pygame.display.flip()

    def blit_info(self):
        """
        Draws information panel.
        """
        if not self.render:
            return

        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 30)
        width = (WIDTH - (MATRIS_OFFSET + BLOCKSIZE * MATRIX_WIDTH + BORDERWIDTH * 2)) - MATRIS_OFFSET * 2

        def renderpair(text, val):
            text = font.render(text, True, textcolor)
            val = font.render(str(val), True, textcolor)

            surf = Surface((width, text.get_rect().height + BORDERWIDTH * 2), pygame.SRCALPHA, 32)

            surf.blit(text, text.get_rect(top=BORDERWIDTH + 10, left=BORDERWIDTH + 10))
            surf.blit(val, val.get_rect(top=BORDERWIDTH + 10, right=width - (BORDERWIDTH + 10)))
            return surf

        # Resizes side panel to allow for all information to be displayed there.
        scoresurf = renderpair("Score", self.matris.score)
        rewardsurf = renderpair("Reward", round(self.matris.reward, 2))  # Display the reward
        levelsurf = renderpair("Level", self.matris.level)
        linessurf = renderpair("Lines", self.matris.lines)
        combosurf = renderpair("Combo", "x{}".format(self.matris.combo))

        height = 20 + (levelsurf.get_rect().height +
                       scoresurf.get_rect().height +
                       rewardsurf.get_rect().height +  # Include the reward surface height
                       linessurf.get_rect().height +
                       combosurf.get_rect().height)

        # Colors side panel
        area = Surface((width, height))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width - BORDERWIDTH * 2, height - BORDERWIDTH * 2))

        # Draws side panel
        area.blit(levelsurf, (0, 0))
        area.blit(scoresurf, (0, levelsurf.get_rect().height))
        area.blit(rewardsurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height))  # Position reward surface
        area.blit(linessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + rewardsurf.get_rect().height))
        area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + rewardsurf.get_rect().height + linessurf.get_rect().height))

        self.screen.blit(area, area.get_rect(bottom=HEIGHT - MATRIS_OFFSET, centerx=TRICKY_CENTERX))

    def blit_tetromino_area(self, tetromino_surf, label, position):
        """
        Draws the tetromino (Next or Hold) in a box to the side of the board.
        """
        if not self.render or tetromino_surf is None:
            return

        area = Surface((BLOCKSIZE * 5, BLOCKSIZE * 5))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE * 5 - BORDERWIDTH * 2, BLOCKSIZE * 5 - BORDERWIDTH * 2))

        font = pygame.font.Font(None, 30)
        text = font.render(label, True, (255, 255, 255))
        area.blit(text, (BORDERWIDTH, BORDERWIDTH))

        areasize = area.get_size()[0]
        tetromino_surf_size = tetromino_surf.get_size()[0]

        center = areasize / 2 - tetromino_surf_size / 2 + BORDERWIDTH
        area.blit(tetromino_surf, (center, center + 20))
        self.screen.blit(area, area.get_rect(**position))

class Menu(object):
    """
    Creates main menu.
    """
    running = True

    def main(self):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("MaTris")
        clock = pygame.time.Clock()
        menu = kezmenu.KezMenu(
            ['Play!', lambda: Game(render=True).main()],
            ['Quit', lambda: setattr(self, 'running', False)],
        )
        menu.position = (50, 50)
        menu.enableEffect('enlarge-font-on-focus', font=None, size=60, enlarge_factor=1.2, enlarge_time=0.3)
        menu.color = (255, 255, 255)
        menu.focus_color = (40, 200, 40)

        nightmare = construct_nightmare(screen.get_size())
        highscoresurf = self.construct_highscoresurf()  # Loads highscore onto menu

        timepassed = clock.tick(30) / 1000.

        while self.running:
            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    exit()

            menu.update(events, timepassed)

            timepassed = clock.tick(30) / 1000.

            if timepassed > 1:  # A game has most likely been played
                highscoresurf = self.construct_highscoresurf()

            screen.blit(nightmare, (0, 0))
            screen.blit(highscoresurf, highscoresurf.get_rect(right=WIDTH - 50, bottom=HEIGHT - 50))
            menu.draw(screen)
            pygame.display.flip()

    def construct_highscoresurf(self):
        """
        Loads high score from file.
        """
        font = pygame.font.Font(None, 50)
        highscore = load_score()
        text = "Highscore: {}".format(highscore)
        return font.render(text, True, (255, 255, 255))

def construct_nightmare(size):
    """
    Constructs background image.
    """
    surf = Surface(size)

    boxsize = 8
    bordersize = 1
    vals = '1235'  # only the lower values, for darker colors and greater fear
    arr = pygame.PixelArray(surf)
    for x in range(0, len(arr), boxsize):
        for y in range(0, len(arr[x]), boxsize):
            color = int(''.join([random.choice(vals) + random.choice(vals) for _ in range(3)]), 16)

            for LX in range(x, x + (boxsize - bordersize)):
                for LY in range(y, y + (boxsize - bordersize)):
                    if LX < len(arr) and LY < len(arr[x]):
                        arr[LX][LY] = color
    del arr
    return surf

if __name__ == '__main__':
    Menu().main()
