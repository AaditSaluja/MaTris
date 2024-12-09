#!/usr/bin/env python
import pygame
from pygame import Rect, Surface
import random
import os

from tetrominoes import list_of_tetrominoes, rotate
from scores import load_score, write_score

class GameOver(Exception):
    """Raised when the game ends."""

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
TRICKY_CENTERX = WIDTH - (WIDTH - (MATRIS_OFFSET + BLOCKSIZE*MATRIX_WIDTH + BORDERWIDTH*2)) / 2
VISIBLE_MATRIX_HEIGHT = MATRIX_HEIGHT - 2

def get_sound(filename):
    return pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "resources", filename))

class Matris(object):
    def __init__(self):
        self.surface = None
        self.matrix = {(y, x): None for y in range(MATRIX_HEIGHT) for x in range(MATRIX_WIDTH)}
        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.hold_tetromino = None
        self.hold_used = False
        self.tetromino_rotation = 0
        self.downwards_timer = 0
        self.base_downwards_speed = 0.4
        self.movement_keys = {'left': 0, 'right': 0}
        self.movement_keys_speed = 0.05
        self.movement_keys_timer = (-self.movement_keys_speed)*2
        self.level = 1
        self.score = 0
        self.lines = 0
        self.combo = 1
        self.paused = False
        self.highscore = load_score()
        self.played_highscorebeaten_sound = False

        self.levelup_sound  = get_sound("levelup.wav")
        self.gameover_sound = get_sound("gameover.wav")
        self.linescleared_sound = get_sound("linecleared.wav")
        self.highscorebeaten_sound = get_sound("highscorebeaten.wav")

        self.set_tetrominoes()

    def set_tetrominoes(self):
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)
        self.hold_used = False
        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.surface_of_hold_tetromino = self.construct_surface_of_hold_tetromino()

    def hold_piece(self):
        if self.hold_used:
            return
        self.hold_used = True
        if self.hold_tetromino is None:
            self.hold_tetromino = self.current_tetromino
            self.current_tetromino = self.next_tetromino
            self.next_tetromino = random.choice(list_of_tetrominoes)
        else:
            self.current_tetromino, self.hold_tetromino = self.hold_tetromino, self.current_tetromino

        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0,3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)

        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.surface_of_hold_tetromino = self.construct_surface_of_hold_tetromino()

    def hard_drop(self):
        amount = 0
        while self.request_movement('down'):
            amount += 1
        self.score += 10*amount
        self.lock_tetromino()

    def update(self, timepassed):
        self.needs_redraw = False
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.QUIT:
                raise GameOver
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.needs_redraw = True
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    raise GameOver
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

        self.downwards_speed = self.base_downwards_speed ** (1 + self.level/10.)
        keypressed = pygame.key.get_pressed()
        downwards_speed = self.downwards_speed*0.10 if (keypressed[pygame.K_DOWN] or keypressed[pygame.K_s]) else self.downwards_speed

        self.downwards_timer += timepassed
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

    def request_rotation(self):
        rotation = (self.tetromino_rotation + 1) % 4
        shape = self.rotated(rotation)
        y, x = self.tetromino_position
        position = (self.fits_in_matrix(shape, (y, x)) or
                    self.fits_in_matrix(shape, (y, x+1)) or
                    self.fits_in_matrix(shape, (y, x-1)) or
                    self.fits_in_matrix(shape, (y, x+2)) or
                    self.fits_in_matrix(shape, (y, x-2)))
        if position and self.blend(shape, position):
            self.tetromino_rotation = rotation
            self.tetromino_position = position
            self.needs_redraw = True
            return True
        return False

    def request_movement(self, direction):
        posY, posX = self.tetromino_position
        if direction == 'left' and self.blend(position=(posY, posX-1)):
            self.tetromino_position = (posY, posX-1)
            self.needs_redraw = True
            return True
        elif direction == 'right' and self.blend(position=(posY, posX+1)):
            self.tetromino_position = (posY, posX+1)
            self.needs_redraw = True
            return True
        elif direction == 'down' and self.blend(position=(posY+1, posX)):
            self.tetromino_position = (posY+1, posX)
            self.needs_redraw = True
            return True
        return False

    def rotated(self, rotation=None):
        if rotation is None:
            rotation = self.tetromino_rotation
        return rotate(self.current_tetromino.shape, rotation)

    def block(self, color, shadow=False):
        colors = {'blue':   (105, 105, 255),
                  'yellow': (225, 242, 41),
                  'pink':   (242, 41, 195),
                  'green':  (22, 181, 64),
                  'red':    (204, 22, 22),
                  'orange': (245, 144, 12),
                  'cyan':   (10, 255, 226)}

        if shadow:
            end = [90]
        else:
            end = []

        border = Surface((BLOCKSIZE, BLOCKSIZE), pygame.SRCALPHA, 32)
        border.fill([c*0.5 for c in colors[color]] + end)
        borderwidth = 2
        box = Surface((BLOCKSIZE-borderwidth*2, BLOCKSIZE-borderwidth*2), pygame.SRCALPHA, 32)
        boxarr = pygame.PixelArray(box)
        for x in range(len(boxarr)):
            for y in range(len(boxarr)):
                col = [min(255, int(c*random.uniform(0.8, 1.2))) for c in colors[color]]
                boxarr[x][y] = tuple(col + end)
        del boxarr
        border.blit(box, Rect(borderwidth, borderwidth, 0, 0))
        return border

    def lock_tetromino(self):
        self.matrix = self.blend()
        lines_cleared = self.remove_lines()
        self.lines += lines_cleared
        if lines_cleared:
            if lines_cleared >= 4:
                self.linescleared_sound.play()
            self.score += 100 * (lines_cleared**2) * self.combo
            if not self.played_highscorebeaten_sound and self.score > self.highscore:
                if self.highscore != 0:
                    self.highscorebeaten_sound.play()
                self.played_highscorebeaten_sound = True
        if self.lines >= self.level*10:
            self.levelup_sound.play()
            self.level += 1
        self.combo = self.combo + 1 if lines_cleared else 1
        self.set_tetrominoes()
        if not self.blend():
            self.gameover_sound.play()
            self.gameover()
        self.needs_redraw = True

    def remove_lines(self):
        lines = []
        for y in range(MATRIX_HEIGHT):
            row = [x for x in range(MATRIX_WIDTH) if self.matrix[(y,x)]]
            if len(row) == MATRIX_WIDTH:
                lines.append(y)
        for line in sorted(lines):
            for x in range(MATRIX_WIDTH):
                self.matrix[(line,x)] = None
            for y in range(line, 0, -1):
                for x in range(MATRIX_WIDTH):
                    self.matrix[(y,x)] = self.matrix.get((y-1,x), None)
        return len(lines)

    def blend(self, shape=None, position=None, matrix=None, shadow=False):
        if shape is None:
            shape = self.rotated()
        if position is None:
            position = self.tetromino_position
        copy = dict(self.matrix if matrix is None else matrix)
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if ((copy.get((y,x), False) is False and shape[y-posY][x-posX]) or
                    (copy.get((y,x)) and shape[y-posY][x-posX] and copy[(y,x)][0] != 'shadow')):
                    return False
                elif shape[y-posY][x-posX]:
                    copy[(y,x)] = ('shadow', self.shadow_block) if shadow else ('block', self.tetromino_block)
        return copy

    def fits_in_matrix(self, shape, position):
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if self.matrix.get((y, x), False) is False and shape[y-posY][x-posX]:
                    return False
        return position

    def gameover(self):
        write_score(self.score)
        raise GameOver("Game Over")

    def construct_surface_of_next_tetromino(self):
        shape = self.next_tetromino.shape
        surf = Surface((len(shape)*BLOCKSIZE, len(shape)*BLOCKSIZE), pygame.SRCALPHA, 32)
        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.next_tetromino.color), (x*BLOCKSIZE, y*BLOCKSIZE))
        return surf

    def construct_surface_of_hold_tetromino(self):
        if self.hold_tetromino is None:
            return Surface((0, 0), pygame.SRCALPHA, 32)
        shape = self.hold_tetromino.shape
        surf = Surface((len(shape)*BLOCKSIZE, len(shape)*BLOCKSIZE), pygame.SRCALPHA, 32)
        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.hold_tetromino.color), (x*BLOCKSIZE, y*BLOCKSIZE))
        return surf

    def place_shadow(self):
        posY, posX = self.tetromino_position
        while self.blend(position=(posY, posX)):
            posY += 1
        position = (posY-1, posX)
        return self.blend(position=position, shadow=True)

    def draw_surface(self):
        with_tetromino = self.blend(matrix=self.place_shadow())
        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                block_location = Rect(x*BLOCKSIZE, (y*BLOCKSIZE - 2*BLOCKSIZE), BLOCKSIZE, BLOCKSIZE)
                if with_tetromino[(y,x)] is None:
                    self.surface.fill(BGCOLOR, block_location)
                else:
                    if with_tetromino[(y,x)][0] == 'shadow':
                        self.surface.fill(BGCOLOR, block_location)
                    self.surface.blit(with_tetromino[(y,x)][1], block_location)


class Game:
    def __init__(self, render=True):
        self.render = render
        self.matris = Matris()

        if self.render:
            pygame.display.set_caption("RL MaTris")
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.screen.blit(self.construct_nightmare(self.screen.get_size()), (0,0))

            matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
            matris_border.fill(BORDERCOLOR)
            self.screen.blit(matris_border, (MATRIS_OFFSET,MATRIS_OFFSET))

            self.matris.surface = self.screen.subsurface(Rect((MATRIS_OFFSET+BORDERWIDTH, MATRIS_OFFSET+BORDERWIDTH),
                                               (MATRIX_WIDTH * BLOCKSIZE, (MATRIX_HEIGHT-2) * BLOCKSIZE)))

            self.redraw()
        else:
            self.screen = None

    def step(self, action_key):
        # Optional delay to let the agent "think"
        pygame.time.wait(100)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': action_key}))
        try:
            if self.matris.update(0.02):
                if self.render:
                    self.redraw()
            done = False
        except GameOver:
            done = True
        return done, self.matris.score

    def redraw(self):
        if not self.render:
            return
        self.blit_tetromino_area(self.matris.surface_of_next_tetromino, "Next", {'top': MATRIS_OFFSET, 'centerx': TRICKY_CENTERX + 30})
        self.blit_tetromino_area(self.matris.surface_of_hold_tetromino, "Hold", {'top': MATRIS_OFFSET, 'centerx': TRICKY_CENTERX - 110})
        self.blit_info()
        self.matris.draw_surface()
        pygame.display.flip()

    def blit_info(self):
        if not self.render:
            return
        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 30)
        width = (WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2)) - MATRIS_OFFSET*2

        def renderpair(text, val):
            t = font.render(text, True, textcolor)
            v = font.render(str(val), True, textcolor)
            surf = Surface((width, t.get_rect().height + BORDERWIDTH*2), pygame.SRCALPHA, 32)
            surf.blit(t, t.get_rect(top=BORDERWIDTH+10, left=BORDERWIDTH+10))
            surf.blit(v, v.get_rect(top=BORDERWIDTH+10, right=width-(BORDERWIDTH+10)))
            return surf

        scoresurf = renderpair("Score", self.matris.score)
        levelsurf = renderpair("Level", self.matris.level)
        linessurf = renderpair("Lines", self.matris.lines)
        combosurf = renderpair("Combo", "x{}".format(self.matris.combo))

        height = 20 + levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height + combosurf.get_rect().height
        area = Surface((width, height))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width-BORDERWIDTH*2, height-BORDERWIDTH*2))

        area.blit(levelsurf, (0,0))
        area.blit(scoresurf, (0, levelsurf.get_rect().height))
        area.blit(linessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height))
        area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height))

        self.screen.blit(area, area.get_rect(bottom=HEIGHT-MATRIS_OFFSET, centerx=TRICKY_CENTERX))

    def blit_tetromino_area(self, tetromino_surf, label, position):
        if not self.render:
            return
        area = Surface((BLOCKSIZE*5, BLOCKSIZE*5))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE*5-BORDERWIDTH*2, BLOCKSIZE*5-BORDERWIDTH*2))

        font = pygame.font.Font(None, 30)
        text = font.render(label, True, (255, 255, 255))
        area.blit(text, (BORDERWIDTH, BORDERWIDTH))

        areasize = area.get_size()[0]
        tetromino_surf_size = tetromino_surf.get_size()[0]

        center = areasize/2 - tetromino_surf_size/2 + BORDERWIDTH
        area.blit(tetromino_surf, (center, center+20))
        self.screen.blit(area, area.get_rect(**position))

    def construct_nightmare(self, size):
        surf = Surface(size)
        boxsize = 8
        bordersize = 1
        vals = '1235'
        arr = pygame.PixelArray(surf)
        for x in range(0, len(arr), boxsize):
            for y in range(0, len(arr[x]), boxsize):
                color = int(''.join([random.choice(vals) + random.choice(vals) for _ in range(3)]), 16)
                for LX in range(x, x+(boxsize - bordersize)):
                    for LY in range(y, y+(boxsize - bordersize)):
                        if LX < len(arr) and LY < len(arr[x]):
                            arr[LX][LY] = color
        del arr
        return surf
