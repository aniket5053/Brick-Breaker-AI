import pygame
import random
import sys
import time
import csv
import signal
import os
from datetime import datetime
from typing import List, Tuple, Optional

# ----------------------------------------------------------------
#  Game Constants
# ----------------------------------------------------------------
BALL_START_SPEED_DEFAULT = 0
BALL_ACCELERATION_MIN_DEFAULT = 0
BALL_ACCELERATION_MAX_DEFAULT = 1
PADDLE_MOVE_SPEED_DEFAULT = 3
BALL_HIT_OFFSET_MULTIPLIER_DEFAULT = 2.5
PADDLE_VELOCITY_MULTIPLIER_DEFAULT = 1

# ----------------------------------------------------------------
#  Size Constants
# ----------------------------------------------------------------
WIDTH, HEIGHT = 360, 360
PADDLE_WIDTH, PADDLE_HEIGHT = 60, 10
BALL_SIZE = 8
BRICK_WIDTH, BRICK_HEIGHT = 40, 15
COLS, MAX_BRICK_ROWS = 10, 10

# ----------------------------------------------------------------
#  Level Colors
# ----------------------------------------------------------------
LEVEL_COLOR_THEMES: List[Optional[List[Tuple[int, int, int]]]] = [
    [(255, 0, 0),   (255, 165, 0),  (0, 255, 0)],
    [(200, 100, 100), (100, 200, 100), (100, 100, 255)],
    [(255, 255, 0), (255, 0, 255), (0, 255, 255)],
    [(128, 0, 128), (255, 140, 0), (0, 255, 127)],
    None # Random
]

# ----------------------------------------------------------------
#  Hard‑coded layouts
# ----------------------------------------------------------------
HARDCODED_LAYOUTS = {
    0: [(x, y) for x in range(9) for y in range(3)],           # 10x3 grid
    1: [(x, y) for x in range(10) for y in [0, 2, 4]],          # Striped layout
    2: [(x, y) for x in range(0, 10, 2) for y in range(3, 6)] +
       [(x, y) for x in range(1, 10, 2) for y in range(6, 9)],
    3: [(x, y) for y in range(1, 4) for x in range(5)] +
       [(x, y + 5) for y in range(2)  for x in range(5, 10)]
}

CSV_FILE = "breakout_game_log.csv"


# ----------------------------------------------------------------
#  Breakout Game
# ----------------------------------------------------------------
class Breakout:
    """
    - SPACE restarts the current attempt (tracked via `restart_attempts`). Hypothetically
    - Level 0 backend (front - end Level 1) is always a 10x3 brick grid.
    - CSV row written on:
        - every level completion
        - normal quit  / Ctrl-C  / un-caught exception
    - Visual scaling on resize - physics fixed.
    """

    def __init__(self, start_level: int = 0,
                 log_to_csv: bool = True,
                 locked_level: Optional[int] = None):
        self.render_enabled = True
        if self.render_enabled:
                pygame.init()
                pygame.display.set_caption("Breakout")
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                self.clock  = pygame.time.Clock()
                self.font   = pygame.font.SysFont("Comic Sans MS", 14)

        # Window size
        self.window_width, self.window_height = WIDTH, HEIGHT

        # Game progression
        self.locked_level   = locked_level
        self.starting_round = start_level if locked_level is None else locked_level
        self.round          = self.starting_round
        self.total_games_completed = 0
        self.total_levels_logged   = 0

        # Session counters
        self.total_attempts   = 0
        self.total_start_time = time.time()

        # Attempt counters (set in reset)
        self.round_start_time = 0
        self.round_bricks_destroyed = 0
        self.round_paddle_hits      = 0
        self.round_points           = 0

        # Global counters
        self.total_bricks_destroyed = 0
        self.total_paddle_hits      = 0
        self.total_points           = 0
        self.restart_attempts       = 0

        # Game flags
        self.paused = False
        self.running = True
        self.show_game_over = False
        self.round_lost = False
        self.log_to_csv = log_to_csv

        # Others
        self.game_id = 0
        self.level_layouts: dict[int, Tuple[list, list]] = {}
        self.prev_paddle_x = 0
        self.combo_count   = 0

        # Parameters
        self.ball_start_speed           = BALL_START_SPEED_DEFAULT
        self.ball_acceleration_min      = BALL_ACCELERATION_MIN_DEFAULT
        self.ball_acceleration_max      = BALL_ACCELERATION_MAX_DEFAULT
        self.paddle_move_speed          = PADDLE_MOVE_SPEED_DEFAULT
        self.ball_hit_offset_multiplier = BALL_HIT_OFFSET_MULTIPLIER_DEFAULT
        self.paddle_velocity_multiplier = PADDLE_VELOCITY_MULTIPLIER_DEFAULT

        # Ensure CSV header exists (the file is there) before writing
        if self.log_to_csv and not csv_exists(os.path.join("output", CSV_FILE)):
            os.makedirs("output", exist_ok=True)
            with open(os.path.join("output", CSV_FILE), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields())
                writer.writeheader()

        # Graceful exits (ctrl+c) capture
        signal.signal(signal.SIGINT, lambda *_: self.log_and_quit(reason="ctrl_c"))
        sys.excepthook = lambda exc_type, exc, tb: self._excepthook(exc_type, exc, tb)

        self.reset()

    # Setter to toggle rendering
    def set_render_enabled(self, enabled: bool):
        if self.render_enabled == enabled:
            return

        self.render_enabled = enabled

        if enabled:
            pygame.init()
            pygame.display.set_caption("Breakout")
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            self.clock  = pygame.time.Clock()
            self.font   = pygame.font.SysFont("Comic Sans MS", 14)
        else:
            pygame.display.quit()

    # Layout helper functions
    def _generate_unique_bricks(self, n: int):
        cells = set()
        while len(cells) < n:
            cells.add((random.randint(0, COLS-1), random.randint(0, MAX_BRICK_ROWS-1)))
        return list(cells)

    @staticmethod
    def _row_color(pattern, y, cache):
        if pattern is None:
            return random.randint(0,255), random.randint(0,255), random.randint(0,255)
        if y not in cache:
            cache[y] = pattern[y % len(pattern)]
        return cache[y]

    def reset(self):
        """Reset attempt for the current round"""
        self.total_attempts     += 1
        self.round_start_time    = time.time()
        self.round_bricks_destroyed = 0
        self.round_paddle_hits      = 0
        self.round_points           = 0
        self.combo_count            = 0
        self.paused                 = False
        self.running                = True
        self.show_game_over         = False

        # Construct layout only if it is not yet cached
        if self.round not in self.level_layouts:
            if self.round in HARDCODED_LAYOUTS:
                layout, ub = HARDCODED_LAYOUTS[self.round], []
            elif self.round == 5:
                layout, ub = self._generate_unique_bricks(30), []
            elif self.round == 6:
                layout = self._generate_unique_bricks(30)
                ub = [p for p in self._generate_unique_bricks(10) if p not in layout][:10]
            else:
                layout, ub = self._generate_unique_bricks(30), []
            self.level_layouts[self.round] = (layout, ub)

        layout, ub = self.level_layouts[self.round]
        max_y = max((y for _, y in layout + ub), default=0)

        # Create game objects
        self.paddle = pygame.Rect((WIDTH - PADDLE_WIDTH)//2, HEIGHT-30,
                                  PADDLE_WIDTH, PADDLE_HEIGHT)
        self.prev_paddle_x = self.paddle.x
        self.ball = pygame.Rect(random.randint(BALL_SIZE*2, WIDTH-BALL_SIZE*2),
                                (max_y+1)*BRICK_HEIGHT + 10,
                                BALL_SIZE, BALL_SIZE)
        self.ball_dx = random.choice([-1, 1])
        self.ball_dy = 1

        pattern = LEVEL_COLOR_THEMES[min(self.round,4)] if self.round <= 5 else None
        row_cache = {}
        self.bricks: List[dict] = []

        for x, y in layout:
            self.bricks.append(
                {"rect": pygame.Rect(x*BRICK_WIDTH, y*BRICK_HEIGHT, BRICK_WIDTH, BRICK_HEIGHT),
                 "colour": self._row_color(pattern, y, row_cache), "breakable": True}
            )
        for x, y in ub:
            self.bricks.append(
                {"rect": pygame.Rect(x*BRICK_WIDTH, y*BRICK_HEIGHT, BRICK_WIDTH, BRICK_HEIGHT),
                 "colour": (200, 200, 200), "breakable": False}
            )

    def get_state(self):
        sx, sy = self.window_width/WIDTH, self.window_height/HEIGHT
        return (int(self.ball.centerx/(40*sx)),
                int(self.ball.centery/(30*sy)),
                int(self.paddle.centerx/(40*sx)),
                0 if self.ball_dx < 0 else 1,
                0 if self.ball_dy < 0 else 1)

    def step(self, action: int):
        if not self.running:
            return self.get_state(), 0.0, True

        # Paddle input movement
        if action == 0:
            self.paddle.move_ip(-self.paddle_move_speed, 0)
        elif action == 1:
            self.paddle.move_ip(self.paddle_move_speed, 0)
        self.paddle.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))
        pv = self.paddle.x - self.prev_paddle_x
        self.prev_paddle_x = self.paddle.x

        # Ball movement adjustment
        self.ball.move_ip(self.ball_dx, self.ball_dy)
        if abs(self.ball_dy) < self.ball_acceleration_min:
            self.ball_dy = self.ball_acceleration_min if self.ball_dy > 0 else -self.ball_acceleration_min
        elif abs(self.ball_dy) > self.ball_acceleration_max:
            self.ball_dy = self.ball_acceleration_max if self.ball_dy > 0 else -self.ball_acceleration_max

        points_earned, done = 0.0, False

        # Define wall boundaries for ball collision
        if self.ball.left <= 0 or self.ball.right >= WIDTH:
            self.ball_dx *= -1
        if self.ball.top <= 0:
            self.ball_dy *= -1

        # Paddle collision and ball angle adjustment
        if self.ball.colliderect(self.paddle) and self.ball_dy > 0:
            self.ball.bottom = self.paddle.top
            offset = (self.ball.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_dx += self.ball_hit_offset_multiplier*offset + self.paddle_velocity_multiplier*pv
            self.ball_dx = max(min(self.ball_dx, self.ball_acceleration_max), -self.ball_acceleration_max)
            self.ball_dx += random.choice([0, 1]) # Prevent cycles by pushing an offset
            self.ball_dy = -abs(self.ball_dy)
            self.total_paddle_hits += 1
            self.round_paddle_hits += 1
            self.combo_count = 0

        # Brick collision - Ensure no phase through
        hit_idx = next((i for i, b in enumerate(self.bricks) if self.ball.colliderect(b["rect"])), None)
        if hit_idx is not None:
            brick = self.bricks[hit_idx]
            r = brick["rect"]

            # Corner angle adjustment
            ol, or_, ot, ob = self.ball.right-r.left, r.right-self.ball.left, self.ball.bottom-r.top, r.bottom-self.ball.top
            minx, miny = min(ol, or_), min(ot, ob)
            horiz = vert = False
            if minx < miny:
                self.ball_dx *= -1; horiz = True
            elif miny < minx:
                self.ball_dy *= -1; vert  = True
            else:
                self.ball_dx *= -1; self.ball_dy *= -1; horiz = vert = True

            # Push ball slightly so that it doesn't get stuck in infinite loop, or stuck
            if horiz:
                if ol < or_:
                    self.ball.right = r.left - 1
                else:
                    self.ball.left  = r.right + 1
            if vert:
                if ot < ob:
                    self.ball.bottom = r.top - 1
                else:
                    self.ball.top    = r.bottom + 1

            # Score calculation
            if brick["breakable"]:
                del self.bricks[hit_idx]
                self.combo_count += 1
                gain = self.combo_count
                points_earned      += gain
                self.round_points  += gain
                self.total_points  += gain
                self.round_bricks_destroyed += 1
                self.total_bricks_destroyed += 1

        # If the round is complete
        if not any(b["breakable"] for b in self.bricks):
            self._log_csv(event="level_complete")   # log right now
            done = True
            if self.locked_level is None:
                self.round += 1
                if self.round > 6:                  # game complete
                    self.total_games_completed += 1
                    self._log_csv(event="game_complete")
                    self.round = 0                  # back to Level 1 for front‑end
                    self.game_id += 1
                    self.show_game_over = True

        # If the ball goes out the bottom, misses the paddle
        if self.ball.bottom >= HEIGHT:
            done = True
            self.round_points = 0
            self.round_lost = True
        else:
            self.round_lost = False
        
        return self.get_state(), points_earned, done

    def render(self):
        if not self.render_enabled:
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.log_and_quit(reason="quit_button")
            elif e.type == pygame.VIDEORESIZE:
                self.window_width, self.window_height = e.size
                self.screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.paused = not self.paused
                elif e.key == pygame.K_SPACE:
                    self.restart_attempts += 1
                    self.reset()

        if self.paused:
            self._draw_center("PAUSED – ESC to resume")
            return

        sx, sy = self.window_width/WIDTH, self.window_height/HEIGHT
        self.screen.fill((0, 0, 0))

        # Brick render
        for b in self.bricks:
            r = b["rect"]
            dst = pygame.Rect(r.x*sx, r.y*sy, r.width*sx, r.height*sy)
            if b["breakable"]:
                pygame.draw.rect(self.screen, b["colour"], dst)
                pygame.draw.rect(self.screen, (0,0,0), dst, 2)
            else:
                pygame.draw.rect(self.screen, b["colour"], dst, 2)
                pygame.draw.line(self.screen, b["colour"], dst.topleft,  dst.bottomright, 2)
                pygame.draw.line(self.screen, b["colour"], dst.topright, dst.bottomleft,  2)

        # Paddle and ball render
        pygame.draw.rect(self.screen, (255,255,255),
                         pygame.Rect(self.paddle.x*sx, self.paddle.y*sy,
                                     self.paddle.width*sx, self.paddle.height*sy))
        pygame.draw.ellipse(self.screen, (255,0,0),
                            pygame.Rect(self.ball.x*sx, self.ball.y*sy,
                                        self.ball.width*sx, self.ball.height*sy))

        self._draw_metrics(sx, sy)
        if self.show_game_over:
            self._draw_center("GAME OVER – SPACE to restart")

        pygame.display.flip()
        self.clock.tick(60*5)

    # UI Helpers
    def _draw_center(self, msg: str):
        surf = self.font.render(msg, True, (255,0,0))
        rect = surf.get_rect(center=(self.window_width//2, self.window_height//2))
        self.screen.blit(surf, rect)
        pygame.display.flip()

    def _draw_metrics(self, sx: float, sy: float):
        total = time.time() - self.total_start_time
        rt    = time.time() - self.round_start_time
        lines = [
            f"Game: {self.game_id+1}",
            f"Level: {self.round+1}",
            f"Attempts: {self.total_attempts}",
            f"Restarts: {self.restart_attempts}",
            f"Bricks: {self.round_bricks_destroyed}",
            f"Paddle Hits: {self.round_paddle_hits}",
            f"Points: {self.round_points}",
            f"Games Completed: {self.total_games_completed}",
            f"Round Time: {rt:.1f}s",
            f"Total Time: {total:.1f}s",
            f"Levels Logged: {self.total_levels_logged}"
        ]
        lh = int(18*sy)
        x0 = 5*sx
        y0 = (MAX_BRICK_ROWS*BRICK_HEIGHT)*sy - PADDLE_HEIGHT*3
        maxw = 0
        for i, txt in enumerate(lines):
            s = self.font.render(txt, True, (255,255,255))
            if s.get_width() > maxw:
                maxw = s.get_width()
            self.screen.blit(s, (x0, y0+i*lh))
        pad = 4*sx
        pygame.draw.rect(self.screen, (255,255,255),
                         pygame.Rect(x0-pad, y0-pad, maxw+pad*2, len(lines)*lh+pad*2),
                         1)

    # CSV Logging
    def _log_csv(self, event: str):
        """Write one row to CSV for `event`."""
        if not self.log_to_csv:
            return

        if event == "level_complete":
            self.total_levels_logged += 1

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "game_id": self.game_id,
            "event": event,
            "level_logged": self.total_levels_logged if event == "level_complete" else "-",
            "current_level": self.round+1,
            "attempts_so_far": self.total_attempts,
            "restart_attempts": self.restart_attempts,
            "total_bricks_destroyed": self.total_bricks_destroyed,
            "total_paddle_hits": self.total_paddle_hits,
            "points_curr": self.round_points,
            "total_points": self.total_points
        }

        file_exists = csv_exists(os.path.join("output", CSV_FILE))

        with open(os.path.join("output", CSV_FILE), "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
            print(f"Game metrics saved to './output/{CSV_FILE}' for event '{event}'")

    def log_and_quit(self, reason="quit"):
        self._log_csv(event=reason)
        pygame.quit()
        sys.exit()

    def _excepthook(self, exc_type, exc, tb):
        self._log_csv(event="crash")
        sys.__excepthook__(exc_type, exc, tb)

    # Close response
    def close(self):
        self.log_and_quit(reason="close_called")

    # Getters and setters
    def get_ball_start_speed(self): return self.ball_start_speed
    def set_ball_start_speed(self, v): self.ball_start_speed = v

    def get_ball_acceleration_min(self): return self.ball_acceleration_min
    def set_ball_acceleration_min(self, v): self.ball_acceleration_min = v

    def get_ball_acceleration_max(self): return self.ball_acceleration_max
    def set_ball_acceleration_max(self, v): self.ball_acceleration_max = v

    def get_paddle_move_speed(self): return self.paddle_move_speed
    def set_paddle_move_speed(self, v): self.paddle_move_speed = v

    def get_ball_hit_offset_multiplier(self): return self.ball_hit_offset_multiplier
    def set_ball_hit_offset_multiplier(self, v): self.ball_hit_offset_multiplier = v

    def get_paddle_velocity_multiplier(self): return self.paddle_velocity_multiplier
    def set_paddle_velocity_multiplier(self, v): self.paddle_velocity_multiplier = v

    def get_bricks_broken_attempt(self):    return self.round_bricks_destroyed
    def get_current_points(self):           return self.round_points
    def get_attempt_time(self):             return time.time() - self.round_start_time
    def get_restart_attempts(self):         return self.restart_attempts
    def get_current_level(self):            return self.total_levels_logged
    def get_number_paddle_hits_round(self): return self.round_paddle_hits
    def get_isRoundLost(self):              return self.round_lost
    def get_total_attempts_round(self):     return self.total_attempts

# Main test (user input)
if __name__ == "__main__":
    game = Breakout(start_level=0, log_to_csv=False)
    while True:
        keys = pygame.key.get_pressed()
        action = 0 if keys[pygame.K_LEFT] else 1 if keys[pygame.K_RIGHT] else 2
        _, _, done = game.step(action)
        game.render()
        if done:
            game.reset()

# CSV Helpers
def csv_fields():
    return ["timestamp", "game_id", "event", "level_logged", "current_level",
            "attempts_so_far", "restart_attempts", "total_bricks_destroyed", 
            "total_paddle_hits", "points_curr", "total_points"]


def csv_exists(path):
    try:
        with open(path, "r"):
            return True
    except FileNotFoundError:
        return False
