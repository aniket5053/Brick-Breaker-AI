import pygame
from breakout import Breakout

def main():
    game = Breakout(start_level=0, log_to_csv=True)
    current_state = game.get_state()

    while True:
        action = -1  # No movement by default

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game.paused = not game.paused
                elif game.show_game_over and event.key == pygame.K_SPACE:
                    game.game_id += 1  # Track the next game number
                    game.reset()

        if not game.paused and not game.show_game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = 0
            elif keys[pygame.K_RIGHT]:
                action = 1

            state, reward, done = game.step(action)
            current_state = state
            if done:
                game.reset()  # Instant respawn

        game.render()

if __name__ == "__main__":
    main()
