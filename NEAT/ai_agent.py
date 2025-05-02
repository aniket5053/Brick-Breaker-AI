import neat
import pickle
import os
import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from breaker import level_1_to_4, CANVAS_WIDTH, PADDLE_WIDTH, PADDLE_Y, CANVAS_HEIGHT, BALL_SIZE
import visualize
import threading
import time
import random
import argparse

# Global declaration for eval_genomes
global eval_genomes


# Constants for the AI
INPUT_NODES = 6  # Ball x, ball y, ball dx, ball dy, paddle x, closest brick y
OUTPUT_NODES = 2  # Left, Right
POPULATION_SIZE = 100
GENERATIONS = 200
PADDLE_SPEED = 10
BLOCK_WIDTH = 60
BLOCK_HEIGHT = 30
SIMULATION_SPEED = 1.0 

# Move PlotReporter outside of run_neat function so it can be pickled
class PlotReporter(neat.reporting.BaseReporter):
    def __init__(self, fitness_history, species_history, level):
        self.fitness_history = fitness_history
        self.species_history = species_history
        self.level = level
        
    def post_evaluate(self, config, population, species, best_genome):
        if not hasattr(best_genome, 'fitness') or best_genome.fitness is None:
            return
        
        if len(self.fitness_history) == 0 or best_genome.fitness > max(self.fitness_history):
            print(f"\nNew best fitness: {best_genome.fitness}")
            # Save the best genome so far
            with open(f'best_so_far_level_{self.level}.pkl', 'wb') as output:
                pickle.dump(best_genome, output, 1)
        
        # Add current generation's stats to history
        self.fitness_history.append(best_genome.fitness)
        self.species_history.append(len(species.species))
        
        # Update plot if needed
        try:
            update_plot(self.fitness_history, self.species_history, self.level)
        except Exception as e:
            print(f"Error updating plot: {e}")

def update_plot(fitness_history, species_history, level):
    """Update the matplotlib plot with current fitness and species data"""
    plt.figure(1)
    plt.clf()
    
    # Create two subplots
    plt.subplot(2, 1, 1)
    plt.plot(fitness_history, 'b-', label="Best Fitness")
    plt.title(f'Fitness History - Level {level}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(species_history, 'r-')
    plt.title('Species History')
    plt.xlabel('Generation')
    plt.ylabel('Number of Species')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'stats_level_{level}.png')
    
    # Use pause to update the plot safely
    try:
        plt.pause(0.01)
    except Exception as e:
        print(f"Plot update error: {e}")

class GameState:
    """Class to track the state of the game for NEAT training"""
    def __init__(self):
        self.ball_pos = (0, 0)
        self.ball_vel = (0, 0)
        self.paddle_pos = 0
        self.score = 0
        self.bricks = []
        self.game_over = False
        self.closest_brick_y = CANVAS_HEIGHT

class GameAI:
    def __init__(self, level=1, visualize_training=False, simulation_speed=SIMULATION_SPEED):
        self.level = level
        self.game_params = {
            1: ([-3], [8], 1/50),
            2: ([-3], [8], 1/100),
            3: ([-5], [12], 1/80),
            4: ([-1, -15, -8], [8], 1/50),
            5: ([-3], [12], 1/60)  # Level 5 with moving obstacles. none of this works
        }
        self.current_game = None
        self.game_state = GameState()
        self.fitness = 0
        self.visualize_training = visualize_training
        self.paddle_move_direction = 0  # -1 for left, 0 for none, 1 for right
        self.simulation_speed = simulation_speed  # Speed multiplier for simulation
        
        if self.visualize_training:
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((CANVAS_WIDTH, CANVAS_HEIGHT))
                self.font = pygame.font.Font(None, 36)
                self.clock = pygame.time.Clock()
                pygame.display.set_caption(f"Brick Breaker AI - Level {level} (Speed: {simulation_speed}x)")
                
                # Add buttons to control simulation speed
                self.speed_up_rect = pygame.Rect(CANVAS_WIDTH - 100, 10, 40, 30)
                self.speed_down_rect = pygame.Rect(CANVAS_WIDTH - 150, 10, 40, 30)
                
            except Exception as e:
                print(f"Warning: Could not initialize visualization: {e}")
                self.visualize_training = False

    def get_state(self):
        "Get the current state of the game for the neural network"
        # Normalize values between 0 and 1
        ball_x = self.game_state.ball_pos[0] / CANVAS_WIDTH
        ball_y = self.game_state.ball_pos[1] / CANVAS_HEIGHT
        ball_dx = (self.game_state.ball_vel[0] + 15) / 30  # Normalize between -15 and 15
        ball_dy = (self.game_state.ball_vel[1] + 15) / 30
        paddle_x = self.game_state.paddle_pos / CANVAS_WIDTH
        closest_brick = self.game_state.closest_brick_y / CANVAS_HEIGHT
        
        return [ball_x, ball_y, ball_dx, ball_dy, paddle_x, closest_brick]

    def make_move(self, net):
        "Use the neural network to make a move"
        try:
            state = self.get_state()
            output = net.activate(state)
            
            # Make sure output has at least 2 elements
            if len(output) < 2:
                print(f"Warning: Neural network output has only {len(output)} elements, expected at least 2")
                # Default to no movement if output is invalid
                self.paddle_move_direction = 0
                return self.paddle_move_direction
            
            # Determine movement based on output
            if output[0] > 0.5:  # Left
                self.paddle_move_direction = -1
            elif output[1] > 0.5:  # Right
                self.paddle_move_direction = 1
            else:
                self.paddle_move_direction = 0
            
            # Update paddle position
            new_pos = self.game_state.paddle_pos + (self.paddle_move_direction * PADDLE_SPEED)
            self.game_state.paddle_pos = max(0, min(CANVAS_WIDTH - PADDLE_WIDTH, new_pos))
            
        except Exception as e:
            print(f"Error in make_move: {e}")
            self.paddle_move_direction = 0  # Default to no movement on error
        
        return self.paddle_move_direction

    def update_closest_brick(self):
        "Update the y-coordinate of the closest brick"
        if not self.game_state.bricks:
            self.game_state.closest_brick_y = CANVAS_HEIGHT
            return
        
        closest_y = CANVAS_HEIGHT
        for brick in self.game_state.bricks:
            brick_y = brick[1]
            if brick_y < closest_y:
                closest_y = brick_y
        self.game_state.closest_brick_y = closest_y

    def start_game_thread(self, dx, dy, pause):
        "Start the game in a separate thread and provide callbacks to get game state"
        def update_game_state(ball_pos, paddle_pos, score, bricks, game_over):
            "Callback to update the game state from the game thread"
            self.game_state.ball_pos = ball_pos
            self.game_state.paddle_pos = paddle_pos
            self.game_state.score = score
            self.game_state.bricks = bricks
            self.game_state.game_over = game_over
            
            # Calculate ball velocity 
            if not hasattr(self, "last_ball_pos"):
                self.last_ball_pos = ball_pos
            else:
                dx = ball_pos[0] - self.last_ball_pos[0]
                dy = ball_pos[1] - self.last_ball_pos[1]
                self.game_state.ball_vel = (dx, dy)
                self.last_ball_pos = ball_pos
            
            # Update fitness
            self.update_fitness()
            self.update_closest_brick()
        
        def get_paddle_move():
            "Callback to get the paddle move from the AI"
            return self.paddle_move_direction
        
        def get_simulation_speed():
            "Callback to get the current simulation speed"
            return self.simulation_speed
        
        # Create a custom game simulation instead of using Tkinter
        def run_game_simulation():
            # Initialize game state
            ball_pos = [CANVAS_WIDTH // 3, 100]  # Start more toward the center
            ball_vel = [dx[0], dy[0]]  # Initial velocity
            paddle_pos = CANVAS_WIDTH // 2 - PADDLE_WIDTH // 2
            score = 0
            game_over = False
            
            # Create bricks, This is what i use instead of the other game.
            bricks = []
            for i in range(3):  # 3 rows of bricks
                for j in range(CANVAS_WIDTH // BLOCK_WIDTH):  # Columns based on screen width
                    bricks.append([j * BLOCK_WIDTH, i * BLOCK_HEIGHT])
            
            # Initial game state update
            update_game_state(
                (ball_pos[0], ball_pos[1]),
                paddle_pos,
                score,
                bricks,
                game_over
            )
            
            # Main game loop
            while not game_over:
                # Get current simulation speed
                current_speed = get_simulation_speed()
                
                # Move paddle based on AI input
                move_direction = get_paddle_move()
                # Scale paddle movement based on speed
                speed_adjusted_paddle_speed = PADDLE_SPEED * current_speed
                if move_direction < 0:
                    paddle_pos = max(0, paddle_pos - speed_adjusted_paddle_speed)
                elif move_direction > 0:
                    paddle_pos = min(CANVAS_WIDTH - PADDLE_WIDTH, paddle_pos + speed_adjusted_paddle_speed)
                
                # Scale ball movement based on speed
                speed_adjusted_ball_vel = [
                    ball_vel[0] * current_speed,
                    ball_vel[1] * current_speed
                ]
                
                # Move ball
                ball_pos[0] += speed_adjusted_ball_vel[0]
                ball_pos[1] += speed_adjusted_ball_vel[1]
                
                # Ball collision with walls
                if ball_pos[0] <= 0 or ball_pos[0] >= CANVAS_WIDTH - BALL_SIZE:
                    ball_vel[0] = -ball_vel[0]
                    # Keep ball within bounds
                    ball_pos[0] = max(0, min(CANVAS_WIDTH - BALL_SIZE, ball_pos[0]))
                
                if ball_pos[1] <= 0:
                    ball_vel[1] = -ball_vel[1]
                    # Keep ball within bounds
                    ball_pos[1] = 0
                
                # Ball collision with paddle
                if (ball_pos[1] + BALL_SIZE >= PADDLE_Y and 
                    ball_pos[1] <= PADDLE_Y + 10 and
                    ball_pos[0] + BALL_SIZE >= paddle_pos and
                    ball_pos[0] <= paddle_pos + PADDLE_WIDTH):
                    ball_vel[1] = -abs(ball_vel[1])  # Always bounce up
                    # Randomly change ball direction sometimes
                    if random.random() < 0.3:
                        ball_vel[0] = random.choice(dx)
                    
                    # Ensure the ball is above the paddle
                    ball_pos[1] = PADDLE_Y - BALL_SIZE - 1
                
                # Ball collision with bricks
                collision = False
                for brick in bricks[:]:
                    if (ball_pos[0] + BALL_SIZE >= brick[0] and
                        ball_pos[0] <= brick[0] + BLOCK_WIDTH and
                        ball_pos[1] + BALL_SIZE >= brick[1] and
                        ball_pos[1] <= brick[1] + BLOCK_HEIGHT):
                        
                        # Don't remove the same brick twice
                        if not collision:
                            bricks.remove(brick)
                            ball_vel[1] = -ball_vel[1]
                            score += 1
                            collision = True
                
                # Check if game is over
                if ball_pos[1] >= CANVAS_HEIGHT - BALL_SIZE or not bricks:
                    game_over = True
                
                # Update game state
                update_game_state(
                    (ball_pos[0], ball_pos[1]),
                    paddle_pos,
                    score,
                    bricks,
                    game_over
                )
                
                # Dynamic pause to control game speed based on simulation speed
                adjusted_pause = pause / max(0.1, current_speed)  # Ensure we don't divide by zero
                time.sleep(adjusted_pause)
        
        # Start the game simulation in a separate thread
        game_thread = threading.Thread(target=run_game_simulation)
        game_thread.daemon = True
        game_thread.start()
        return game_thread

    def run_game(self, genome, config):
        """Run a single game with the given genome"""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.fitness = 0
        self.game_state = GameState()
        
        # Start the game with the current level parameters
        dx, dy, pause = self.game_params[self.level]
        # Adjust pause time based on simulation speed
        adjusted_pause = pause / self.simulation_speed
        game_thread = self.start_game_thread(dx, dy, adjusted_pause)
        
        if self.visualize_training:
            self.screen.fill((255, 255, 255))
        
        # Main AI control loop
        current_fitness = 0.0  # Initialize current fitness display value
        while not self.game_state.game_over:
            # Get game state and make move
            self.make_move(net)
            
            # Update current fitness display value
            if hasattr(genome, 'fitness') and genome.fitness is not None:
                current_fitness = genome.fitness
            else:
                current_fitness = self.fitness
            
            if self.visualize_training:
                self.draw_game_state(current_fitness)
                pygame.display.flip()
                # Handle events including speed control
                self.handle_events()
                # Adjust frame rate based on simulation speed
                self.clock.tick(60 * self.simulation_speed)
            
            # Small delay to not overwhelm the CPU - adjusted for speed
            delay = max(0.001, 0.01 / self.simulation_speed)
            time.sleep(delay)
        
        # Wait for game thread to finish
        game_thread.join(timeout=1.0)
        
        genome.fitness = self.fitness
        return self.fitness

    def draw_game_state(self, fitness):
        "Draw the current game state for visualization"
        try:
            if not hasattr(self, 'screen') or self.screen is None:
                return
            
            self.screen.fill((255, 255, 255))
            
            # Draw paddle
            pygame.draw.rect(self.screen, (0, 0, 0), 
                            (self.game_state.paddle_pos, PADDLE_Y, PADDLE_WIDTH, 10))
            
            # Draw ball
            pygame.draw.circle(self.screen, (255, 0, 0), 
                             (int(self.game_state.ball_pos[0]), int(self.game_state.ball_pos[1])), 
                             BALL_SIZE // 2)
            
            # Draw bricks
            for brick in self.game_state.bricks:
                pygame.draw.rect(self.screen, (0, 0, 255), 
                               (brick[0], brick[1], 60, 30))
            
            # Draw fitness and score
            # Handle the case when fitness is None
            fitness_value = 0.0 if fitness is None else fitness
            fitness_text = self.font.render(f'Fitness: {fitness_value:.2f}', True, (0, 0, 0))
            score_text = self.font.render(f'Score: {self.game_state.score}', True, (0, 0, 0))
            level_text = self.font.render(f'Level: {self.level}', True, (0, 0, 0))
            speed_text = self.font.render(f'Speed: {self.simulation_speed:.1f}x', True, (0, 0, 0))
            
            self.screen.blit(fitness_text, (10, 10))
            self.screen.blit(score_text, (10, 50))
            self.screen.blit(level_text, (10, 90))
            self.screen.blit(speed_text, (10, 130))
            
            # Draw speed control buttons
            pygame.draw.rect(self.screen, (150, 150, 255), self.speed_up_rect)
            pygame.draw.rect(self.screen, (255, 150, 150), self.speed_down_rect)
            speed_up_text = self.font.render('+', True, (0, 0, 0))
            speed_down_text = self.font.render('-', True, (0, 0, 0))
            self.screen.blit(speed_up_text, (self.speed_up_rect.x + 15, self.speed_up_rect.y + 5))
            self.screen.blit(speed_down_text, (self.speed_down_rect.x + 15, self.speed_down_rect.y + 5))
            
        except Exception as e:
            print(f"Visualization error: {e}")
            # Disable visualization if there's an error
            self.visualize_training = False

    def handle_events(self):
        "Handle pygame events including speed control"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                # Check if speed control buttons were clicked
                if self.speed_up_rect.collidepoint(mouse_pos):
                    self.simulation_speed = min(10.0, self.simulation_speed + 0.5)
                    pygame.display.set_caption(f"Brick Breaker AI - Level {self.level} (Speed: {self.simulation_speed}x)")
                elif self.speed_down_rect.collidepoint(mouse_pos):
                    self.simulation_speed = max(0.5, self.simulation_speed - 0.5)
                    pygame.display.set_caption(f"Brick Breaker AI - Level {self.level} (Speed: {self.simulation_speed}x)")

    def update_fitness(self):
        "Update the fitness score based on game state"
        # Calculate total number of bricks at the start (3 rows * width of screen / brick width)
        total_bricks = 3 * (CANVAS_WIDTH // BLOCK_WIDTH)
        remaining_bricks = len(self.game_state.bricks)
        cleared_bricks = total_bricks - remaining_bricks
        
        # Big reward for clearing all bricks
        if remaining_bricks == 0:
            self.fitness += 1000  # Huge bonus for clearing all bricks
        
        # Main reward is based on percentage of bricks cleared
        brick_clear_percentage = cleared_bricks / total_bricks
        self.fitness = brick_clear_percentage * 500  # Scale to make it meaningful
        
        # Additional rewards/penalties
        # Smaller reward for keeping the ball in play
        self.fitness += 0.5
        
        # Reward for hitting the ball with the paddle
        if self.game_state.ball_pos[1] >= PADDLE_Y - 20 and abs(self.game_state.ball_pos[0] - self.game_state.paddle_pos) < PADDLE_WIDTH:
            self.fitness += 1.0
        
        # Reward for moving towards the ball (smaller weight now)
        ball_direction = 1 if self.game_state.ball_pos[0] > self.game_state.paddle_pos else -1
        paddle_direction = self.paddle_move_direction
        if ball_direction == paddle_direction:
            self.fitness += 1.0
        
        # Heavy penalty for missing the ball
        if self.game_state.ball_pos[1] >= CANVAS_HEIGHT - 20:
            self.fitness -= 10.0

def eval_genomes(genomes, config):
    "Evaluate all genomes in the population"
    # Visualize only for the first few genomes to speed up training
    for i, (genome_id, genome) in enumerate(genomes):
        # Show visualization for first genome and every 10th genome
        show_visual = i < 1 or i % 10 == 0
        # Use faster simulation speed for training
        game = GameAI(level=1, visualize_training=show_visual, simulation_speed=1.0)
        genome.fitness = game.run_game(genome, config)
        print(f"Genome {i+1}/{len(genomes)}: Fitness = {genome.fitness}")

def run_neat(config_file, level=1):
    "Run the NEAT algorithm"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    
    # Create the population
    p = neat.Population(config)
    
    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=f'checkpoint_level_{level}_'))
    
    # Initialize history arrays for plotting
    fitness_history = []
    species_history = []
    
    # Initialize the plot
    plt.figure(figsize=(10, 8))
    plt.ion() # Turn on interactive mode
    plt.show()
    
    # Add the custom plot reporter (defined outside of this function)
    plot_reporter = PlotReporter(fitness_history, species_history, level)
    p.add_reporter(plot_reporter)
    
    # Run for specified number of generations
    winner = p.run(eval_genomes, GENERATIONS)
    
    # Save the winner
    with open(f'winner_level_{level}.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
    
    # Visualize the winner network
    try:
        visualize.plot_genome_structure(config, winner, view=True, filename=f"network_level_{level}")
        visualize.plot_stats(stats, ylog=False, view=True, filename=f'stats_level_{level}.svg')
        visualize.plot_species(stats, view=True, filename=f'species_level_{level}.svg')
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Close the plot
    plt.ioff()
    plt.close()
    
    return winner

def play_with_winner(config_file, level=1, simulation_speed=1.0):
    "Load the winner and play the game"
    try:
        with open(f'winner_level_{level}.pkl', 'rb') as input_file:
            winner = pickle.load(input_file)
    except FileNotFoundError:
        print(f"No trained model found for level {level}. Training new model...")
        winner = run_neat(config_file, level)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    
    # Create a game instance with visualization enabled
    game = GameAI(level=level, visualize_training=True, simulation_speed=simulation_speed)
    
    print(f"\nWatching the trained AI play level {level}...")
    fitness = game.run_game(winner, config)
    print(f"Final fitness: {fitness}")
    
    # Keep the window open until the user closes it
    if game.visualize_training:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check for speed control button clicks
                    mouse_pos = pygame.mouse.get_pos()
                    if game.speed_up_rect.collidepoint(mouse_pos):
                        game.simulation_speed = min(10.0, game.simulation_speed + 0.5)
                    elif game.speed_down_rect.collidepoint(mouse_pos):
                        game.simulation_speed = max(0.5, game.simulation_speed - 0.5)
                    pygame.display.set_caption(f"Brick Breaker AI - Level {level} (Speed: {game.simulation_speed}x)")
            
            # Keep updating the display
            game.draw_game_state(fitness)
            pygame.display.flip()
            time.sleep(0.05)
        
        pygame.quit()

def train_all_levels():
    """Train the AI on all levels sequentially"""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    for level in range(1, 5):
        print(f"\nTraining on Level {level}...")
        run_neat(config_path, level)
        print(f"\nTesting Level {level}...")
        play_with_winner(config_path, level)

def test_game(level=1, simulation_speed=1.0):
    """Run the game with a simple fixed policy to verify the environment works."""
    print(f"\nTesting game environment with a simple fixed policy on level {level}...")
    
    # Create a game instance with visualization
    game = GameAI(level=level, visualize_training=True, simulation_speed=simulation_speed)
    
    # Create a simple fixed policy that just follows the ball
    class SimplePolicy:
        def __init__(self):
            self.fitness = 0
            
        def activate(self, inputs):
            # Simple policy: move paddle toward the ball
            ball_x = inputs[0]  # Normalized ball x position
            paddle_x = inputs[4]  # Normalized paddle position
            
            # Follow the ball
            if ball_x < paddle_x - 0.05:
                return [1.0, 0.0]  # Move left
            elif ball_x > paddle_x + 0.05:
                return [0.0, 1.0]  # Move right
            else:
                return [0.0, 0.0]  # Stay in place
    
    # Create a dummy config for the game
    class DummyConfig:
        pass
    
    # Create a simple policy
    policy = SimplePolicy()
    config = DummyConfig()
    
    # Run the game with the simple policy
    fitness = game.run_game(policy, config)
    print(f"Simple policy finished with fitness: {fitness}")
    
    # Keep the visualization window open
    if game.visualize_training:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check for speed control button clicks
                    mouse_pos = pygame.mouse.get_pos()
                    if game.speed_up_rect.collidepoint(mouse_pos):
                        game.simulation_speed = min(10.0, game.simulation_speed + 0.5)
                    elif game.speed_down_rect.collidepoint(mouse_pos):
                        game.simulation_speed = max(0.5, game.simulation_speed - 0.5)
                    pygame.display.set_caption(f"Brick Breaker AI - Level {level} (Speed: {game.simulation_speed}x)")
            
            # Keep updating the display
            game.draw_game_state(fitness)
            pygame.display.flip()
            time.sleep(0.05)
        
        pygame.quit()

if __name__ == '__main__':
    # Get the path to the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    # Update the existing config file to ensure output nodes are correctly set
    print("Verifying config file...")
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Check if num_outputs is correctly set
    if 'num_outputs' in config_content:
        import re
        # Update num_outputs if needed
        config_content = re.sub(
            r'num_outputs\s*=\s*\d+', 
            f'num_outputs            = {OUTPUT_NODES}', 
            config_content
        )
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"Updated config file to set num_outputs = {OUTPUT_NODES}")
    
    # Command line option to choose what to do

    #These were originally created to train on multiple levels with different speeds. Did not happen.

    #currently just train using --train 1
    parser = argparse.ArgumentParser(description='Brick Breaker AI with NEAT')
    parser.add_argument('--play', type=int, help='Play with trained AI on level (1-4)', default=1)
    parser.add_argument('--train', type=int, help='Train the AI on specified level (1-5)', default=1)
    parser.add_argument('--test', action='store_true', help='Test game environment with a simple policy')
    parser.add_argument('--speed', type=float, help='Simulation speed multiplier (default: 1.0)', default=1.0)
    parser.add_argument('--train-speed', type=float, help='Training simulation speed multiplier (default: 5.0)', default=2.0)
    args = parser.parse_args()
    
    # Set the simulation speed based on arguments
    SIMULATION_SPEED = args.speed
    
    if args.test:
        # Test the game environment
        game = GameAI(level=1, visualize_training=True, simulation_speed=SIMULATION_SPEED)
        game.test_game()
    elif args.play > 0:
        # Play with trained AI
        level = min(max(args.play, 1), 4)  # Ensure level is between 1 and 4
        play_with_winner(config_path, level, SIMULATION_SPEED)
    elif args.train > 0:
        # Train the AI on specified level
        level = min(max(args.train, 1), 5)  # Ensure level is between 1 and 5
        print(f"Training AI on level {level} at speed {args.train_speed}x...")
        # Create a game instance with the training speed
        game = GameAI(level=level, visualize_training=True, simulation_speed=args.train_speed)
        run_neat(config_path, level)
    else:
        print("Please specify an action: --test, --play <level>, or --train <level>") 
