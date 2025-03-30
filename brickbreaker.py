import pygame 
import random 
  
pygame.init() 
  
# Dimensions of the screen 
WIDTH, HEIGHT = 600, 500
  
# Colors 
BLACK = (0, 0, 0) 
WHITE = (255, 255, 255) 
GREEN = (0, 255, 0) 
RED = (255, 0, 0) 
  
font = pygame.font.Font('freesansbold.ttf', 15) 
  
screen = pygame.display.set_mode((WIDTH, HEIGHT)) 
pygame.display.set_caption("Block Breaker") 
  
# Frame rate control
clock = pygame.time.Clock() 
FPS = 30

# Striker class 
class Striker: 
    def __init__(self, posx, posy, width, height, speed, color): 
        self.posx, self.posy = posx, posy 
        self.width, self.height = width, height 
        self.speed = speed 
        self.color = color 
        self.strikerRect = pygame.Rect(posx, posy, width, height) 

    def display(self): 
        pygame.draw.rect(screen, self.color, self.strikerRect) 

    def update(self, xFac): 
        self.posx += self.speed * xFac 
        if self.posx <= 0: 
            self.posx = 0
        elif self.posx + self.width >= WIDTH: 
            self.posx = WIDTH - self.width 
        self.strikerRect = pygame.Rect(self.posx, self.posy, self.width, self.height) 

    def getRect(self): 
        return self.strikerRect 

# Block Class 
class Block: 
    def __init__(self, posx, posy, width, height, color): 
        self.posx, self.posy = posx, posy 
        self.width, self.height = width, height 
        self.original_color = color  # Store original color
        self.damage = 100

        if color == WHITE: 
            self.health = 200
        else: 
            self.health = 100

        self.blockRect = pygame.Rect(posx, posy, width, height) 

    def display(self): 
        if self.health > 0:
            # Change color to green if white block has 100 health left
            if self.health == 100 and self.original_color == WHITE:
                current_color = GREEN
            else:
                current_color = self.original_color
            
            pygame.draw.rect(screen, current_color, self.blockRect) 

    def hit(self): 
        self.health -= self.damage 

    def getRect(self): 
        return self.blockRect 

    def getHealth(self): 
        return self.health 

# Ball Class 
class Ball: 
    def __init__(self, posx, posy, radius, speed, color): 
        self.posx, self.posy = posx, posy 
        self.radius = radius 
        self.speed = speed 
        self.color = color 
        self.xFac, self.yFac = 1, 1
        self.ball = pygame.draw.circle(screen, color, (posx, posy), radius) 

    def display(self): 
        self.ball = pygame.draw.circle(screen, self.color, 
                                     (self.posx, self.posy), self.radius) 

    def update(self): 
        self.posx += self.xFac * self.speed 
        self.posy += self.yFac * self.speed 
        if self.posx <= 0 or self.posx >= WIDTH: 
            self.xFac *= -1
        if self.posy <= 0: 
            self.yFac *= -1
        if self.posy >= HEIGHT: 
            return True
        return False

    def reset(self): 
        self.posx = 0
        self.posy = HEIGHT 
        self.xFac, self.yFac = 1, -1

    def hit(self): 
        self.yFac *= -1

    def getRect(self): 
        return self.ball 

# Helper functions
def collisionChecker(rect, ball): 
    return pygame.Rect.colliderect(rect, ball)

def populateBlocks(blockWidth, blockHeight, horizontalGap, verticalGap): 
    listOfBlocks = [] 
    for i in range(0, WIDTH, blockWidth + horizontalGap): 
        for j in range(0, HEIGHT//2, blockHeight + verticalGap): 
            listOfBlocks.append(Block(i, j, blockWidth, blockHeight,  
                                    random.choice([WHITE, GREEN]))) 
    return listOfBlocks 

def gameOver(): 
    gameOver = True
    while gameOver: 
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                return False
            if event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_SPACE: 
                    return True

# Game Manager 
def main(): 
    running = True
    lives = 1
    score = 0

    striker = Striker(0, HEIGHT-50, 100, 20, 10, WHITE) 
    strikerXFac = 0

    ball = Ball(0, HEIGHT-150, 7, 5, WHITE) 

    blockWidth, blockHeight = 40, 15
    horizontalGap, verticalGap = 20, 20

    listOfBlocks = populateBlocks(blockWidth, blockHeight, horizontalGap, verticalGap) 

    while running: 
        screen.fill(BLACK) 
        scoreText = font.render("Score : " + str(score), True, WHITE) 
        screen.blit(scoreText, (10, HEIGHT-30)) 

        if not listOfBlocks: 
            listOfBlocks = populateBlocks(blockWidth, blockHeight, horizontalGap, verticalGap) 

        if lives <= 0: 
            running = gameOver() 
            while listOfBlocks: 
                listOfBlocks.pop(0) 
            lives = 1
            score = 0
            listOfBlocks = populateBlocks(blockWidth, blockHeight, horizontalGap, verticalGap) 

        # Event handling
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                running = False
            if event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_LEFT: 
                    strikerXFac = -1
                if event.key == pygame.K_RIGHT: 
                    strikerXFac = 1
            if event.type == pygame.KEYUP: 
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT): 
                    strikerXFac = 0

        # Collision detection
        if collisionChecker(striker.getRect(), ball.getRect()): 
            ball.hit() 
        for block in listOfBlocks: 
            if collisionChecker(block.getRect(), ball.getRect()): 
                ball.hit() 
                block.hit() 
                if block.getHealth() <= 0: 
                    listOfBlocks.pop(listOfBlocks.index(block)) 
                    score += 5

        # Update game state
        striker.update(strikerXFac) 
        lifeLost = ball.update() 

        if lifeLost: 
            lives -= 1
            ball.reset() 

        # Draw objects
        striker.display() 
        ball.display() 
        for block in listOfBlocks: 
            block.display() 

        pygame.display.update() 
        clock.tick(FPS) 

if __name__ == "__main__": 
    main() 
    pygame.quit()