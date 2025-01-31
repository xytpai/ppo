import torch
import os


EMPTY = 0
EAT = 2
WALL = 1
ME = 3


def get_reward(next_state):
    if next_state == EAT:
        return 1
    elif next_state == WALL:
        return -2
    else:
        return -0.1


class PlayGround:
    def __init__(self, height, width, num_eat, score=0):
        self.height = height
        self.width = width
        self.num_eat = num_eat
        self.space = torch.full((self.height, self.width), EMPTY)
        indices = torch.randint(0, (height-2) * (width-2), (num_eat,))
        self.space[0, :] = WALL
        self.space[height-1, :] = WALL
        self.space[:, 0] = WALL
        self.space[:, width-1] = WALL
        rows = indices // (width-2)
        cols = indices % (width-2)
        self.space[1+rows, 1+cols] = EAT
        self.score = score

    def set(self, y, x):
        self.y = y
        self.x = x
        self.space[y, x] = ME

    def set_random(self):
        indices = torch.randint(0, (self.height-2) * (self.width-2), (1,))
        rows = indices // (self.width-2)
        cols = indices % (self.width-2)
        self.y = rows[0] + 1
        self.x = cols[0] + 1
        self.space[self.y, self.x] = ME
    
    def get_space(self):
        return self.space

    def interact(self, action):
        new_y = self.y
        new_x = self.x
        if action == 'left':
            new_x = self.x - 1
        elif action == 'right':
            new_x = self.x + 1
        elif action == 'up':
            new_y = self.y - 1
        elif action == 'down':
            new_y = self.y + 1
        else:
            raise KeyError('invalid key:' + action)
        next_state = self.space[new_y, new_x].item()
        if next_state == WALL:
            pass
        else:
            self.space[self.y, self.x] = EMPTY
            self.space[new_y, new_x] = ME
            self.y = new_y
            self.x = new_x
        rw = get_reward(next_state)
        self.score += rw
        return rw

    def print(self):
        os.system('clear')
        print('score:' + str(self.score))
        print(self.space, flush=True)

    def play(self):
        self.set_random()
        self.print()
        while True:
            action = input('ation:')
            if action == 'q':
                return
            elif action == 'w':
                self.interact('up')
            elif action == 's':
                self.interact('down')
            elif action == 'a':
                self.interact('left')
            elif action == 'd':
                self.interact('right')
            self.print()


if __name__ == '__main__':
    playground = PlayGround(5, 8, 3)
    playground.play()
