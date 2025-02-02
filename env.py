import os
import torch


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
    def __init__(self, batch_size, height, width, num_eat):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.num_eat = num_eat
        self.space = torch.full((batch_size, height, width), EMPTY)
        self.space[:, 0, :] = WALL
        self.space[:, height-1, :] = WALL
        self.space[:, :, 0] = WALL
        self.space[:, :, width-1] = WALL
        for b in range(batch_size):
            indices = torch.randint(0, (height-2) * (width-2), (num_eat,))
            rows = indices // (width-2)
            cols = indices % (width-2)
            self.space[b, 1+rows, 1+cols] = EAT
        self.y = torch.empty(batch_size, dtype=torch.long)
        self.x = torch.empty(batch_size, dtype=torch.long)
        self.score = torch.zeros(batch_size, dtype=torch.float)

    def set_random(self):
        for b in range(self.batch_size):
            indices = torch.randint(0, (self.height-2) * (self.width-2), (1,))
            rows = indices // (self.width-2)
            cols = indices % (self.width-2)
            self.y[b] = rows[0] + 1
            self.x[b] = cols[0] + 1
            self.space[b, self.y[b], self.x[b]] = ME
    
    def get_space(self):
        return self.space

    def interact(self, action):
        rw = torch.empty(self.batch_size, dtype=torch.float)
        for b in range(self.batch_size):
            action_b = action[b]
            new_y = self.y[b]
            new_x = self.x[b]
            if action_b == 'left':
                new_x = self.x[b] - 1
            elif action_b == 'right':
                new_x = self.x[b] + 1
            elif action_b == 'up':
                new_y = self.y[b] - 1
            elif action_b == 'down':
                new_y = self.y[b] + 1
            else:
                raise KeyError('invalid key:' + action_b)
            next_state = self.space[b, new_y, new_x].item()
            if next_state == WALL:
                pass
            else:
                self.space[b, self.y[b], self.x[b]] = EMPTY
                self.space[b, new_y, new_x] = ME
                self.y[b] = new_y
                self.x[b] = new_x
            rw[b] = get_reward(next_state)
            self.score[b] += rw[b]
        return rw

    def print(self):
        os.system('clear')
        print('score:' + str(self.score))
        print(self.space, flush=True)

    def play(self):
        self.set_random()
        self.print()
        while True:
            action_input = input('ation:')
            action = []
            for b in range(self.batch_size):
                if action_input[b] == 'q':
                    return
                elif action_input[b] == 'w':
                    action.append('up')
                elif action_input[b] == 's':
                    action.append('down')
                elif action_input[b] == 'a':
                    action.append('left')
                elif action_input[b] == 'd':
                    action.append('right')
            self.interact(action)
            self.print()


if __name__ == '__main__':
    playground = PlayGround(2, 5, 8, 3)
    playground.play()
