import io
import numpy as np
import sys

# Import the local discrete environment helper without package context
import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 4x4 grid looks as follows:

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4,4], transition_prob=1.0):
        """
        Inicializa o ambiente GridWorld.
        
        Args:
            shape: dimensões do grid [altura, largura]
            transition_prob: probabilidade de a ação escolhida ser executada.
                            Com prob (1 - transition_prob), a ação "escorrega"
                            uniformemente para uma direção ortogonal.
        """
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        self.transition_prob = transition_prob

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        # Direções ortogonais para cada ação
        # UP/DOWN são ortogonais a LEFT/RIGHT e vice-versa
        orthogonal = {
            UP: [LEFT, RIGHT],
            DOWN: [LEFT, RIGHT],
            LEFT: [UP, DOWN],
            RIGHT: [UP, DOWN]
        }

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == 0 or s == (nS - 1)
            reward = 0.0 if is_done(s) else -1.0

            # Próximos estados para cada ação
            ns_up = s if y == 0 else s - MAX_X
            ns_right = s if x == (MAX_X - 1) else s + 1
            ns_down = s if y == (MAX_Y - 1) else s + MAX_X
            ns_left = s if x == 0 else s - 1
            
            next_states = {
                UP: ns_up,
                RIGHT: ns_right,
                DOWN: ns_down,
                LEFT: ns_left
            }

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                for action in range(nA):
                    ns_intended = next_states[action]
                    
                    if transition_prob == 1.0:
                        # Determinístico
                        P[s][action] = [(1.0, ns_intended, reward, is_done(ns_intended))]
                    else:
                        # Estocástico: prob p vai para onde quer, (1-p)/2 para cada ortogonal
                        transitions = []
                        slip_prob = (1.0 - transition_prob) / 2.0
                        
                        # Ação pretendida
                        transitions.append((transition_prob, ns_intended, reward, is_done(ns_intended)))
                        
                        # Escorregamento para direções ortogonais
                        for orth_action in orthogonal[action]:
                            ns_slip = next_states[orth_action]
                            transitions.append((slip_prob, ns_slip, reward, is_done(ns_slip)))
                        
                        P[s][action] = transitions

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout

         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
