"""
1. State description
    The state is defined as tuple of 4 tuples. The first tuple is the position of the player (the coordinates that define the position).
    The second tuple is the representation of the first obstacle (the coordinates of the left end and the movement direction (L - left or R - right)).
    The third tuple is the representation of the second obstacle (the coordinates of the top right corner and the movement direction (UR - upper-right or LL - lower-left)).
    The forth tuple is the repesentation of the third obstacle (the coordinates of the lower end and the movement direction (D - down or U - up))
    example: state=((1,1), (2,2,"L"), (8,2,"UR"), (9,7,"D"))

2. Description of the transition functions
    First we check the position of the obstacles and we move them accordingly. We also check if we should change obstacle's direction due to reaching the end of the map.
    We check for possible player's moves and we move the player after we are sure that we won't hit any obstacle or we won't go over the table (the map).
    At the end of every transition, we check if we have reached the goal state (the house on the map).

3. Description of the heuristic function
    We calculate the heuristic function's value with the use of Euclidean distance equation. We calculate Euclidean distance between the current position and the house. 

"""

import sys
import bisect

infinity = float('inf')  # system's defined value for infinity

class Queue:
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)


def Stack():
    """A Last-In-First-Out Queue."""
    return []


class FIFOQueue(Queue):
    """A First-In-First-Out Queue."""

    def __init__(self):
        self.A = []
        self.start = 0

    def append(self, item):
        self.A.append(item)

    def __len__(self):
        return len(self.A) - self.start

    def extend(self, items):
        self.A.extend(items)

    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A) / 2:
            self.A = self.A[self.start:]
            self.start = 0
        return e

    def __contains__(self, item):
        return item in self.A[self.start:]


class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup. This structure will be most useful in informed searches"""

    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)

class Problem:
    """The abstract class for a formal problem.  You should subclass this and
    implement the method successor, and possibly __init__, goal_test, and
    path_cost. Then you will create instances of your subclass and solve them
    with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        """Given a state, return a dictionary of {action : state} pairs reachable
        from this state. If there are many successors, consider an iterator
        that yields the successors one at a time, rather than building them
        all at once. Iterators will work fine within the framework. Yielding is not supported in Python 2.7"""
        raise NotImplementedError

    def actions(self, state):
        """Given a state, return a list of all actions possible from that state"""
        raise NotImplementedError

    def result(self, state, action):
        """Given a state and action, return the resulting state"""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Implement this
        method if checking against a single self.goal is not enough."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get   up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        "Return a child node from this node"
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def solve(self):
        "Return the sequence of states to go from the root to this node."
        return [node.state for node in self.path()[0:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        x, result = self, []
        while x:
            result.append(x)
            x = x.parent
        return list(reversed(result))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


# ________________________________________________________________________________________________________
#Uninformed tree search

def tree_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue."""
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        print node.state
        if problem.goal_test(node.state):
            return node
        fringe.extend(node.expand(problem))
    return None


def breadth_first_tree_search(problem):
    "Search the shallowest nodes in the search tree first."
    return tree_search(problem, FIFOQueue())


def depth_first_tree_search(problem):
    "Search the deepest nodes in the search tree first."
    return tree_search(problem, Stack())


# ________________________________________________________________________________________________________
#Uninformed graph search

def graph_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    If two paths reach a state, only use the best one."""
    closed = {}
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state):
            return node
        if node.state not in closed:
            closed[node.state] = True
            fringe.extend(node.expand(problem))
    return None


def breadth_first_graph_search(problem):
    "Search the shallowest nodes in the search tree first."
    return graph_search(problem, FIFOQueue())


def depth_first_graph_search(problem):
    "Search the deepest nodes in the search tree first."
    return graph_search(problem, Stack())


def uniform_cost_search(problem):
    "Search the nodes in the search tree with lowest cost first."
    return graph_search(problem, PriorityQueue(lambda a, b: a.path_cost < b.path_cost))


def depth_limited_search(problem, limit=50):
    "depth first search with limited depth"

    def recursive_dls(node, problem, limit):
        "helper function for depth limited"
        cutoff_occurred = False
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            for successor in node.expand(problem):
                result = recursive_dls(successor, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result != None:
                    return result
        if cutoff_occurred:
            return 'cutoff'
        else:
            return None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):

    for depth in xrange(sys.maxint):
        result = depth_limited_search(problem, depth)
        if result is not 'cutoff':
            return result

def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if not memoized_fn.cache.has_key(args):
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}
    return memoized_fn


# ________________________________________________________________________________________________________
#Informed search
def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""

    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None


def greedy_best_first_graph_search(problem, h=None):
    "Greedy best-first search is accomplished by specifying f(n) = h(n)"
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, h)


def astar_search(problem, h=None):
    "A* search is best-first graph search with f(n) = g(n)+h(n)."
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))

def recursive_best_first_search(problem, h=None):
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0  # (The second value is immaterial)
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            # Order by lowest f value
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = infinity
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, infinity)
    return result

# check if the coordinates x and y are in the field
def mapRangeCheck(x, y):
    return ((x in range(5)) and (y in range(6))) or ((x in range(5,11)) and (y in range(11)))

class Obstacles(Problem):

    def __init__(self, initial, goal=(10,10)):
        self.initial = initial
        self.goal = goal

    # the heuristic function will calculate the Euclidean distance between the node and the house (the goal)
    def h(self, node):
        from math import sqrt
        return sqrt((self.goal[0] - node.state[0][0])**2 + (self.goal[1] - node.state[0][1])**2)

    def successor(self, state):
        """Given a state, return a dictionary of {action : state} pairs reachable
        from this state. If there are many successors, consider an iterator
        that yields the successors one at a time, rather than building them
        all at once. Iterators will work fine within the framework. Yielding is not supported in Python 2.7"""
        succ = {}
        obstacle1 = list(state[1])
        obstacle2 = list(state[2])
        obstacle3 = list(state[3])

        # movement of the first obstacle
        if(obstacle1[2] == "L"):
            if(obstacle1[1]==0):
                obstacle1[2] = "R"
                obstacle1[1] += 1
            else:
                obstacle1[1] -= 1
        else:
            if(obstacle1[1]==4):
                obstacle1[1] -= 1
                obstacle1[2] = "L"
            else:
                obstacle1[1] += 1

        # movement of the second obstacle
        if(obstacle2[2]=="UR"):
            if(obstacle2[1]==5):
                obstacle2[2]="LL"
                obstacle2[1] -= 1
                obstacle2[0] += 1
            else:
                obstacle2[0] -= 1
                obstacle2[1] += 1
        else:
            if(obstacle2[1]==1):
                obstacle2[2]="UR"
                obstacle2[1] += 1
                obstacle2[0] -= 1
            else:
                obstacle2[1] -= 1
                obstacle2[0] += 1

        # movement of third obstacle
        if(obstacle3[2]=="D"):
            if(obstacle3[0]==10):
                obstacle3[2] = "U"
                obstacle3[0] -= 1
            else:
                obstacle3[0] += 1
        else:
            if(obstacle3[0]==6):
                obstacle3[2] = "D"
                obstacle3[0] += 1
            else:
                obstacle3[0] -= 1
        
        # in this list we will place all 8 fields covered by the obstacles
        obstacles = []
        # fields of the first obstacle
        obstacles.append((obstacle1[0], obstacle1[1]))
        obstacles.append((obstacle1[0], obstacle1[1] + 1))
        # fields of the second obstacle
        obstacles.append((obstacle2[0], obstacle2[1]))
        obstacles.append((obstacle2[0], obstacle2[1]-1))
        obstacles.append((obstacle2[0]+1, obstacle2[1]-1))
        obstacles.append((obstacle2[0]+1, obstacle2[1]))
        # fields of the third obstacle
        obstacles.append((obstacle3[0]-1, obstacle3[1]))
        obstacles.append((obstacle3[0], obstacle3[1]))
        

        # now we analyze player's movements
        x1, y1 = state[0]
        # can the player go up?
        x = x1 - 1
        y = y1
        if mapRangeCheck(x, y) and ((x,y) not in obstacles):
            succ["UP"]=((x,y), tuple(obstacle1), tuple(obstacle2), tuple(obstacle3))
        # can the player go down?
        x = x1 + 1
        y = y1
        if mapRangeCheck(x, y) and ((x,y) not in obstacles):
            succ["DOWN"]=((x,y), tuple(obstacle1), tuple(obstacle2), tuple(obstacle3))
        # can the player go left?
        x = x1
        y = y1 - 1
        if mapRangeCheck(x, y) and ((x,y) not in obstacles):
            succ["LEFT"]=((x,y), tuple(obstacle1), tuple(obstacle2), tuple(obstacle3))
        # can the player go right?
        x = x1
        y = y1 + 1
        if mapRangeCheck(x, y) and ((x,y) not in obstacles):
            succ["RIGHT"]=((x,y), tuple(obstacle1), tuple(obstacle2), tuple(obstacle3))
        # can the player go to the upper-left field?
        x = x1 - 1
        y = y1 - 1
        if mapRangeCheck(x, y) and ((x,y) not in obstacles):
            succ["UPPER-LEFT"]=((x,y), tuple(obstacle1), tuple(obstacle2), tuple(obstacle3))
        # can the player go to the upper-right field?
        x = x1 - 1
        y = y1 + 1
        if mapRangeCheck(x, y) and ((x,y) not in obstacles):
            succ["UPPER-RIGHT"]=((x,y), tuple(obstacle1), tuple(obstacle2), tuple(obstacle3))
        # can the player go to the lower-left field?
        x = x1 + 1
        y = y1 - 1
        if mapRangeCheck(x, y) and ((x,y) not in obstacles):
            succ["LOWER-LEFT"]=((x,y), tuple(obstacle1), tuple(obstacle2), tuple(obstacle3))
        # can the player go to the lower-right field?
        x = x1 + 1
        y = y1 + 1
        if mapRangeCheck(x, y) and ((x,y) not in obstacles):
            succ["LOWER-RIGHT"]=((x,y), tuple(obstacle1), tuple(obstacle2), tuple(obstacle3))

        return succ;

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Implement this
        method if checking against a single self.goal is not enough."""
        return state[0] == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get   up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

if __name__=="__main__":
    x = int(input("Enter the x coordinate of the player's starting position: "))
    y = int(input("Enter the y coordinate of the player's starting position: "))

    if not mapRangeCheck(x, y):
        print("You entered invalid coordinates, the player can't be located on that position.")
    else:
        obstaclesProblem = Obstacles(((x,y),(2,2,"L"),(7,3,"UR"), (8,8,"D")))
        solution = astar_search(obstaclesProblem)
        if solution == None:
            print("The problem doesn't have a solution.")
        else:
            for item in solution.path():
                print("Action: " + str(item.action))
                print("State: " + str(item.state))