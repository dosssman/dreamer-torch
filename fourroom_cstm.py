#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Based on https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/fourrooms.py
    Provides additinal configurations for the MiniGrid-FourRoom environment
"""

from gym_minigrid.minigrid import * 
# from gym_minigrid.register import register
from gym.envs.registration import register
from gym_minigrid.envs.fourrooms import FourRoomsEnv

class CstmFourRoomEnv(MiniGridEnv):
    def __init__(self, grid_size=15, agent_pos=None, goal_pos=None):
        self.grid_size = grid_size
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=grid_size, max_steps=100)
    
    def _gen_grid(self, width, height):
        # customization: a holder for some important 
        self.env_data = {}
        env_data = self.env_data

        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        env_data["room_w"], env_data["room_h"] = \
            room_w, room_h
        env_data["door_positions"] = []

        # For each row of rooms
        room_idx = 0
        env_data["rooms"] = []
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)
                    env_data["door_positions"].append(pos)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)
                    env_data["door_positions"].append(pos)
                
                # add absolute coordinate delimiting each room
                # the reference is the upper-left corner of coord (0,0)
                env_data["rooms"].append({
                    "idx": room_idx,
                    "isdoor": False,
                    "x_boundaries": (xL+1,xR-1),
                    "y_boundaries": (yT+1,yB-1)
                })
                room_idx += 1

        # For convenience, we also add each door as a room of size 0
        for xL, yT in env_data["door_positions"]:
            env_data["rooms"].append({
                    "idx": room_idx,
                    "isdoor": True,
                    "x_boundaries": (xL,xL),
                    "y_boundaries": (yT,yT)
            })
            room_idx += 1
        
        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
            env_data["goal_pos"] = goal.init_pos
        else:
            # NOTE: separating the randomization of the goal generation
            # and that of the maze generation by overriding the self.place_obj
            goal_pos = self.place_obj(
                Goal(),
                reject_fn=lambda y, x: self._is_in_agent_room(self.agent_pos, x) or 
                    self._is_within_range(self.agent_pos, x) or 
                    self._is_at_door_pos(x)
            )
            env_data["goal_pos"] = goal_pos


        self.env_data["n_rooms"] = len(self.env_data["rooms"])
        self.visited_doors_pos = []

        # computation of the factored_state_shape:
        # 8: room number + rel_x size + rel_y size + 4: agent directions
        # then only the three terms above for the goal information representation
        self.env_data["factored_state_shape"] = \
            8 + (room_w - 1) + (room_h -1) + 4 + \
            8 + (room_w - 1) + (room_h -1)
        # Below: when not using one hot encoded state representation
        # self.env_data["factored_state_shape"] = 7

        self.mission = 'Reach the goal'

    def _detect_room(self, pos):
        """
            Return the index of the big room the agent is located in.
            This is used to make sure that the goal is not generated in the 
            same big room as the agent, to make tasks more challenging.
        """
        for room_data_dict in self.env_data["rooms"]:
            room_idx, _, room_x_bnd, room_y_bnd = room_data_dict.values()
            # Note: strictly inferior becuase of the walls that surround the rooms
            # where neither the agents nor the goals should be.
            if room_x_bnd[0] <= pos[0] <= room_x_bnd[1] and \
               room_y_bnd[0] <= pos[1] <= room_y_bnd[1]:
               return room_idx
        
        return -1 # This is bad
    
    def _is_in_agent_room(self, ref_pos, pos):
        """
            Informs us whether or not the 'pos' is located in the same room
            as the 'ref+pos'.
            For example, when we want to make sure the agent is not in the 
            same room as the goal
        """
        ref_pos_room = self._detect_room(ref_pos)
        pos_room = self._detect_room(pos)
        assert ref_pos_room != -1, "Failed to detect the room of the reference object"
        assert pos_room != -1, "Failed to detect the room of the object"
        return ref_pos_room == pos_room
    
    def _is_within_range(self, ref_pos, pos, threshold=5):
        """
            Measures the distance in cells between the ref pos and the pos
            This is a helper to make sure that when generating goals and agent positions
            randomly, they are separated by at least 'range' cells, thus making tasks more challenging
        """
        # NOTE: add an assert to make sure that threshld is lesser than
        # min(w,h)where w,h are the maximal width / height of the maze
        # print("ref pos type: ", type(ref_pos))
        # print("pos type: ", type(pos))
        # print(ref_pos, pos, int( math.sqrt(np.sum((ref_pos - pos) ** 2))))

        threshold = min(int(math.sqrt(self.grid_size)), threshold)
        return int( math.sqrt(np.sum((ref_pos - pos) ** 2))) < threshold
    
    def _is_at_door_pos(self, pos):
        """
            Checks if the 'pos' corresponds to one of the 'door' 
            in the maze.
            Example: we don't want the goal or the agent to start at in a door,
            so this helper function is used for the cheks.
        """
        for room_data_dict in self.env_data["rooms"]:
            if room_data_dict["isdoor"] == True:
                xL, yB = \
                    room_data_dict["x_boundaries"][0], \
                    room_data_dict["y_boundaries"][0]
                if pos[0] == xL and pos[1] == yB:
                    return True

        return False

    def _get_room_index(self, pos, isdoor=False):
        for room_data_dict in self.env_data["rooms"]:
            room_idx, room_is_door, room_x_bnd, room_y_bnd = room_data_dict.values()
            # Note: strictly inferior becuase of the walls that surround the rooms
            # where neither the agents nor the goals should be.
            if room_x_bnd[0] <= pos[0] <= room_x_bnd[1] and \
               room_y_bnd[0] <= pos[1] <= room_y_bnd[1]:
                if isdoor:
                    return room_idx, room_is_door
                return room_idx
        
        if isdoor:
            return -1, False
        
        return -1

    # def step(self, action):
    #     obs, reward, done, info = MiniGridEnv.step(self, action)
    #     return obs, reward, done, info
    
    # Over the mother class to implement reward on door reached.
    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        # Compute bonus reward if the agent has reached a door.
        # Do it only once.
        # agent_room, agent_room_isdoor = self._get_room_index(self.agent_pos, isdoor=True)
        # assert agent_room != -1, "Broken door reward logic"

        # if agent_room_isdoor and agent_room not in self.visited_doors_pos:
        #     reward += .2
        #     self.visited_doors_pos.append(agent_room)

        return obs, reward, done, {}

    # Override of mother class fns
    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
        ):
        """
        Place an object at an empty position in the grid
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            # NOTE: the position is generated independently of the env's
            # random number generator fixed by the seed
            
            pos = np.array((
                np.random.randint(top[0], min(top[0] + size[0], self.grid.width)),
                np.random.randint(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2 , self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask
    
    def gen_factored_state_representation(self):
        """
            Generates a factored state representation based on human
            intuition for the FourRoom task.
        """

        # Get information related to the goal
        goal_pos = self.env_data["goal_pos"]
        # goal_reached = (self.agent_pos == goal_pos).all()
        goal_room_idx = self._detect_room(goal_pos)
        goal_room_data = self.env_data["rooms"][goal_room_idx]
        room_xL, room_yT = \
            goal_room_data["x_boundaries"][0], \
            goal_room_data["y_boundaries"][0]
        goal_rel_x, goal_rel_y = \
            goal_pos[0] - room_xL, \
            goal_pos[1] - room_yT

        current_room_idx = self._detect_room(self.agent_pos)
        room_data = self.env_data["rooms"][current_room_idx]
        room_xL, room_yT = \
            room_data["x_boundaries"][0], \
            room_data["y_boundaries"][0]

        current_rel_x = self.agent_pos[0] - room_xL
        current_rel_y = self.agent_pos[1] - room_yT
        agent_direction = self.agent_dir

        # print("goal reached: ", goal_reached)
        # print("goal room idx: :", goal_room_idx)
        # print("goal_rel_x, rel_y: ", goal_rel_x, goal_rel_y)
        # print("current room idx: ", current_room_idx)
        # print("agent rel_x, rel_y: ", current_rel_x, current_rel_y)
        # print("agent direction: ", agent_direction)

        # print(np.eye(self.env_data["n_rooms"])[goal_room_idx]) # r: goal's room
        # print(np.eye(self.env_data["room_w"] - 1)[goal_rel_x]) # goal rel x coord in the room
        # print(np.eye(self.env_data["room_h"] - 1)[goal_rel_y])
        # print("# DBG: custom state representation")
        # print("Agent room: ", current_room_idx, " -> ", np.eye(self.env_data["n_rooms"])[current_room_idx])
        # print("Agent rel x: ", current_rel_x, " -> ", np.eye(self.env_data["room_w"] - 1)[current_rel_x])
        # print("Agent rel y: ", current_rel_y, " -> ", np.eye(self.env_data["room_h"] - 1)[current_rel_y])
        # print("Agent dir: ", agent_direction, " -> ", np.eye(4)[agent_direction])
        # print("Goal room: ", goal_room_idx, " -> ", np.eye(self.env_data["n_rooms"])[goal_room_idx])
        # print("Goal rel x: ", goal_rel_x, " -> ", np.eye(self.env_data["room_w"] - 1)[goal_rel_x])
        # print("Goal rel y: ", goal_rel_y, " -> ", np.eye(self.env_data["room_h"] - 1)[goal_rel_y])

        hl_state = np.concatenate([
            np.eye(self.env_data["n_rooms"])[current_room_idx], # r: room
            np.eye(self.env_data["n_rooms"])[goal_room_idx], #  goal's room
        ])

        # return np.array([
        #     current_room_idx,
        #     goal_room_idx,
        #     current_rel_x,
        #     current_rel_y,
        #     agent_direction,
        #     goal_rel_x,
        #     goal_rel_y
        # ]), {}

        # Changed the order of the one hot vector that represent each
        # component of the state representation
        # return np.concatenate([
        #     np.eye(self.env_data["n_rooms"])[current_room_idx], # agent room index
        #     np.eye(self.env_data["room_w"] - 1)[current_rel_x], # relative x coord in the room
        #     np.eye(self.env_data["room_h"] - 1)[current_rel_y], # relative y coord in the room
        #     np.eye(4)[agent_direction], # d: agent's direction

        #     np.eye(self.env_data["n_rooms"])[goal_room_idx], # goal room index
        #     np.eye(self.env_data["room_w"] - 1)[goal_rel_x], # goal rel x coord in the room
        #     np.eye(self.env_data["room_h"] - 1)[goal_rel_y], # goal rel y coord in the room
        # ]), hl_state

        return np.concatenate([
            hl_state,
            np.eye(self.env_data["room_w"] - 1)[current_rel_x], # relative x coord in the room
            np.eye(self.env_data["room_h"] - 1)[current_rel_y], # relative y coord in the room
            np.eye(4)[agent_direction], # d: agent's direction

            np.eye(self.env_data["room_w"] - 1)[goal_rel_x], # goal rel x coord in the room
            np.eye(self.env_data["room_h"] - 1)[goal_rel_y], # goal rel y coord in the room
        ]), hl_state
    
    def get_hl_state(self):
        return self.gen_factored_state_representation()[1]

class CstmFourRoomEnvPOMDP(MiniGridEnv):
    def __init__(self, grid_size=15, agent_pos=None, goal_pos=None):
        self.grid_size = grid_size
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=grid_size, max_steps=100)
    
    def _gen_grid(self, width, height):
        # customization: a holder for some important 
        self.env_data = {}
        env_data = self.env_data

        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        env_data["room_w"], env_data["room_h"] = \
            room_w, room_h
        env_data["door_positions"] = []

        # For each row of rooms
        room_idx = 0
        env_data["rooms"] = []
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)
                    env_data["door_positions"].append(pos)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)
                    env_data["door_positions"].append(pos)
                
                # add absolute coordinate delimiting each room
                # the reference is the upper-left corner of coord (0,0)
                env_data["rooms"].append({
                    "idx": room_idx,
                    "isdoor": False,
                    "x_boundaries": (xL+1,xR-1),
                    "y_boundaries": (yT+1,yB-1)
                })
                room_idx += 1

        # For convenience, we also add each door as a room of size 0
        for xL, yT in env_data["door_positions"]:
            env_data["rooms"].append({
                    "idx": room_idx,
                    "isdoor": True,
                    "x_boundaries": (xL,xL),
                    "y_boundaries": (yT,yT)
            })
            room_idx += 1
        
        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
            env_data["goal_pos"] = goal.init_pos
        else:
            # NOTE: separating the randomization of the goal generation
            # and that of the maze generation by overriding the self.place_obj
            goal_pos = self.place_obj(
                Goal(),
                reject_fn=lambda y, x: self._is_in_agent_room(self.agent_pos, x) or 
                    self._is_within_range(self.agent_pos, x) or 
                    self._is_at_door_pos(x)
            )
            env_data["goal_pos"] = goal_pos


        self.env_data["n_rooms"] = len(self.env_data["rooms"])
        self.visited_doors_pos = []

        # computation of the factored_state_shape:
        # 8: room number + rel_x size + rel_y size + 4: agent directions
        # then only the three terms above for the goal information representation
        self.env_data["factored_state_shape"] = \
            8 + (room_w - 1) + (room_h -1) + 4 + \
            2 + room_w + room_h # for the POMDP case
        # Below: when not using one hot encoded state representation
        # self.env_data["factored_state_shape"] = 7

        self.mission = 'Reach the goal'

    def _detect_room(self, pos):
        """
            Return the index of the big room the agent is located in.
            This is used to make sure that the goal is not generated in the 
            same big room as the agent, to make tasks more challenging.
        """
        for room_data_dict in self.env_data["rooms"]:
            room_idx, _, room_x_bnd, room_y_bnd = room_data_dict.values()
            # Note: strictly inferior becuase of the walls that surround the rooms
            # where neither the agents nor the goals should be.
            if room_x_bnd[0] <= pos[0] <= room_x_bnd[1] and \
               room_y_bnd[0] <= pos[1] <= room_y_bnd[1]:
               return room_idx
        
        return -1 # This is bad
    
    def _is_in_agent_room(self, ref_pos, pos):
        """
            Informs us whether or not the 'pos' is located in the same room
            as the 'ref+pos'.
            For example, when we want to make sure the agent is not in the 
            same room as the goal
        """
        ref_pos_room = self._detect_room(ref_pos)
        pos_room = self._detect_room(pos)
        assert ref_pos_room != -1, "Failed to detect the room of the reference object"
        assert pos_room != -1, "Failed to detect the room of the object"
        return ref_pos_room == pos_room
    
    def _is_within_range(self, ref_pos, pos, threshold=5):
        """
            Measures the distance in cells between the ref pos and the pos
            This is a helper to make sure that when generating goals and agent positions
            randomly, they are separated by at least 'range' cells, thus making tasks more challenging
        """
        # NOTE: add an assert to make sure that threshld is lesser than
        # min(w,h)where w,h are the maximal width / height of the maze
        # print("ref pos type: ", type(ref_pos))
        # print("pos type: ", type(pos))
        # print(ref_pos, pos, int( math.sqrt(np.sum((ref_pos - pos) ** 2))))

        threshold = min(int(math.sqrt(self.grid_size)), threshold)
        return int( math.sqrt(np.sum((ref_pos - pos) ** 2))) < threshold
    
    def _is_at_door_pos(self, pos):
        """
            Checks if the 'pos' corresponds to one of the 'door' 
            in the maze.
            Example: we don't want the goal or the agent to start at in a door,
            so this helper function is used for the cheks.
        """
        for room_data_dict in self.env_data["rooms"]:
            if room_data_dict["isdoor"] == True:
                xL, yB = \
                    room_data_dict["x_boundaries"][0], \
                    room_data_dict["y_boundaries"][0]
                if pos[0] == xL and pos[1] == yB:
                    return True

        return False

    def _get_room_index(self, pos, isdoor=False):
        for room_data_dict in self.env_data["rooms"]:
            room_idx, room_is_door, room_x_bnd, room_y_bnd = room_data_dict.values()
            # Note: strictly inferior becuase of the walls that surround the rooms
            # where neither the agents nor the goals should be.
            if room_x_bnd[0] <= pos[0] <= room_x_bnd[1] and \
               room_y_bnd[0] <= pos[1] <= room_y_bnd[1]:
                if isdoor:
                    return room_idx, room_is_door
                return room_idx
        
        if isdoor:
            return -1, False
        
        return -1

    # def step(self, action):
    #     obs, reward, done, info = MiniGridEnv.step(self, action)
    #     return obs, reward, done, info
    
    # Over the mother class to implement reward on door reached.
    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        # Compute bonus reward if the agent has reached a door.
        # Do it only once.
        # agent_room, agent_room_isdoor = self._get_room_index(self.agent_pos, isdoor=True)
        # assert agent_room != -1, "Broken door reward logic"

        # if agent_room_isdoor and agent_room not in self.visited_doors_pos:
        #     reward += .2
        #     self.visited_doors_pos.append(agent_room)

        return obs, reward, done, {}

    # Override of mother class fns
    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
        ):
        """
        Place an object at an empty position in the grid
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            # NOTE: the position is generated independently of the env's
            # random number generator fixed by the seed
            
            pos = np.array((
                np.random.randint(top[0], min(top[0] + size[0], self.grid.width)),
                np.random.randint(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2 , self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask
    
    def gen_factored_state_representation(self):
        """
            Generates a factored state representation based on human
            intuition for the FourRoom task.
            For the POMDP case, the correct information about the goal
            is only returned when the agent is in the same room as the goal.
        """

        # Get information related to the goal
        goal_pos = self.env_data["goal_pos"]
        # goal_reached = (self.agent_pos == goal_pos).all()
        goal_room_idx = self._detect_room(goal_pos)
        goal_room_data = self.env_data["rooms"][goal_room_idx]
        room_xL, room_yT = \
            goal_room_data["x_boundaries"][0], \
            goal_room_data["y_boundaries"][0]
        goal_rel_x, goal_rel_y = \
            goal_pos[0] - room_xL, \
            goal_pos[1] - room_yT

        current_room_idx = self._detect_room(self.agent_pos)
        room_data = self.env_data["rooms"][current_room_idx]
        room_xL, room_yT = \
            room_data["x_boundaries"][0], \
            room_data["y_boundaries"][0]

        current_rel_x = self.agent_pos[0] - room_xL
        current_rel_y = self.agent_pos[1] - room_yT
        agent_direction = self.agent_dir

        # print("goal reached: ", goal_reached)
        # print("goal room idx: :", goal_room_idx)
        # print("goal_rel_x, rel_y: ", goal_rel_x, goal_rel_y)
        # print("current room idx: ", current_room_idx)
        # print("agent rel_x, rel_y: ", current_rel_x, current_rel_y)
        # print("agent direction: ", agent_direction)

        # print(np.eye(self.env_data["n_rooms"])[goal_room_idx]) # r: goal's room
        # print(np.eye(self.env_data["room_w"] - 1)[goal_rel_x]) # goal rel x coord in the room
        # print(np.eye(self.env_data["room_h"] - 1)[goal_rel_y])
        # print("# DBG: custom state representation")
        # print("Agent room: ", current_room_idx, " -> ", np.eye(self.env_data["n_rooms"])[current_room_idx])
        # print("Agent rel x: ", current_rel_x, " -> ", np.eye(self.env_data["room_w"] - 1)[current_rel_x])
        # print("Agent rel y: ", current_rel_y, " -> ", np.eye(self.env_data["room_h"] - 1)[current_rel_y])
        # print("Agent dir: ", agent_direction, " -> ", np.eye(4)[agent_direction])
        # print("Goal room: ", goal_room_idx, " -> ", np.eye(self.env_data["n_rooms"])[goal_room_idx])
        # print("Goal rel x: ", goal_rel_x, " -> ", np.eye(self.env_data["room_w"] - 1)[goal_rel_x])
        # print("Goal rel y: ", goal_rel_y, " -> ", np.eye(self.env_data["room_h"] - 1)[goal_rel_y])

        agent_is_in_goal_room = int(current_room_idx == goal_room_idx)
        hl_state = np.concatenate([
            np.eye(self.env_data["n_rooms"])[current_room_idx], # r: room
            np.eye(2)[agent_is_in_goal_room], #  goal's room
        ])

        # return np.array([
        #     current_room_idx,
        #     goal_room_idx,
        #     current_rel_x,
        #     current_rel_y,
        #     agent_direction,
        #     goal_rel_x,
        #     goal_rel_y
        # ]), {}

        # Changed the order of the one hot vector that represent each
        # component of the state representation
        # return np.concatenate([
        #     np.eye(self.env_data["n_rooms"])[current_room_idx], # agent room index
        #     np.eye(self.env_data["room_w"] - 1)[current_rel_x], # relative x coord in the room
        #     np.eye(self.env_data["room_h"] - 1)[current_rel_y], # relative y coord in the room
        #     np.eye(4)[agent_direction], # d: agent's direction

        #     np.eye(self.env_data["n_rooms"])[goal_room_idx], # goal room index
        #     np.eye(self.env_data["room_w"] - 1)[goal_rel_x], # goal rel x coord in the room
        #     np.eye(self.env_data["room_h"] - 1)[goal_rel_y], # goal rel y coord in the room
        # ]), hl_state

        # When the agent is not in the goal's room, we return
        aug_goal_rel_x, aug_goal_rel_y = self.env_data["room_w"]-1, self.env_data["room_h"]-1
        if agent_is_in_goal_room:
            aug_goal_rel_x = goal_rel_x
            aug_goal_rel_y = goal_rel_y
        
        return np.concatenate([
            hl_state,
            np.eye(self.env_data["room_w"] - 1)[current_rel_x], # relative x coord in the room
            np.eye(self.env_data["room_h"] - 1)[current_rel_y], # relative y coord in the room
            np.eye(4)[agent_direction], # d: agent's direction

            np.eye(self.env_data["room_w"])[aug_goal_rel_x], # goal rel x coord in the room
            np.eye(self.env_data["room_h"])[aug_goal_rel_y], # goal rel y coord in the room
        ]), hl_state
    
    def get_hl_state(self):
        return self.gen_factored_state_representation()[1]

# Variant of the Custom Four Room envs that reward the agent for eaching 
# any one of the door, and termiate the episode upon success
class CstmFourRoomEnvDoorReachingPretrain(MiniGridEnv):
    def __init__(self, grid_size=15, agent_pos=None, goal_pos=None):
        self.grid_size = grid_size
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=grid_size, max_steps=100)
    
    def _gen_grid(self, width, height):
        # customization: a holder for some important 
        self.env_data = {}
        env_data = self.env_data

        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        env_data["room_w"], env_data["room_h"] = \
            room_w, room_h
        env_data["door_positions"] = []

        # For each row of rooms
        room_idx = 0
        env_data["rooms"] = []
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)
                    env_data["door_positions"].append(pos)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)
                    env_data["door_positions"].append(pos)
                
                # add absolute coordinate delimiting each room
                # the reference is the upper-left corner of coord (0,0)
                env_data["rooms"].append({
                    "idx": room_idx,
                    "isdoor": False,
                    "x_boundaries": (xL+1,xR-1),
                    "y_boundaries": (yT+1,yB-1)
                })
                room_idx += 1

        # For convenience, we also add each door as a room of size 0
        for xL, yT in env_data["door_positions"]:
            env_data["rooms"].append({
                    "idx": room_idx,
                    "isdoor": True,
                    "x_boundaries": (xL,xL),
                    "y_boundaries": (yT,yT)
            })
            room_idx += 1
        
        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
            env_data["goal_pos"] = goal.init_pos
        else:
            # NOTE: separating the randomization of the goal generation
            # and that of the maze generation by overriding the self.place_obj
            goal_pos = self.place_obj(
                Goal(),
                reject_fn=lambda y, x: self._is_in_agent_room(self.agent_pos, x) or 
                    self._is_within_range(self.agent_pos, x) or 
                    self._is_at_door_pos(x)
            )
            env_data["goal_pos"] = goal_pos


        self.env_data["n_rooms"] = len(self.env_data["rooms"])
        self.visited_doors_pos = []

        # computation of the factored_state_shape:
        # 8: room number + rel_x size + rel_y size + 4: agent directions
        # then only the three terms above for the goal information representation
        self.env_data["factored_state_shape"] = \
            8 + (room_w - 1) + (room_h -1) + 4 + \
            8 + (room_w - 1) + (room_h -1)
        # Below: when not using one hot encoded state representation
        # self.env_data["factored_state_shape"] = 7

        self.mission = 'Reach the goal'

    def _detect_room(self, pos):
        """
            Return the index of the big room the agent is located in.
            This is used to make sure that the goal is not generated in the 
            same big room as the agent, to make tasks more challenging.
        """
        for room_data_dict in self.env_data["rooms"]:
            room_idx, _, room_x_bnd, room_y_bnd = room_data_dict.values()
            # Note: strictly inferior becuase of the walls that surround the rooms
            # where neither the agents nor the goals should be.
            if room_x_bnd[0] <= pos[0] <= room_x_bnd[1] and \
               room_y_bnd[0] <= pos[1] <= room_y_bnd[1]:
               return room_idx
        
        return -1 # This is bad
    
    def _is_in_agent_room(self, ref_pos, pos):
        """
            Informs us whether or not the 'pos' is located in the same room
            as the 'ref+pos'.
            For example, when we want to make sure the agent is not in the 
            same room as the goal
        """
        ref_pos_room = self._detect_room(ref_pos)
        pos_room = self._detect_room(pos)
        assert ref_pos_room != -1, "Failed to detect the room of the reference object"
        assert pos_room != -1, "Failed to detect the room of the object"
        return ref_pos_room == pos_room
    
    def _is_within_range(self, ref_pos, pos, threshold=5):
        """
            Measures the distance in cells between the ref pos and the pos
            This is a helper to make sure that when generating goals and agent positions
            randomly, they are separated by at least 'range' cells, thus making tasks more challenging
        """
        # NOTE: add an assert to make sure that threshld is lesser than
        # min(w,h)where w,h are the maximal width / height of the maze
        # print("ref pos type: ", type(ref_pos))
        # print("pos type: ", type(pos))
        # print(ref_pos, pos, int( math.sqrt(np.sum((ref_pos - pos) ** 2))))

        threshold = min(int(math.sqrt(self.grid_size)), threshold)
        return int( math.sqrt(np.sum((ref_pos - pos) ** 2))) < threshold
    
    def _is_at_door_pos(self, pos):
        """
            Checks if the 'pos' corresponds to one of the 'door' 
            in the maze.
            Example: we don't want the goal or the agent to start at in a door,
            so this helper function is used for the cheks.
        """
        for room_data_dict in self.env_data["rooms"]:
            if room_data_dict["isdoor"] == True:
                xL, yB = \
                    room_data_dict["x_boundaries"][0], \
                    room_data_dict["y_boundaries"][0]
                if pos[0] == xL and pos[1] == yB:
                    return True

        return False

    def _get_room_index(self, pos, isdoor=False):
        for room_data_dict in self.env_data["rooms"]:
            room_idx, room_is_door, room_x_bnd, room_y_bnd = room_data_dict.values()
            # Note: strictly inferior becuase of the walls that surround the rooms
            # where neither the agents nor the goals should be.
            if room_x_bnd[0] <= pos[0] <= room_x_bnd[1] and \
               room_y_bnd[0] <= pos[1] <= room_y_bnd[1]:
                if isdoor:
                    return room_idx, room_is_door
                return room_idx
        
        if isdoor:
            return -1, False
        
        return -1

    # def step(self, action):
    #     obs, reward, done, info = MiniGridEnv.step(self, action)
    #     return obs, reward, done, info
    
    # Over the mother class to implement reward on door reached.
    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        # Compute bonus reward if the agent has reached a door.
        # Do it only once.
        agent_room, agent_room_isdoor = self._get_room_index(self.agent_pos, isdoor=True)
        assert agent_room != -1, "Broken door reward logic"

        if agent_room_isdoor and agent_room not in self.visited_doors_pos:
            reward = self._reward()
            self.visited_doors_pos.append(agent_room)
            done = True

        return obs, reward, done, {}

    # Override of mother class fns
    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
        ):
        """
        Place an object at an empty position in the grid
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            # NOTE: the position is generated independently of the env's
            # random number generator fixed by the seed
            
            pos = np.array((
                np.random.randint(top[0], min(top[0] + size[0], self.grid.width)),
                np.random.randint(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2 , self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask
    
    def gen_factored_state_representation(self):
        """
            Generates a factored state representation based on human
            intuition for the FourRoom task.
        """

        # Get information related to the goal
        goal_pos = self.env_data["goal_pos"]
        # goal_reached = (self.agent_pos == goal_pos).all()
        goal_room_idx = self._detect_room(goal_pos)
        goal_room_data = self.env_data["rooms"][goal_room_idx]
        room_xL, room_yT = \
            goal_room_data["x_boundaries"][0], \
            goal_room_data["y_boundaries"][0]
        goal_rel_x, goal_rel_y = \
            goal_pos[0] - room_xL, \
            goal_pos[1] - room_yT

        current_room_idx = self._detect_room(self.agent_pos)
        room_data = self.env_data["rooms"][current_room_idx]
        room_xL, room_yT = \
            room_data["x_boundaries"][0], \
            room_data["y_boundaries"][0]

        current_rel_x = self.agent_pos[0] - room_xL
        current_rel_y = self.agent_pos[1] - room_yT
        agent_direction = self.agent_dir

        # print("goal reached: ", goal_reached)
        # print("goal room idx: :", goal_room_idx)
        # print("goal_rel_x, rel_y: ", goal_rel_x, goal_rel_y)
        # print("current room idx: ", current_room_idx)
        # print("agent rel_x, rel_y: ", current_rel_x, current_rel_y)
        # print("agent direction: ", agent_direction)

        # print(np.eye(self.env_data["n_rooms"])[goal_room_idx]) # r: goal's room
        # print(np.eye(self.env_data["room_w"] - 1)[goal_rel_x]) # goal rel x coord in the room
        # print(np.eye(self.env_data["room_h"] - 1)[goal_rel_y])
        # print("# DBG: custom state representation")
        # print("Agent room: ", current_room_idx, " -> ", np.eye(self.env_data["n_rooms"])[current_room_idx])
        # print("Agent rel x: ", current_rel_x, " -> ", np.eye(self.env_data["room_w"] - 1)[current_rel_x])
        # print("Agent rel y: ", current_rel_y, " -> ", np.eye(self.env_data["room_h"] - 1)[current_rel_y])
        # print("Agent dir: ", agent_direction, " -> ", np.eye(4)[agent_direction])
        # print("Goal room: ", goal_room_idx, " -> ", np.eye(self.env_data["n_rooms"])[goal_room_idx])
        # print("Goal rel x: ", goal_rel_x, " -> ", np.eye(self.env_data["room_w"] - 1)[goal_rel_x])
        # print("Goal rel y: ", goal_rel_y, " -> ", np.eye(self.env_data["room_h"] - 1)[goal_rel_y])

        hl_state = np.concatenate([
            np.eye(self.env_data["n_rooms"])[current_room_idx], # r: room
            np.eye(self.env_data["n_rooms"])[goal_room_idx], #  goal's room
        ])

        # return np.array([
        #     current_room_idx,
        #     goal_room_idx,
        #     current_rel_x,
        #     current_rel_y,
        #     agent_direction,
        #     goal_rel_x,
        #     goal_rel_y
        # ]), {}

        # Changed the order of the one hot vector that represent each
        # component of the state representation
        # return np.concatenate([
        #     np.eye(self.env_data["n_rooms"])[current_room_idx], # agent room index
        #     np.eye(self.env_data["room_w"] - 1)[current_rel_x], # relative x coord in the room
        #     np.eye(self.env_data["room_h"] - 1)[current_rel_y], # relative y coord in the room
        #     np.eye(4)[agent_direction], # d: agent's direction

        #     np.eye(self.env_data["n_rooms"])[goal_room_idx], # goal room index
        #     np.eye(self.env_data["room_w"] - 1)[goal_rel_x], # goal rel x coord in the room
        #     np.eye(self.env_data["room_h"] - 1)[goal_rel_y], # goal rel y coord in the room
        # ]), hl_state

        return np.concatenate([
            hl_state,
            np.eye(self.env_data["room_w"] - 1)[current_rel_x], # relative x coord in the room
            np.eye(self.env_data["room_h"] - 1)[current_rel_y], # relative y coord in the room
            np.eye(4)[agent_direction], # d: agent's direction

            np.eye(self.env_data["room_w"] - 1)[goal_rel_x], # goal rel x coord in the room
            np.eye(self.env_data["room_h"] - 1)[goal_rel_y], # goal rel y coord in the room
        ]), hl_state
    
    def get_hl_state(self):
        return self.gen_factored_state_representation()[1]

for grid_size in range(7,20):
    # MDP case
    register(
        id=f'MiniGrid-FourRooms-Size{grid_size}-v0',
        entry_point='fourroom_cstm:CstmFourRoomEnv',
        kwargs={
            "grid_size": grid_size
        }
    )
    # # MDP case, Door Reaching pretraining environment
    # register(
    #     id=f'MiniGrid-FourRooms-Size{grid_size}-DoorReaching-v0',
    #     entry_point='hwm.gym_minigrid_2.fourroom_cstm:CstmFourRoomEnvDoorReachingPretrain',
    #     kwargs={
    #         "grid_size": grid_size
    #     }
    # )
    # # POMDP case
    # register(
    #     id=f'MiniGrid-FourRooms-Size{grid_size}-POMDP-v0',
    #     entry_point='hwm.gym_minigrid_2.fourroom_cstm:CstmFourRoomEnvPOMDP',
    #     kwargs={
    #         "grid_size": grid_size
    #     }
    # )
